# -*- coding: utf-8 -*-
import os
import time
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier

num_species = 11255  # Nombre de toutes les classes uniques dans les données PO et PA.
seed = 42

class TrainDataset(Dataset):
    def __init__(self, data_dir, metadata, subset, transform=None):
        self.subset = subset  # Type de sous-ensemble (ex: train, val, test)
        self.transform = transform  # Transformation à appliquer aux échantillons
        self.data_dir = data_dir  # Répertoire contenant les fichiers de données
        self.metadata = metadata
        # Supprimer les lignes sans speciesId et réinitialiser les index
        self.metadata = self.metadata.dropna(subset=["speciesId"]).reset_index(drop=True)
        self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)  # Convertir speciesId en entier
        
        self.metadata = self.metadata.drop_duplicates(subset=["group_number","speciesId"]).reset_index(drop=True)
        
        self.species_dict = self.metadata.groupby('group_number')['speciesId'].apply(list).to_dict()
        self.surveyId_dict = self.metadata.groupby('group_number')['surveyId'].apply(list).to_dict()
        
        self.metadata = self.metadata.drop_duplicates(subset="group_number").reset_index(drop=True)

    def __len__(self):
        # Retourne le nombre de surveyId uniques
        return len(self.metadata)

    def __getitem__(self, idx):
        
        group_number = self.metadata.loc[idx,"group_number"]
        final_sample = torch.zeros((6,4,21))
        survey_ids = self.surveyId_dict.get(group_number, [])
        
        # Charger tous les tenseurs en une seule fois et les empiler
        samples = [torch.nan_to_num(torch.load(os.path.join(self.data_dir, f"GLC24-PO-{self.subset}-landsat_time_series_{survey_id}_cube.pt"))) 
                   for survey_id in survey_ids]

        # Si aucun tenseur n'a été chargé, renvoyer un tenseur nul
        if len(samples) == 0:
            final_sample = torch.zeros((6, 4, 21))
        else:
            stacked_samples = torch.stack(samples)
            final_sample = stacked_samples.mean(dim=0)
        
        species_ids = self.species_dict.get(group_number, [])  # Obtenir la liste des species IDs pour le group_number
        label = torch.zeros(num_species)  
        
        for species_id in species_ids:
            label[species_id] = 1

        # Assurer que l'échantillon est dans le bon format pour la transformation
        if isinstance(final_sample, torch.Tensor):
            final_sample = final_sample.permute(1, 2, 0)  # Changer la forme du tenseur de (C, H, W) à (H, W, C)
            final_sample = final_sample.numpy()  

        if self.transform:
            final_sample = self.transform(final_sample)

        return final_sample, label, group_number
    
class ModifiedResNet18(nn.Module):
    def __init__(self, num_species):
        super(ModifiedResNet18, self).__init__()

        self.norm_input = nn.LayerNorm([6,4,21])
        self.resnet18 = models.resnet18(weights=None)
        # We have to modify the first convolutional layer to accept 4 channels instead of 3
        self.resnet18.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()
        self.ln = nn.LayerNorm(1000)
        self.fc1 = nn.Linear(1000, 2056)
        self.fc2 = nn.Linear(2056, num_species)

    def forward(self, x):
        x = self.norm_input(x)
        x = self.resnet18(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12353"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Process {rank} initialized.")

def cleanup():
    destroy_process_group()

def main(rank, world_size, num_epochs, batch_size, save_every):
    print(f"Starting process {rank}")
    ddp_setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data_path = "/home/dakbarin/data/data/GEOLIFECLEF/GLC24-PO-train-landsat_time_series"
    train_metadata_path = "/home/dakbarin/data/data/GEOLIFECLEF/PO_grouped.csv"
    train_metadata = pd.read_csv(train_metadata_path)
    train_dataset = TrainDataset(train_data_path, train_metadata, subset="train", transform=transform)

    training, validation = random_split(train_dataset,
                                    [int(len(train_dataset)*0.85), len(train_dataset)-int(len(train_dataset)*0.85)],
                                    generator=torch.Generator().manual_seed(seed))

    # Create samplers
    train_sampler = DistributedSampler(training, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(validation, num_replicas=world_size, rank=rank)

    # Create data loaders
    train_loader = DataLoader(training, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    test_loader = DataLoader(validation, batch_size=batch_size, sampler=test_sampler, num_workers=0, pin_memory=True)
    
    print(len(train_loader))
    
    # Create model, optimizer, and scheduler
    model = ModifiedResNet18(num_species).to(device)
    state_dict = torch.load("/home/dakbarin/data/models/resnet18_with_bioclimatic_cubes_epoch_20.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=25)
    
    # Structure pour stocker les pertes
    losses = {
        "epoch": [],
        "train_loss": [],
        "val_loss": []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Training...")
        model.train()
        running_loss = 0.0
        train_sampler.set_epoch(epoch)
        barrier()  # Synchronisation des processus avant la boucle d'entraînement
        for data, targets, _ in tqdm(train_loader, desc=f"Training (Rank {rank})", leave=False):
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            out = model(data)
            
            loss = criterion(out, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - start_time
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}, Time: {epoch_time:.2f} seconds")

        # Validation phase
        model.eval()
        val_loss = 0.0
        barrier()  # Synchronisation des processus avant la validation
        with torch.no_grad():
            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Validation...")
            for data, targets, _ in tqdm(test_loader, desc=f"Validation (Rank {rank})", leave=False):
                data = data.to(device)
                targets = targets.to(device)

                out = model(data)
                
                loss = criterion(out, targets)
                
                val_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(test_loader)}")

            # Enregistrer les pertes dans la structure
            losses["epoch"].append(epoch + 1)
            losses["train_loss"].append(running_loss / len(train_loader))
            losses["val_loss"].append(val_loss / len(test_loader))

            # Sauvegarder les pertes dans un fichier CSV
            df = pd.DataFrame(losses)
            df.to_csv("training_losses.csv", index=False)

            # Save the model checkpoint
            if epoch % save_every == 0:
                torch.save(model.state_dict(), f"resnet18_with_bioclimatic_cubes_epoch_{epoch}_fine-tuned.pth")
    
    cleanup()

if __name__ == "__main__":
    total_epochs = 11
    save_every = 4
    batch_size = 256
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, batch_size, save_every), nprocs=world_size)
