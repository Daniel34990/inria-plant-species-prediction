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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch import optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

num_species = 11255  # Nombre de toutes les classes uniques dans les données PO et PA.
num_classes = 5
seed = 42

class TrainDataset(Dataset):
    def __init__(self, data_dir, metadata, subset, transform=None):
        self.subset = subset  # Type de sous-ensemble (ex: train, val, test)
        self.transform = transform  # Transformation à appliquer aux échantillons
        self.data_dir = data_dir  # Répertoire contenant les fichiers de données
        self.metadata = metadata  # Données de métadonnées
        # Supprimer les lignes sans speciesId et réinitialiser les index
        self.metadata = self.metadata.dropna(subset=["speciesId"]).reset_index(drop=True)
        self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)  # Convertir speciesId en entier
        
        # Colonnes des labels de présence
        self.label_columns = ['absence', 'presence_a_2_digit', 
                              'presence_a_3_digit', 'presence_a_4_digit', 'presence_seule']
        
        # Créer un dictionnaire des labels par surveyId
        self.label_dict = self.metadata.groupby('surveyId', group_keys=False).apply(
            lambda x: x.set_index('speciesId')[self.label_columns].to_dict(orient='index')
        ).to_dict()
        
        # Supprimer les doublons de surveyId et réinitialiser les index
        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)

    def __len__(self):
        # Retourne le nombre de surveyId uniques
        return len(self.metadata)

    def __getitem__(self, idx):
        survey_id = self.metadata.surveyId[idx]
        sample = torch.nan_to_num(torch.load(os.path.join(self.data_dir, f"GLC24-PA-{self.subset}-landsat-time-series_{survey_id}_cube.pt")))
        labels = self.label_dict.get(survey_id, {})
        label = torch.zeros((num_species, len(self.label_columns))) 
        label[:,0] = 1
        # Remplir le tenseur de labels avec les données de présence
        for species_id, presence_data in labels.items():
            if species_id < num_species:  # S'assurer que species_id est dans la plage valide
                label[species_id] = torch.tensor(list(presence_data.values()), dtype=torch.float32)

        # S'assurer que l'échantillon est au bon format pour la transformation
        if isinstance(sample, torch.Tensor):
            # Changer la forme du tenseur de (C, H, W) à (H, W, C)
            sample = sample.permute(1, 2, 0)  
            sample = sample.numpy()  

        # Appliquer la transformation si elle est définie
        if self.transform:
            sample = self.transform(sample)

        # Retourner l'échantillon, les labels et le surveyId
        return sample, label, survey_id

class ModifiedResNet18(nn.Module):
    def __init__(self, num_species):
        super(ModifiedResNet18, self).__init__()
        self.norm_input = nn.LayerNorm([6, 4, 21])
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.maxpool = nn.Identity()
        self.ln = nn.LayerNorm(1000)
        self.fc0 = nn.Linear(1000, 2056)
        self.fc1 = nn.Linear(2056, num_species*num_classes)

    def forward(self, x):
        x = self.norm_input(x)
        x = self.resnet18(x)
        x = self.ln(x)
        x = self.fc0(x)
        x = self.fc1(x)
        x = x.view(-1, num_species, num_classes)
        return x

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Process {rank} initialized.")

def cleanup():
    destroy_process_group()
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convertir les logits en probabilités avec softmax
        probas = F.softmax(inputs, dim=-1)
        
        # Récupérer la probabilité prédite pour la classe cible
        pt = probas.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculer la perte de base de l'entropie croisée
        log_pt = torch.log(pt)
        loss = -1 * (1 - pt) ** self.gamma * log_pt
        
        # Appliquer les poids de classe si fournis
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def main(rank, world_size, num_epochs, batch_size, save_every):
    print(f"Starting process {rank}")
    ddp_setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Load datasets
    train_data_path = "/home/dakbarin/data/data/GEOLIFECLEF/GLC24-PA-train-landsat-time-series"
    train_metadata_path = "/home/dakbarin/data/data/GEOLIFECLEF/metadata_with_classes.csv"
    train_metadata = pd.read_csv(train_metadata_path)
    train_dataset = TrainDataset(train_data_path, train_metadata, subset="train", transform=transforms.ToTensor())
    
    test_data_path = "/home/dakbarin/data/data/GEOLIFECLEF/GLC24-PA-train-landsat-time-series"
    test_metadata_path = "/home/dakbarin/data/data/GEOLIFECLEF/metadata_with_classes_test.csv"
    test_metadata = pd.read_csv(test_metadata_path)
    test_dataset = TrainDataset(test_data_path, test_metadata, subset="train", transform=transforms.ToTensor())

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    
    # Create model, optimizer, and scheduler
    model = ModifiedResNet18(num_species).to(device)
    model = DDP(model, device_ids=[rank])
    
    class_counts = np.array([1, 1, 1, 1, 1])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalisation
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class Weights:", class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = FocalLoss(gamma=2, alpha=class_weights_tensor)
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
        for data, targets, _ in tqdm(train_loader, desc=f"Training (Rank {rank})", leave=False):
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            out = model(data)
            
            # Convertir les targets en indices de classe
            target_indices = torch.argmax(targets, dim=2)
            
            # Calculer la perte pour chaque espèce individuellement
            loss = 0.0
            for i in range(num_species):
                loss += criterion(out[:, i, :], target_indices[:, i])

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
        with torch.no_grad():
            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Validation...")
            for data, targets, _ in tqdm(test_loader, desc=f"Validation (Rank {rank})", leave=False):
                data = data.to(device)
                targets = targets.to(device)

                out = model(data)
                
                target_indices = torch.argmax(targets, dim=2)

                # Calculer la perte pour chaque espèce individuellement
                loss = 0.0
                for i in range(num_species):
                    loss += criterion(out[:, i, :], target_indices[:, i])
                
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
        if rank == 0 and epoch % save_every == 0:
            torch.save(model.state_dict(), f"resnet18_with_bioclimatic_cubes_epoch_{epoch}.pth")
    
    cleanup()

if __name__ == "__main__":
    total_epochs = 30
    save_every = 4
    batch_size = 512
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, batch_size, save_every), nprocs=world_size)
