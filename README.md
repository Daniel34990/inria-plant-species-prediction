# 🌍 INRIA Plant Species Distribution Modeling with Remote Sensing Data

This repository contains the code and experiments developed during a **1-month research internship at INRIA** — the French National Institute for Research in Digital Science and Technology — in the context of a large-scale biodiversity prediction challenge.


## 📌 Project Summary

The goal of this internship project was to design and train **Species Distribution Models (SDMs)** capable of predicting plant species composition at fine spatial and temporal resolutions across Europe.

The task involves integrating multi-modal geospatial data — including satellite imagery, climate time series, and environmental rasters — to predict the **presence or absence** of plant species at a given location. Models were built using **PyTorch** and trained on a large-scale dataset of over **5 million species observations**, part of which were collected by citizen scientists using the [Pl@ntNet](https://plantnet.org) application, developed at INRIA.




## 🔍 Project Context

This research was part of a biodiversity modeling challenge focused on **species presence-only (PO)** and **presence-absence (PA)** data:

- **PO data**: ~5M records collected from diverse sources (e.g., GBIF, Pl@ntNet)
- **PA data**: ~90K plots used for validation and evaluation
- **Multi-modal inputs**:
  - Sentinel-2 RGB/NIR image patches (128×128)
  - 20-year multi-spectral satellite time series (Landsat)
  - Monthly climate data (CHELSA)
  - Environmental rasters (elevation, land cover, human footprint, soil grids)

![Project Overview](/README-src/image.png)


## 🧪 My Contributions

- ✨ Developed **feature engineering techniques** to better exploit presence-only data
- 🧭 Implemented a method to **group geographically close observations** to approximate absence and increase model discriminability
- 📈 This technique led to a **slight increase in F1 score** 
- 🔁 Cleaned and preprocessed metadata and environmental inputs
- 🧱 Built & trained **ResNet18-based architectures** for image-based species prediction
- ⚡ Models were trained on **2× NVIDIA A100 GPUs** using PyTorch

---

## 🧱 Repository Structure

| File / Notebook                 | Description |
|--------------------------------|-------------|
| `data_gen.ipynb`               | Preprocessing and dataset preparation |
| `PO_augment.ipynb`             | PO data augmentation via spatial grouping |
| `metadata_augment.py`          | Metadata transformations and enrichment |
| `final_model_PO.ipynb/.py`     | Model trained using presence-only data |
| `final_model_PA.ipynb`         | Model trained using presence-absence data |
| `final_model_presence.ipynb/.py`| Combined presence signal modeling |
| `final_model_classes.ipynb/.py`| Multi-label classification model logic |

---

## 🧠 Model Architecture

```text
Input (3×128×128 RGB or 4×128×128 RGB+NIR)
↓
ResNet-18 Backbone
↓
Global Average Pooling
↓
Fully Connected Layer
↓
Sigmoid (multi-label output for 10K+ species)
