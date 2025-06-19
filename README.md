# NeurIPS - Open Polymer Prediction 2025

Predicting polymer properties from SMILES using machine learning.

## Overview
This project is for the NeurIPS 2025 Open Polymer Prediction competition. The goal is to predict five key polymer properties (Tg, FFV, Tc, Density, Rg) from SMILES strings using ML models.

## Project Structure
- `notebooks/` - Jupyter notebooks for exploration and modeling
- `src/` - Utility scripts (feature engineering, metrics, etc.)
- `data/` - Place train/test CSVs here

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the competition data and place it in the `data/` folder.

## Workflow
- Data loading and EDA
- Feature engineering (SMILES to descriptors)
- Model training and evaluation
- Submission file creation

## Submission
- Output a `submission.csv` in the required format for the competition. 