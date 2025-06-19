import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import DataLoader
import lightgbm as lgb

from utils import smiles_to_features
from gnn_model import PolymerGNNModel, smiles_to_graph
from transformer_model import PolymerTransformerModel

# Configuration
RANDOM_SEED = 42
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROPERTY_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train a LightGBM model with optimized parameters."""
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    return model

def prepare_gnn_data(smiles_list, targets=None):
    """Convert SMILES to graph data for GNN."""
    graphs = []
    for i, smile in enumerate(smiles_list):
        try:
            graph = smiles_to_graph(smile)
            if targets is not None:
                graph.y = torch.tensor(targets[i], dtype=torch.float)
            graphs.append(graph)
        except Exception as e:
            print(f"Error processing SMILES {smile}: {str(e)}")
    return graphs

def train_models():
    # 1. Load and preprocess data
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    
    # Initialize models for each property
    models = {prop: {'lgb': [], 'gnn': [], 'transformer': []} for prop in PROPERTY_COLS}
    scalers = {prop: StandardScaler() for prop in PROPERTY_COLS}
    
    # 2. Feature extraction for LightGBM
    print("\nExtracting features for LightGBM...")
    features = []
    valid_indices = []
    for i, smile in enumerate(train_df['SMILES']):
        if i % 100 == 0:
            print(f"Processing row {i}...")
        try:
            feat = smiles_to_features(smile)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)
        except Exception as e:
            print(f"Error processing SMILES {smile}: {str(e)}")
            continue
    
    X = np.vstack(features)
    valid_df = train_df.iloc[valid_indices]
    
    # 3. Cross-validation training
    print("\nStarting cross-validation training...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/{N_FOLDS}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        train_smiles = valid_df['SMILES'].iloc[train_idx].values
        val_smiles = valid_df['SMILES'].iloc[val_idx].values
        
        for prop in PROPERTY_COLS:
            print(f"\nTraining models for {prop}")
            
            # Get target values
            y = valid_df[prop].values
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale targets
            y_train_scaled = scalers[prop].fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val_scaled = scalers[prop].transform(y_val.reshape(-1, 1)).ravel()
            
            # Train LightGBM
            print("Training LightGBM...")
            lgb_model = train_lightgbm(X_train, y_train_scaled, X_val, y_val_scaled)
            models[prop]['lgb'].append(lgb_model)
            
            # Train GNN
            print("Training GNN...")
            gnn_model = PolymerGNNModel(num_tasks=1, device=DEVICE)
            train_graphs = prepare_gnn_data(train_smiles, y_train_scaled.reshape(-1, 1))
            val_graphs = prepare_gnn_data(val_smiles, y_val_scaled.reshape(-1, 1))
            
            train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=32)
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(100):
                train_loss = gnn_model.train_step(train_loader)
                val_loss = gnn_model.train_step(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            models[prop]['gnn'].append(gnn_model)
            
            # Train Transformer
            print("Training Transformer...")
            transformer_model = PolymerTransformerModel(num_tasks=1, device=DEVICE)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(10):  # Fewer epochs for transformer due to computational cost
                for i in range(0, len(train_smiles), 32):
                    batch_smiles = train_smiles[i:i + 32]
                    batch_y = torch.tensor(y_train_scaled[i:i + 32], dtype=torch.float)
                    loss = transformer_model.train_step(batch_smiles, batch_y.reshape(-1, 1))
                
                # Validate
                val_pred = transformer_model.predict(val_smiles.tolist())
                val_loss = np.mean((val_pred.ravel() - y_val_scaled) ** 2)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            models[prop]['transformer'].append(transformer_model)
    
    # 4. Save models and scalers
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    
    for prop in PROPERTY_COLS:
        # Save LightGBM models
        for i, model in enumerate(models[prop]['lgb']):
            model.booster_.save_model(f'models/{prop}_lgb_fold{i}.txt')
        
        # Save GNN models
        for i, model in enumerate(models[prop]['gnn']):
            model.save_model(f'models/{prop}_gnn_fold{i}.pt')
        
        # Save Transformer models
        for i, model in enumerate(models[prop]['transformer']):
            model.save_model(f'models/{prop}_transformer_fold{i}.pt')
        
        # Save scaler
        np.save(f'models/{prop}_scaler.npy', {
            'mean': scalers[prop].mean_,
            'scale': scalers[prop].scale_
        })
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_models()



