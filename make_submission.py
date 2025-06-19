import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
import lightgbm as lgb

from utils import smiles_to_features
from gnn_model import PolymerGNNModel, smiles_to_graph
from transformer_model import PolymerTransformerModel

# Configuration
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROPERTY_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def load_models(property_name):
    """Load all models for a given property."""
    models = {
        'lgb': [],
        'gnn': [],
        'transformer': []
    }
    
    # Load LightGBM models
    for i in range(N_FOLDS):
        model_path = f'models/{property_name}_lgb_fold{i}.txt'
        model = lgb.Booster(model_file=model_path)
        models['lgb'].append(model)
    
    # Load GNN models
    for i in range(N_FOLDS):
        model = PolymerGNNModel(num_tasks=1, device=DEVICE)
        model.load_model(f'models/{property_name}_gnn_fold{i}.pt')
        models['gnn'].append(model)
    
    # Load Transformer models
    for i in range(N_FOLDS):
        model = PolymerTransformerModel(num_tasks=1, device=DEVICE)
        model.load_model(f'models/{property_name}_transformer_fold{i}.pt')
        models['transformer'].append(model)
    
    # Load scaler
    scaler_data = np.load(f'models/{property_name}_scaler.npy', allow_pickle=True).item()
    scaler = {'mean': scaler_data['mean'], 'scale': scaler_data['scale']}
    
    return models, scaler

def predict_property(smiles_list, models, scaler):
    """Make predictions using all models for a property."""
    predictions = []
    
    # 1. LightGBM predictions
    features = []
    for smile in smiles_list:
        try:
            feat = smiles_to_features(smile)
            if feat is not None:
                features.append(feat)
            else:
                features.append(np.zeros_like(features[0]))  # Use zero features for failed cases
        except:
            if features:
                features.append(np.zeros_like(features[0]))
            else:
                continue
    
    X = np.vstack(features)
    lgb_preds = []
    for model in models['lgb']:
        pred = model.predict(X)
        lgb_preds.append(pred)
    lgb_preds = np.mean(lgb_preds, axis=0)
    predictions.append(lgb_preds)
    
    # 2. GNN predictions
    graphs = []
    for smile in smiles_list:
        try:
            graph = smiles_to_graph(smile)
            graphs.append(graph)
        except:
            # For failed cases, use a simple graph
            graph = smiles_to_graph('C')
            graphs.append(graph)
    
    loader = DataLoader(graphs, batch_size=32)
    gnn_preds = []
    for model in models['gnn']:
        preds = []
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                pred = model.model(batch.x, batch.edge_index, batch.batch)
                preds.append(pred.cpu().numpy())
        gnn_preds.append(np.vstack(preds))
    gnn_preds = np.mean(gnn_preds, axis=0).ravel()
    predictions.append(gnn_preds)
    
    # 3. Transformer predictions
    transformer_preds = []
    for model in models['transformer']:
        pred = model.predict(smiles_list)
        transformer_preds.append(pred)
    transformer_preds = np.mean(transformer_preds, axis=0).ravel()
    predictions.append(transformer_preds)
    
    # Ensemble predictions (weighted average)
    weights = [0.4, 0.3, 0.3]  # LightGBM, GNN, Transformer weights
    final_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        final_pred += weight * pred
    
    # Inverse transform predictions
    final_pred = final_pred * scaler['scale'] + scaler['mean']
    return final_pred

def main():
    # Load test data
    print("Loading test data...")
    test = pd.read_csv('data/test.csv')
    print(f"Loaded {len(test)} test samples")
    
    # Make predictions for each property
    predictions = {}
    for prop in PROPERTY_COLS:
        print(f"\nMaking predictions for {prop}...")
        models, scaler = load_models(prop)
        predictions[prop] = predict_property(test['SMILES'].values, models, scaler)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test['id']
    })
    for prop in PROPERTY_COLS:
        submission[prop] = predictions[prop]
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to 'submission.csv'")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(submission.head())

if __name__ == "__main__":
    main()
