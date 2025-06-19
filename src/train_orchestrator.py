import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import wandb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import optuna
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime

from .gnn_model import UncertaintyMPNN
from .transformer_model import PolymerTransformer
from .polymer_features import PolymerFeatureExtractor
from .advanced_ensemble import BayesianModelAveraging

class TrainingOrchestrator:
    """Orchestrates the training of multiple models with advanced techniques."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Initialize components
        self.feature_extractor = PolymerFeatureExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize W&B
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config['project_name'],
                config=self.config
            )
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, data_path: str) -> Tuple[Dict, Dict]:
        """Prepare data for training with advanced preprocessing."""
        # Load data
        df = pd.read_csv(data_path)
        
        # Extract features
        self.logger.info("Extracting molecular features...")
        features = {}
        for smiles in df['SMILES'].values:
            features[smiles] = self.feature_extractor.extract_polymer_features(smiles)
        
        # Prepare targets
        targets = {}
        for prop in self.config['property_names']:
            targets[prop] = df[prop].values
            
        return features, targets
    
    def create_models(self) -> Dict[str, Dict[str, Any]]:
        """Create models for each property."""
        models = {prop: {} for prop in self.config['property_names']}
        
        for prop in self.config['property_names']:
            # GNN model
            models[prop]['gnn'] = UncertaintyMPNN(
                node_dim=self.config['gnn']['node_dim'],
                edge_dim=self.config['gnn']['edge_dim'],
                hidden_dim=self.config['gnn']['hidden_dim'],
                num_layers=self.config['gnn']['num_layers'],
                num_tasks=1,
                dropout_rate=self.config['gnn']['dropout_rate']
            ).to(self.device)
            
            # Transformer model
            models[prop]['transformer'] = PolymerTransformer(
                num_tasks=1
            ).to(self.device)
        
        return models
    
    def train(self, features: Dict, targets: Dict[str, np.ndarray]):
        """Train models with cross-validation and advanced techniques."""
        kf = KFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_seed=42
        )
        
        # Store results for each fold
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(list(features.keys()))):
            self.logger.info(f"Training fold {fold + 1}/{self.config['cv_folds']}")
            
            # Split data
            train_features = {k: features[k] for k in list(features.keys())[train_idx]}
            val_features = {k: features[k] for k in list(features.keys())[val_idx]}
            
            train_targets = {
                prop: targets[prop][train_idx] for prop in self.config['property_names']
            }
            val_targets = {
                prop: targets[prop][val_idx] for prop in self.config['property_names']
            }
            
            # Create models
            models = self.create_models()
            
            # Train individual models
            for prop in self.config['property_names']:
                self.logger.info(f"Training models for {prop}")
                
                # Train GNN
                self._train_gnn(
                    models[prop]['gnn'],
                    train_features,
                    train_targets[prop],
                    val_features,
                    val_targets[prop]
                )
                
                # Train Transformer
                self._train_transformer(
                    models[prop]['transformer'],
                    train_features,
                    train_targets[prop],
                    val_features,
                    val_targets[prop]
                )
            
            # Create and optimize ensemble
            ensemble = BayesianModelAveraging(models, self.config['property_names'])
            ensemble.optimize_ensemble(val_features, val_targets)
            
            # Evaluate fold
            metrics = ensemble.evaluate(val_features, val_targets)
            cv_results.append(metrics)
            
            # Log results
            if self.config.get('use_wandb', False):
                wandb.log({
                    f'fold_{fold}/metrics': metrics,
                    'fold': fold
                })
            
            # Save models
            self._save_models(models, ensemble, fold)
        
        # Aggregate and log final results
        final_metrics = self._aggregate_cv_results(cv_results)
        self.logger.info("Final cross-validation results:")
        self.logger.info(json.dumps(final_metrics, indent=2))
        
        if self.config.get('use_wandb', False):
            wandb.log({'final_metrics': final_metrics})
    
    def _train_gnn(self, model, train_features, train_targets, val_features, val_targets):
        """Train GNN model with advanced techniques."""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['gnn']['learning_rate'],
            weight_decay=self.config['gnn']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config['gnn']['t0'],
            T_mult=2
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['gnn']['epochs']):
            # Training
            model.train()
            train_loss = 0
            
            for batch in self._create_gnn_batches(train_features, train_targets):
                optimizer.zero_grad()
                mean, logvar = model(batch)
                loss = self._gaussian_nll_loss(mean, logvar, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in self._create_gnn_batches(val_features, val_targets):
                    mean, logvar = model(batch)
                    loss = self._gaussian_nll_loss(mean, logvar, batch.y)
                    val_loss += loss.item()
            
            # Update scheduler
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['gnn']['patience']:
                break
    
    def _train_transformer(self, model, train_features, train_targets, val_features, val_targets):
        """Train Transformer model with advanced techniques."""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['transformer']['learning_rate'],
            weight_decay=self.config['transformer']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['transformer']['max_lr'],
            epochs=self.config['transformer']['epochs'],
            steps_per_epoch=len(train_features) // self.config['batch_size']
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['transformer']['epochs']):
            # Training
            model.train()
            train_loss = 0
            
            for batch_smiles, batch_targets in self._create_transformer_batches(
                train_features, train_targets
            ):
                optimizer.zero_grad()
                mean, logvar = model(batch_smiles)
                loss = self._gaussian_nll_loss(mean, logvar, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_smiles, batch_targets in self._create_transformer_batches(
                    val_features, val_targets
                ):
                    mean, logvar = model(batch_smiles)
                    loss = self._gaussian_nll_loss(mean, logvar, batch_targets)
                    val_loss += loss.item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['transformer']['patience']:
                break
    
    def _gaussian_nll_loss(self, mean, logvar, targets):
        """Calculate Gaussian negative log likelihood loss."""
        return 0.5 * torch.mean(
            logvar + (targets - mean)**2 / torch.exp(logvar)
        )
    
    def _create_gnn_batches(self, features, targets):
        """Create batches for GNN training."""
        data_list = []
        
        for smiles, feat in features.items():
            # Convert to PyTorch Geometric data object
            data = Data(
                x=torch.tensor(feat['basic'], dtype=torch.float),
                edge_index=torch.tensor(feat['graph_edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(feat['graph_edge_attr'], dtype=torch.float),
                y=torch.tensor([targets[smiles]], dtype=torch.float)
            )
            data_list.append(data)
        
        return DataLoader(
            data_list,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
    
    def _create_transformer_batches(self, features, targets):
        """Create batches for Transformer training."""
        smiles_list = list(features.keys())
        target_list = [targets[s] for s in smiles_list]
        
        for i in range(0, len(smiles_list), self.config['batch_size']):
            batch_smiles = smiles_list[i:i + self.config['batch_size']]
            batch_targets = torch.tensor(
                target_list[i:i + self.config['batch_size']],
                dtype=torch.float
            )
            yield batch_smiles, batch_targets
    
    def _save_models(self, models: Dict, ensemble: BayesianModelAveraging, fold: int):
        """Save models and ensemble parameters."""
        save_dir = Path(self.config['save_dir']) / f'fold_{fold}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for prop in self.config['property_names']:
            for model_name, model in models[prop].items():
                torch.save(
                    model.state_dict(),
                    save_dir / f'{prop}_{model_name}_model.pt'
                )
        
        # Save ensemble parameters
        ensemble_params = {
            'weights': ensemble.weights,
            'temperature': ensemble.temperature,
            'calibration': ensemble.calibration
        }
        
        with open(save_dir / 'ensemble_params.json', 'w') as f:
            json.dump(ensemble_params, f, indent=2)
    
    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results."""
        aggregated = {}
        
        for prop in self.config['property_names']:
            aggregated[prop] = {}
            
            for metric in cv_results[0][prop].keys():
                values = [fold[prop][metric] for fold in cv_results]
                aggregated[prop][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        return aggregated 