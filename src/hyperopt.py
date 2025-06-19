import optuna
import torch
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Union
import lightgbm as lgb
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
import logging

class HyperOptimizer:
    def __init__(self, n_trials: int = 100, n_jobs: int = -1,
                 study_name: str = "polymer_prediction",
                 storage: Optional[str] = None):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs (-1 for all available)
            study_name: Name of the optimization study
            storage: Optional storage URL for the study
        """
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage
        
    def optimize_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         metric: str = 'rmse') -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            metric: Optimization metric
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            param = {
                'objective': 'regression',
                'metric': metric,
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            
            model = lgb.LGBMRegressor(**param)
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=100,
                     verbose=False)
            
            val_score = model.best_score_['valid_0'][metric]
            return val_score
        
        study = optuna.create_study(
            study_name=f"{self.study_name}_lightgbm",
            storage=self.storage,
            direction="minimize",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        return study.best_params
    
    def optimize_mpnn(self, train_loader: DataLoader, val_loader: DataLoader,
                     node_dim: int, edge_dim: int,
                     device: str = 'cuda') -> Dict[str, Any]:
        """
        Optimize MPNN hyperparameters.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            node_dim: Input node feature dimension
            edge_dim: Input edge feature dimension
            device: Device to run on
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 2, 8),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
            }
            
            # Create model with trial parameters
            model = UncertaintyMPNN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                num_tasks=1,  # Assuming single task optimization
                dropout_rate=params['dropout_rate']
            ).to(device)
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(100):  # Maximum 100 epochs
                # Train
                model.train()
                train_loss = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    mean, logvar = model(batch)
                    loss = gaussian_nll_loss(mean, logvar, batch.y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validate
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        mean, logvar = model(batch)
                        loss = gaussian_nll_loss(mean, logvar, batch.y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            return best_val_loss
        
        study = optuna.create_study(
            study_name=f"{self.study_name}_mpnn",
            storage=self.storage,
            direction="minimize",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)  # n_jobs=1 for GPU
        return study.best_params
    
    def optimize_ensemble_weights(self, predictions: Dict[str, np.ndarray],
                                targets: np.ndarray,
                                metric: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, float]:
        """
        Optimize ensemble model weights.
        
        Args:
            predictions: Dictionary of model predictions
            targets: True target values
            metric: Metric function to optimize
            
        Returns:
            Dictionary of optimal weights for each model
        """
        def objective(trial):
            weights = np.array([
                trial.suggest_float(f'weight_{i}', 0, 1)
                for i in range(len(predictions))
            ])
            weights = weights / weights.sum()  # Normalize weights
            
            # Compute weighted ensemble prediction
            ensemble_pred = np.zeros_like(targets)
            for i, (model_name, pred) in enumerate(predictions.items()):
                ensemble_pred += weights[i] * pred
            
            return metric(targets, ensemble_pred)
        
        study = optuna.create_study(
            study_name=f"{self.study_name}_ensemble",
            storage=self.storage,
            direction="minimize",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # Get optimal weights
        weights = {}
        weight_sum = 0
        for i, model_name in enumerate(predictions.keys()):
            weights[model_name] = study.best_params[f'weight_{i}']
            weight_sum += weights[model_name]
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= weight_sum
        
        return weights
    
    @staticmethod
    def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
        """Compute negative log likelihood loss."""
        precision = torch.exp(-logvar)
        loss = 0.5 * (logvar + (mean - target)**2 * precision)
        return torch.mean(loss) 