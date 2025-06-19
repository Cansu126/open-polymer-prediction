import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from torch_geometric.data import Data, Batch
import logging
from scipy.stats import norm
from .gnn_model import UncertaintyMPNN
from .transformer_model import PolymerTransformer

class BayesianModelAveraging:
    """Bayesian Model Averaging with temperature scaling and uncertainty calibration."""
    
    def __init__(self, models: Dict[str, Dict[str, Any]], property_names: List[str]):
        self.models = models
        self.property_names = property_names
        self.weights = {prop: {} for prop in property_names}
        self.temperature = {prop: {} for prop in property_names}
        self.calibration = {prop: {} for prop in property_names}
    
    def optimize_ensemble(self, val_data: Dict, val_targets: Dict[str, np.ndarray]):
        """Optimize ensemble weights, temperature, and calibration parameters."""
        for prop in self.property_names:
            study = optuna.create_study(direction="minimize")
            
            def objective(trial):
                # Sample weights
                weights = {
                    model_name: trial.suggest_float(f"weight_{model_name}", 0, 1)
                    for model_name in self.models[prop].keys()
                }
                # Normalize weights
                total = sum(weights.values())
                weights = {k: v/total for k, v in weights.items()}
                
                # Temperature scaling parameter
                temp = {
                    model_name: trial.suggest_float(f"temp_{model_name}", 0.1, 5.0)
                    for model_name in self.models[prop].keys()
                }
                
                # Calibration parameters
                calib = {
                    model_name: {
                        'scale': trial.suggest_float(f"scale_{model_name}", 0.1, 10.0),
                        'shift': trial.suggest_float(f"shift_{model_name}", -2.0, 2.0)
                    }
                    for model_name in self.models[prop].keys()
                }
                
                # Make predictions
                predictions, uncertainties = self._predict_with_params(
                    val_data, weights, temp, calib
                )
                
                # Calculate metrics
                mse = mean_squared_error(val_targets[prop], predictions[prop])
                nll = self._negative_log_likelihood(
                    val_targets[prop], 
                    predictions[prop], 
                    uncertainties[prop]
                )
                calibration_error = self._calculate_calibration_error(
                    val_targets[prop],
                    predictions[prop],
                    uncertainties[prop]
                )
                
                return mse + 0.1 * nll + 0.1 * calibration_error
            
            # Optimize
            study.optimize(objective, n_trials=100)
            
            # Store best parameters
            best_params = study.best_params
            self.weights[prop] = {
                model_name: best_params[f"weight_{model_name}"] / sum(
                    best_params[f"weight_{m}"] for m in self.models[prop].keys()
                )
                for model_name in self.models[prop].keys()
            }
            self.temperature[prop] = {
                model_name: best_params[f"temp_{model_name}"]
                for model_name in self.models[prop].keys()
            }
            self.calibration[prop] = {
                model_name: {
                    'scale': best_params[f"scale_{model_name}"],
                    'shift': best_params[f"shift_{model_name}"]
                }
                for model_name in self.models[prop].keys()
            }
    
    def _negative_log_likelihood(self, targets, predictions, uncertainties):
        """Calculate negative log likelihood assuming Gaussian distribution."""
        return np.mean(
            0.5 * np.log(2 * np.pi * uncertainties**2) + 
            (targets - predictions)**2 / (2 * uncertainties**2)
        )
    
    def _calculate_calibration_error(self, targets, predictions, uncertainties):
        """Calculate calibration error using reliability diagram."""
        z_scores = (targets - predictions) / uncertainties
        expected_proportions = [0.682, 0.954, 0.997]  # 1, 2, 3 sigma
        actual_proportions = [
            np.mean(np.abs(z_scores) <= 1),
            np.mean(np.abs(z_scores) <= 2),
            np.mean(np.abs(z_scores) <= 3)
        ]
        return np.mean(np.abs(np.array(actual_proportions) - np.array(expected_proportions)))
    
    def _predict_with_params(self, data: Any, weights: Dict, temp: Dict, calib: Dict) -> Tuple[Dict, Dict]:
        """Make predictions using specified parameters."""
        predictions = {}
        uncertainties = {}
        
        for prop in self.property_names:
            prop_predictions = []
            prop_uncertainties = []
            
            for model_name, model in self.models[prop].items():
                if isinstance(model, (UncertaintyMPNN, PolymerTransformer)):
                    pred, uncert = model.predict_with_uncertainty(data)
                else:
                    pred = model.predict(data)
                    uncert = np.zeros_like(pred)
                
                # Apply temperature scaling and calibration
                scaled_pred = pred / temp[model_name]
                scaled_uncert = (uncert / temp[model_name]) * calib[model_name]['scale'] + calib[model_name]['shift']
                
                prop_predictions.append(weights[model_name] * scaled_pred)
                prop_uncertainties.append((weights[model_name] * scaled_uncert)**2)
            
            predictions[prop] = np.sum(prop_predictions, axis=0)
            uncertainties[prop] = np.sqrt(np.sum(prop_uncertainties, axis=0))
        
        return predictions, uncertainties
    
    def predict(self, data: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Make predictions with the optimized ensemble."""
        return self._predict_with_params(data, self.weights, self.temperature, self.calibration)
    
    def evaluate(self, test_data: Any, test_targets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Evaluate ensemble performance with multiple metrics."""
        predictions, uncertainties = self.predict(test_data)
        metrics = {}
        
        for prop in self.property_names:
            pred = predictions[prop]
            uncert = uncertainties[prop]
            target = test_targets[prop]
            
            metrics[prop] = {
                'rmse': np.sqrt(mean_squared_error(target, pred)),
                'r2': r2_score(target, pred),
                'nll': self._negative_log_likelihood(target, pred, uncert),
                'calibration_error': self._calculate_calibration_error(target, pred, uncert),
                'mean_uncertainty': np.mean(uncert),
                'uncertainty_correlation': np.corrcoef(
                    np.abs(target - pred), uncert
                )[0, 1]
            }
            
            # Calculate prediction intervals and coverage
            for confidence in [0.68, 0.95, 0.997]:  # 1, 2, and 3 sigma
                z_score = norm.ppf((1 + confidence) / 2)
                covered = np.abs(target - pred) <= z_score * uncert
                metrics[prop][f'coverage_{confidence}'] = np.mean(covered)
        
        return metrics
    
    def get_model_contributions(self, data: Any) -> Dict[str, Dict[str, float]]:
        """Analyze individual model contributions to the ensemble."""
        contributions = {}
        predictions, _ = self.predict(data)
        
        for prop in self.property_names:
            contributions[prop] = {}
            total_pred = predictions[prop]
            
            for model_name, model in self.models[prop].items():
                # Calculate prediction without this model
                temp_weights = {k: v for k, v in self.weights[prop].items()}
                temp_weights[model_name] = 0
                # Renormalize weights
                total = sum(temp_weights.values())
                if total > 0:
                    temp_weights = {k: v/total for k, v in temp_weights.items()}
                
                temp_pred, _ = self._predict_with_params(
                    data,
                    {prop: temp_weights},
                    self.temperature,
                    self.calibration
                )
                
                # Calculate contribution
                contribution = np.mean(np.abs(total_pred - temp_pred[prop]))
                contributions[prop][model_name] = contribution
        
        return contributions 