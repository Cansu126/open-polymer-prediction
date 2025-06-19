import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error
import torch
from torch_geometric.data import Data
import logging
from .hyperopt import HyperOptimizer

class WeightedEnsemble:
    def __init__(self, models: Dict[str, Dict[str, Any]], property_names: List[str]):
        """
        Initialize weighted ensemble.
        
        Args:
            models: Dictionary of models for each property
            property_names: List of property names to predict
        """
        self.models = models
        self.property_names = property_names
        self.weights = {prop: None for prop in property_names}
        self.hyperopt = HyperOptimizer(n_trials=50)
        
    def optimize_weights(self, val_predictions: Dict[str, Dict[str, np.ndarray]],
                        val_targets: Dict[str, np.ndarray]):
        """
        Optimize ensemble weights using validation data.
        
        Args:
            val_predictions: Dictionary of validation predictions for each model and property
            val_targets: Dictionary of validation targets for each property
        """
        for prop in self.property_names:
            # Optimize weights for each property
            self.weights[prop] = self.hyperopt.optimize_ensemble_weights(
                predictions=val_predictions[prop],
                targets=val_targets[prop],
                metric=mean_squared_error
            )
            
            logging.info(f"Optimized weights for {prop}:")
            for model_name, weight in self.weights[prop].items():
                logging.info(f"  {model_name}: {weight:.3f}")
    
    def predict(self, data: Any, return_uncertainty: bool = True) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Make ensemble predictions with uncertainty estimation.
        
        Args:
            data: Input data (can be different formats for different models)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary of predictions and uncertainties for each property
        """
        results = {}
        
        for prop in self.property_names:
            if self.weights[prop] is None:
                raise ValueError(f"Weights not optimized for property {prop}")
            
            predictions = []
            uncertainties = []
            
            # Get predictions from each model
            for model_name, model in self.models[prop].items():
                weight = self.weights[prop][model_name]
                
                if hasattr(model, 'predict_with_uncertainty'):
                    # Model supports uncertainty estimation
                    pred, uncert = model.predict_with_uncertainty(data)
                    predictions.append(weight * pred)
                    uncertainties.append((weight * uncert)**2)  # Square for variance
                else:
                    # Model without uncertainty estimation
                    pred = model.predict(data)
                    predictions.append(weight * pred)
            
            # Combine predictions
            ensemble_pred = np.sum(predictions, axis=0)
            
            if return_uncertainty and uncertainties:
                # Combine uncertainties (sum of variances for weighted predictions)
                ensemble_uncert = np.sqrt(np.sum(uncertainties, axis=0))
                results[prop] = (ensemble_pred, ensemble_uncert)
            else:
                results[prop] = (ensemble_pred, None)
        
        return results
    
    def evaluate(self, test_data: Any, test_targets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ensemble performance.
        
        Args:
            test_data: Test input data
            test_targets: Dictionary of test targets for each property
            
        Returns:
            Dictionary of metrics for each property
        """
        predictions = self.predict(test_data, return_uncertainty=True)
        metrics = {}
        
        for prop in self.property_names:
            pred, uncert = predictions[prop]
            target = test_targets[prop]
            
            # Calculate metrics
            mse = mean_squared_error(target, pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(target - pred))
            
            # Calculate metrics considering uncertainty
            if uncert is not None:
                # Negative log likelihood assuming Gaussian distribution
                nll = np.mean(0.5 * (np.log(2 * np.pi * uncert**2) + 
                                   (target - pred)**2 / uncert**2))
                
                # Calculate calibration error
                z_scores = (target - pred) / uncert
                expected_proportions = [0.682, 0.954, 0.997]  # 1, 2, 3 sigma
                actual_proportions = [
                    np.mean(np.abs(z_scores) <= 1),
                    np.mean(np.abs(z_scores) <= 2),
                    np.mean(np.abs(z_scores) <= 3)
                ]
                calibration_error = np.mean(np.abs(np.array(actual_proportions) - 
                                                 np.array(expected_proportions)))
                
                metrics[prop] = {
                    'rmse': rmse,
                    'mae': mae,
                    'nll': nll,
                    'calibration_error': calibration_error
                }
            else:
                metrics[prop] = {
                    'rmse': rmse,
                    'mae': mae
                }
        
        return metrics
    
    def save_weights(self, path: str):
        """Save ensemble weights."""
        np.save(path, self.weights)
    
    def load_weights(self, path: str):
        """Load ensemble weights."""
        self.weights = np.load(path, allow_pickle=True).item() 