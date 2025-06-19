import torch
import numpy as np
import shap
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import seaborn as sns
from torch_geometric.data import Data, Batch

class ModelInterpreter:
    def __init__(self, models: Dict[str, Any], property_names: List[str]):
        """
        Initialize model interpreter.
        
        Args:
            models: Dictionary of trained models
            property_names: List of property names to interpret
        """
        self.models = models
        self.property_names = property_names
        
    def explain_lightgbm(self, model, X: np.ndarray, feature_names: List[str]) -> shap.Explanation:
        """Calculate SHAP values for LightGBM model."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X,
            feature_names=feature_names
        )
    
    def explain_gnn_attention(self, model, data: Data) -> Dict[str, np.ndarray]:
        """Extract attention weights from GNN model."""
        model.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([data]).to(next(model.parameters()).device)
            attention_weights = model.get_attention_weights(batch)
        return {
            'node_attention': attention_weights['node'].cpu().numpy(),
            'edge_attention': attention_weights['edge'].cpu().numpy() if 'edge' in attention_weights else None
        }
    
    def visualize_molecule_attention(self, smiles: str, attention_weights: np.ndarray,
                                   property_name: str, save_path: str = None):
        """Visualize molecular structure with attention weights."""
        mol = Chem.MolFromSmiles(smiles)
        
        # Normalize attention weights
        weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
        
        # Generate atom colors based on attention weights
        atom_colors = {}
        for i, weight in enumerate(weights):
            # Convert weight to RGB color (red intensity)
            atom_colors[i] = (1, 1-weight, 1-weight)
        
        # Draw molecule with highlighted atoms
        img = Draw.MolToImage(mol, size=(400, 400), highlightAtoms=range(len(weights)),
                            highlightColor=None, atomHighlights=atom_colors)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f'Attention Weights for {property_name}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, shap_values: shap.Explanation, property_name: str,
                              save_path: str = None):
        """Plot SHAP feature importance summary."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, plot_type='bar', show=False)
        plt.title(f'Feature Importance for {property_name}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def analyze_uncertainty(self, predictions: np.ndarray, uncertainties: np.ndarray,
                          property_name: str, save_path: str = None):
        """Analyze and visualize prediction uncertainties."""
        plt.figure(figsize=(10, 6))
        
        # Plot prediction vs uncertainty
        plt.scatter(predictions, uncertainties, alpha=0.5)
        plt.xlabel('Predicted Value')
        plt.ylabel('Uncertainty (Std)')
        plt.title(f'Prediction Uncertainty Analysis for {property_name}')
        
        # Add trend line
        z = np.polyfit(predictions, uncertainties, 1)
        p = np.poly1d(z)
        plt.plot(predictions, p(predictions), "r--", alpha=0.8)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def explain_prediction(self, smiles: str, features: np.ndarray,
                          feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single prediction.
        
        Returns:
            Dictionary containing explanations for each model type and property
        """
        explanations = {}
        
        for prop in self.property_names:
            prop_explanations = {}
            
            # LightGBM SHAP values
            if 'lgb' in self.models[prop]:
                shap_values = self.explain_lightgbm(
                    self.models[prop]['lgb'],
                    features.reshape(1, -1),
                    feature_names
                )
                prop_explanations['shap'] = shap_values
            
            # GNN attention weights
            if 'gnn' in self.models[prop]:
                attention = self.explain_gnn_attention(
                    self.models[prop]['gnn'],
                    self._smiles_to_graph(smiles)
                )
                prop_explanations['attention'] = attention
            
            explanations[prop] = prop_explanations
        
        return explanations
    
    def generate_interpretation_report(self, smiles: str, features: np.ndarray,
                                    feature_names: List[str], save_dir: str):
        """Generate a comprehensive interpretation report with visualizations."""
        explanations = self.explain_prediction(smiles, features, feature_names)
        
        for prop in self.property_names:
            # Create property-specific directory
            prop_dir = f"{save_dir}/{prop}"
            os.makedirs(prop_dir, exist_ok=True)
            
            # Plot SHAP values
            if 'shap' in explanations[prop]:
                self.plot_feature_importance(
                    explanations[prop]['shap'],
                    prop,
                    f"{prop_dir}/feature_importance.png"
                )
            
            # Plot attention visualization
            if 'attention' in explanations[prop]:
                self.visualize_molecule_attention(
                    smiles,
                    explanations[prop]['attention']['node'],
                    prop,
                    f"{prop_dir}/attention_visualization.png"
                )
            
            # Plot uncertainty analysis if available
            if hasattr(self.models[prop]['gnn'], 'predict_with_uncertainty'):
                predictions, uncertainties = self.models[prop]['gnn'].predict_with_uncertainty(
                    self._smiles_to_graph(smiles)
                )
                self.analyze_uncertainty(
                    predictions,
                    uncertainties,
                    prop,
                    f"{prop_dir}/uncertainty_analysis.png"
                )
    
    @staticmethod
    def _smiles_to_graph(smiles: str) -> Data:
        """Convert SMILES to PyTorch Geometric Data object."""
        mol = Chem.MolFromSmiles(smiles)
        
        # Get node features
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetIsAromatic()),
                atom.GetHybridization(),
                atom.GetTotalNumHs()
            ]
            node_features.append(features)
        
        # Get edge indices and features
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_indices += [[i, j], [j, i]]
            
            features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing())
            ]
            edge_features += [features, features]
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr) 