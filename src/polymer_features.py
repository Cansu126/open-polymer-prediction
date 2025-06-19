import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdDepictor
from typing import List, Dict, Optional, Tuple
from mordred import Calculator, descriptors
import deepchem as dc
from scipy.stats import skew, kurtosis

class PolymerFeatureExtractor:
    """Advanced feature extraction for polymer molecules."""
    
    def __init__(self):
        # Initialize Mordred calculator with all descriptors
        self.mordred_calc = Calculator(descriptors, ignore_3D=False)
        
        # Initialize DeepChem featurizers
        self.graph_featurizer = dc.feat.ConvMolFeaturizer()
        self.mordred_featurizer = dc.feat.MordredDescriptors(ignore_3D=False)
        
        # Cache for computed features
        self.feature_cache = {}
    
    def extract_polymer_features(self, smiles: str) -> Dict[str, np.ndarray]:
        """Extract comprehensive polymer-specific features."""
        if smiles in self.feature_cache:
            return self.feature_cache[smiles]
        
        try:
            # Replace polymer end groups with explicit atoms
            processed_smiles = self._process_polymer_smiles(smiles)
            mol = Chem.MolFromSmiles(processed_smiles)
            if mol is None:
                raise ValueError(f"Could not parse SMILES: {smiles}")
            
            # Generate 3D conformer
            mol = self._generate_optimized_3d_conformer(mol)
            
            features = {}
            
            # Basic polymer properties
            features['basic'] = self._get_basic_polymer_properties(mol)
            
            # Topological features
            features['topological'] = self._get_topological_features(mol)
            
            # Electronic properties
            features['electronic'] = self._get_electronic_properties(mol)
            
            # Structural features
            features['structural'] = self._get_structural_features(mol)
            
            # Advanced descriptors using Mordred
            features['mordred'] = self._get_mordred_descriptors(mol)
            
            # DeepChem features
            features['deepchem'] = self._get_deepchem_features(mol)
            
            # Polymer-specific patterns
            features['patterns'] = self._get_polymer_patterns(mol)
            
            # Cache the results
            self.feature_cache[smiles] = features
            return features
            
        except Exception as e:
            print(f"Error extracting features for {smiles}: {str(e)}")
            return self._get_empty_features()
    
    def _process_polymer_smiles(self, smiles: str) -> str:
        """Process polymer SMILES by handling end groups and repeat units."""
        # Replace generic end groups with explicit atoms
        processed = smiles.replace('*', '[CH3]')
        return processed
    
    def _generate_optimized_3d_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        """Generate and optimize 3D conformer using force field."""
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
        
        # Optimize all conformers
        energies = []
        for conf_id in range(mol.GetNumConformers()):
            try:
                # MMFF94 optimization
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                # Get energy
                mp = AllChem.MMFFGetMoleculeProperties(mol)
                energy = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id).CalcEnergy()
                energies.append((conf_id, energy))
            except:
                continue
        
        if energies:
            # Keep only the lowest energy conformer
            best_conf_id = min(energies, key=lambda x: x[1])[0]
            conf_ids = list(range(mol.GetNumConformers()))
            conf_ids.remove(best_conf_id)
            for conf_id in conf_ids:
                mol.RemoveConformer(conf_id)
        
        return mol
    
    def _get_basic_polymer_properties(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate basic polymer properties."""
        properties = []
        
        # Molecular weight and size
        properties.extend([
            Descriptors.ExactMolWt(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),
            Descriptors.FractionCSP3(mol),
        ])
        
        # Polymer chain analysis
        chain_length = len(mol.GetAtoms())
        branching_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
        terminal_groups = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() == 1)
        
        properties.extend([
            chain_length,
            branching_points,
            terminal_groups,
            branching_points / chain_length if chain_length > 0 else 0
        ])
        
        return np.array(properties)
    
    def _get_topological_features(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate topological features."""
        features = []
        
        # Basic topological descriptors
        features.extend([
            Descriptors.BertzCT(mol),
            Descriptors.Chi0n(mol),
            Descriptors.Chi1n(mol),
            Descriptors.Chi2n(mol),
            Descriptors.Chi3n(mol),
            Descriptors.Chi4n(mol),
        ])
        
        # Ring analysis
        ri = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ri.AtomRings()]
        
        if ring_sizes:
            features.extend([
                len(ring_sizes),  # Number of rings
                np.mean(ring_sizes),
                np.std(ring_sizes),
                np.min(ring_sizes),
                np.max(ring_sizes),
                skew(ring_sizes),
                kurtosis(ring_sizes)
            ])
        else:
            features.extend([0] * 7)
        
        return np.array(features)
    
    def _get_electronic_properties(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate electronic properties."""
        properties = []
        
        # Charge-related descriptors
        properties.extend([
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
            Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.MinAbsPartialCharge(mol)
        ])
        
        # Electronic effect descriptors
        properties.extend([
            Descriptors.NumValenceElectrons(mol),
            rdMolDescriptors.CalcNumRadicalElectrons(mol),
            Descriptors.MaxAbsEStateIndex(mol),
            Descriptors.MinAbsEStateIndex(mol),
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinEStateIndex(mol)
        ])
        
        return np.array(properties)
    
    def _get_structural_features(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate structural features."""
        features = []
        
        # Conformational descriptors
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            
            # Calculate principal moments of inertia
            try:
                features.extend([
                    Descriptors.PMI1(mol),
                    Descriptors.PMI2(mol),
                    Descriptors.PMI3(mol),
                    Descriptors.NPR1(mol),
                    Descriptors.NPR2(mol),
                    Descriptors.RadiusOfGyration(mol),
                    Descriptors.InertialShapeFactor(mol),
                    Descriptors.Eccentricity(mol),
                    Descriptors.Asphericity(mol),
                    Descriptors.SpherocityIndex(mol)
                ])
            except:
                features.extend([0] * 10)
        else:
            features.extend([0] * 10)
        
        return np.array(features)
    
    def _get_mordred_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate Mordred descriptors."""
        try:
            # Calculate all available Mordred descriptors
            calc = self.mordred_calc.pandas([mol])
            # Convert to numeric, replacing non-numeric values with 0
            numeric_vals = calc.apply(pd.to_numeric, errors='coerce').fillna(0)
            return numeric_vals.values.flatten()
        except:
            # Return zeros if calculation fails
            return np.zeros(1613)  # Standard number of Mordred descriptors
    
    def _get_deepchem_features(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate DeepChem features."""
        try:
            # Graph features
            graph_features = self.graph_featurizer.featurize([mol])[0]
            
            # Mordred features through DeepChem
            mordred_features = self.mordred_featurizer.featurize([mol])[0]
            
            return np.concatenate([
                graph_features.get_atom_features().flatten(),
                graph_features.get_bond_features().flatten(),
                mordred_features
            ])
        except:
            return np.zeros(2048)  # Reasonable default size
    
    def _get_polymer_patterns(self, mol: Chem.Mol) -> np.ndarray:
        """Identify polymer-specific structural patterns."""
        patterns = []
        
        # Common polymer substructures
        substructures = [
            'CC(C)CC',  # Polyethylene-like
            'CC(=O)O',  # Ester group
            'c1ccccc1',  # Aromatic ring
            'CN',       # Amine
            'C=O',      # Carbonyl
            'CO',       # Ether
            'CS',       # Thioether
            'C=C',      # Alkene
            'C#C',      # Alkyne
            'C(F)(F)F'  # Trifluoromethyl
        ]
        
        # Count occurrences of each pattern
        for smarts in substructures:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                count = len(mol.GetSubstructMatches(pattern))
                patterns.append(count)
            else:
                patterns.append(0)
        
        return np.array(patterns)
    
    def _get_empty_features(self) -> Dict[str, np.ndarray]:
        """Return empty feature arrays with correct shapes."""
        return {
            'basic': np.zeros(12),
            'topological': np.zeros(13),
            'electronic': np.zeros(10),
            'structural': np.zeros(10),
            'mordred': np.zeros(1613),
            'deepchem': np.zeros(2048),
            'patterns': np.zeros(10)
        } 