import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.EState import EState_VSA
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleDeg, GetDihedralDeg
from typing import Optional, List, Dict
from sklearn.preprocessing import StandardScaler

def generate_3d_conformer(mol, num_conf=10):
    """Generate and optimize 3D conformers."""
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, randomSeed=42)
    
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
    
    if not energies:
        return None
    
    # Get lowest energy conformer
    best_conf_id = min(energies, key=lambda x: x[1])[0]
    return mol.GetConformer(best_conf_id)

def get_3d_descriptors(mol, conf) -> List[float]:
    """Calculate 3D molecular descriptors with enhanced error handling."""
    descriptors = []
    
    try:
        # Basic 3D descriptors with proper error handling
        descriptor_functions = [
            (Descriptors.PMI1, "PMI1"),
            (Descriptors.PMI2, "PMI2"),
            (Descriptors.PMI3, "PMI3"),
            (Descriptors.NPR1, "NPR1"),
            (Descriptors.NPR2, "NPR2"),
            (Descriptors.RadiusOfGyration, "RadiusOfGyration"),
            (Descriptors.Asphericity, "Asphericity"),
            (Descriptors.Eccentricity, "Eccentricity"),
            (Descriptors.InertialShapeFactor, "InertialShapeFactor"),
            (Descriptors.SpherocityIndex, "SpherocityIndex")
        ]

        for desc_func, desc_name in descriptor_functions:
            try:
                value = desc_func(mol, conf) if desc_name in ["PMI1", "PMI2", "PMI3"] else desc_func(mol)
                descriptors.append(value)
            except Exception as e:
                print(f"Warning: Failed to calculate {desc_name}: {str(e)}")
                descriptors.append(0.0)
        
        # Enhanced 3D shape descriptors
        try:
            coords = conf.GetPositions()
            center_of_mass = np.mean(coords, axis=0)
            centered_coords = coords - center_of_mass
            
            # Calculate principal axes and moments
            inertia_tensor = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    inertia_tensor[i,j] = np.sum(centered_coords[:,i] * centered_coords[:,j])
            
            eigenvals, eigenvecs = np.linalg.eigh(inertia_tensor)
            
            # Additional shape descriptors
            descriptors.extend([
                np.sqrt(np.sum(centered_coords**2) / len(coords)),  # RMSD from center
                np.max(np.linalg.norm(centered_coords, axis=1)),    # Maximum extent
                np.min(np.linalg.norm(centered_coords, axis=1)),    # Minimum extent
                np.std(np.linalg.norm(centered_coords, axis=1))     # Std dev of atomic distances
            ])
            
        except Exception as e:
            print(f"Warning: Failed to calculate enhanced 3D descriptors: {str(e)}")
            descriptors.extend([0.0] * 4)
        
        # Calculate bond lengths, angles, and dihedrals
        bonds = mol.GetBonds()
        angles = []
        dihedrals = []
        
        for bond in bonds:
            try:
                # Bond length
                length = GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                descriptors.append(length)
                
                # Bond angles
                for neighbor in bond.GetBeginAtom().GetNeighbors():
                    if neighbor.GetIdx() != bond.GetEndAtomIdx():
                        try:
                            angle = GetAngleDeg(
                                conf,
                                neighbor.GetIdx(),
                                bond.GetBeginAtomIdx(),
                                bond.GetEndAtomIdx()
                            )
                            angles.append(angle)
                        except:
                            continue
                
                # Dihedral angles
                for n1 in bond.GetBeginAtom().GetNeighbors():
                    for n2 in bond.GetEndAtom().GetNeighbors():
                        if (n1.GetIdx() != bond.GetEndAtomIdx() and 
                            n2.GetIdx() != bond.GetBeginAtomIdx()):
                            try:
                                dihedral = GetDihedralDeg(
                                    conf,
                                    n1.GetIdx(),
                                    bond.GetBeginAtomIdx(),
                                    bond.GetEndAtomIdx(),
                                    n2.GetIdx()
                                )
                                dihedrals.append(dihedral)
                            except:
                                continue
            except Exception as e:
                print(f"Warning: Failed to calculate geometric features: {str(e)}")
                continue
        
        # Add statistical measures of angles and dihedrals
        angle_stats = [np.mean, np.std, np.min, np.max]
        for stat_func in angle_stats:
            try:
                if angles:
                    descriptors.append(stat_func(angles))
                else:
                    descriptors.append(0.0)
                    
                if dihedrals:
                    descriptors.append(stat_func(dihedrals))
                else:
                    descriptors.append(0.0)
            except:
                descriptors.extend([0.0, 0.0])
            
    except Exception as e:
        print(f"Error calculating 3D descriptors: {str(e)}")
        return [0.0] * 22  # Return zeros for all descriptors if calculation fails
        
    return descriptors

def get_polymer_specific_descriptors(mol) -> List[float]:
    """Calculate polymer-specific descriptors."""
    descriptors = []
    
    try:
        # Ring analysis
        ri = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ri.AtomRings()]
        
        descriptors.extend([
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            np.mean(ring_sizes) if ring_sizes else 0,
            np.std(ring_sizes) if ring_sizes else 0,
            max(ring_sizes) if ring_sizes else 0,
            min(ring_sizes) if ring_sizes else 0
        ])
        
        # Polymer chain features
        chain_length = len(mol.GetAtoms())
        branching_points = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
        terminal_groups = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() == 1)
        
        descriptors.extend([
            chain_length,
            branching_points,
            terminal_groups,
            branching_points / chain_length if chain_length > 0 else 0
        ])
        
    except Exception as e:
        print(f"Error calculating polymer descriptors: {str(e)}")
        return [0] * 15  # Return zeros if calculation fails
    
    return descriptors

def get_molecular_descriptors(mol) -> List[float]:
    """Calculate comprehensive molecular descriptors."""
    descriptors = []
    
    # Basic properties
    descriptors.extend([
        Descriptors.ExactMolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol),
        Descriptors.FractionCSP3(mol),
    ])
    
    # Topological descriptors
    descriptors.extend([
        Descriptors.BertzCT(mol),
        Descriptors.Chi0n(mol),
        Descriptors.Chi1n(mol),
        Descriptors.Chi2n(mol),
        Descriptors.Chi3n(mol),
        Descriptors.Chi4n(mol),
    ])
    
    # Electronic descriptors
    descriptors.extend([
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
        Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MinAbsPartialCharge(mol),
    ])
    
    return descriptors

def get_fingerprints(mol) -> Dict[str, np.ndarray]:
    """Generate multiple types of molecular fingerprints with modern RDKit methods."""
    fps = {}
    
    try:
        # Morgan (ECFP) fingerprints with feature invariants
        fps['morgan'] = np.zeros(2048)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            2,  # radius 2 for ECFP4
            nBits=2048,
            useFeatures=True  # Use feature-based invariants
        )
        fps['morgan'][list(morgan_fp.GetOnBits())] = 1
        
        # Morgan fingerprints with chirality
        fps['morgan_chiral'] = np.zeros(2048)
        morgan_chiral_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            2,
            nBits=2048,
            useChirality=True
        )
        fps['morgan_chiral'][list(morgan_chiral_fp.GetOnBits())] = 1
        
        # MACCS keys (166 bits)
        maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
        fps['maccs'] = np.array(list(maccs_fp.ToBitString())).astype(int)
        
        # Modern topological fingerprint (2048 bits)
        fps['topological'] = np.zeros(2048)
        rdkit_fp = Chem.RDKFingerprint(
            mol,
            fpSize=2048,
            minPath=1,
            maxPath=7,
            useHs=True
        )
        fps['topological'][list(rdkit_fp.GetOnBits())] = 1
        
        # Atom pairs fingerprint with modern parameters
        fps['atom_pairs'] = np.zeros(2048)
        ap_fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
            mol,
            nBits=2048,
            minLength=1,
            maxLength=30,
            includeChirality=True
        )
        fps['atom_pairs'][list(ap_fp.GetOnBits())] = 1
        
        # Pattern fingerprint
        pattern_fp = Chem.PatternFingerprint(mol, fpSize=2048)
        fps['pattern'] = np.zeros(2048)
        fps['pattern'][list(pattern_fp.GetOnBits())] = 1
        
        # Layered fingerprint
        layered_fp = Chem.LayeredFingerprint(mol, fpSize=2048)
        fps['layered'] = np.zeros(2048)
        fps['layered'][list(layered_fp.GetOnBits())] = 1
        
        # 3D shape fingerprint if 3D conformer exists
        try:
            if mol.GetNumConformers() > 0:
                shape_fp = AllChem.Generate3DDistanceMatrix(mol, confId=0, maxDist=20)
                shape_fp = np.array(shape_fp).flatten()
                shape_fp = np.clip(shape_fp, 0, 20)  # Clip distances to reasonable range
                shape_fp = shape_fp / 20.0  # Normalize
                fps['shape_3d'] = shape_fp
        except:
            pass
            
    except Exception as e:
        print(f"Error generating fingerprints: {str(e)}")
        # Return empty fingerprints in case of error
        fps = {
            'morgan': np.zeros(2048),
            'morgan_chiral': np.zeros(2048),
            'maccs': np.zeros(166),
            'topological': np.zeros(2048),
            'atom_pairs': np.zeros(2048),
            'pattern': np.zeros(2048),
            'layered': np.zeros(2048)
        }
    
    return fps

def smiles_to_features(smiles: str) -> Optional[np.ndarray]:
    """Convert SMILES to feature vector with enhanced descriptors and fingerprints."""
    try:
        # Replace * with any atom (e.g., carbon) for RDKit processing
        processed_smiles = smiles.replace('*', '[C]')
        mol = Chem.MolFromSmiles(processed_smiles)
        if mol is None:
            return None
        
        # Generate 3D conformer
        conf = generate_3d_conformer(mol)
        if conf is None:
            return None
            
        # Collect all features
        features = []
        
        # Basic molecular descriptors
        features.extend(get_molecular_descriptors(mol))
        
        # 3D conformer-based descriptors
        features.extend(get_3d_descriptors(mol, conf))
        
        # Polymer-specific descriptors
        features.extend(get_polymer_specific_descriptors(mol))
        
        # Fingerprints
        fps = get_fingerprints(mol)
        for fp_type in ['morgan', 'maccs', 'topological', 'atom_pairs', 'morgan_chiral', 'pattern', 'layered', 'shape_3d']:
            features.extend(fps[fp_type])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return None
