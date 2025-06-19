import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))  # src klasörünü import yoluna ekle

from utils import smiles_to_features
import pandas as pd

# Load the training CSV
train = pd.read_csv('data/train.csv')

# Show the first 5 SMILES
print(f"First 5 SMILES:\n{train['SMILES'].head()}")

# Test the first SMILES string
example = train['SMILES'].iloc[0]
features = smiles_to_features(example)

# Print results
print("Example SMILES:", example)
print("Extracted features:", features)


