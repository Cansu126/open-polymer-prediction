import pandas as pd
import numpy as np

# Load test data
property_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
test = pd.read_csv('data/test.csv')

# Create dummy predictions (all zeros)
dummy_preds = np.zeros((len(test), len(property_cols)))

# Create submission DataFrame
submission = test[['id']].copy()
for i, col in enumerate(property_cols):
    submission[col] = dummy_preds[:, i]
submission = submission[['id'] + property_cols]
submission.to_csv('submission.csv', index=False)
print('Dummy submission file saved as submission.csv')
print(submission.head()) 