# NeurIPS - Open Polymer Prediction 2025

Predicting polymer properties from SMILES using machine learning.

## Overview
This project is for the NeurIPS 2025 Open Polymer Prediction competition. The goal is to predict five key polymer properties (Tg, FFV, Tc, Density, Rg) from SMILES strings using ML models.

## Project Structure
- `notebooks/` - Jupyter notebooks for exploration and modeling
- `src/` - Utility scripts (feature engineering, metrics, etc.)
- `data/` - Place train/test CSVs here

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the competition data and place it in the `data/` folder.

## Workflow
- Data loading and EDA
- Feature engineering (SMILES to descriptors)
- Model training and evaluation
- Submission file creation

## Submission
- Output a `submission.csv` in the required format for the competition. 

## Submission File Creation

The submission file is created by running the final cell in the notebook. The code block to apply changes from is:

```python
property_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
submission = test[['id']].copy()
for i, col in enumerate(property_cols):
    submission[col] = stacked_preds[:, i]  # or your final predictions array
submission = submission[['id'] + property_cols]
submission.to_csv('submission.csv', index=False)
print('Submission file saved as submission.csv')
display(submission.head())
```

This code block creates a submission file by copying the test dataset, adding the predicted values, and saving it as `submission.csv`.

## Troubleshooting

If `submission.csv` was not created, it means the final prediction/saving cell in your notebook did not run successfully, or there was an error earlier in the pipeline.

Here's how to troubleshoot and fix this:

1. **Check the Final Cell**:
   - Make sure you have a cell like this at the end of your notebook:
     ```python
     property_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
     submission = test[['id']].copy()
     for i, col in enumerate(property_cols):
         submission[col] = stacked_preds[:, i]  # or your final predictions array
     submission = submission[['id'] + property_cols]
     submission.to_csv('submission.csv', index=False)
     print('Submission file saved as submission.csv')
     display(submission.head())
     ```
   - If you are using a different variable for predictions (e.g., `test_preds`, `ensemble_preds`), make sure to use that variable in place of `stacked_preds`.

2. **Check for Errors**:
   - Did any cell throw an error before the submission cell?
   - Did the notebook finish running all cells, or did it stop partway?

3. **Check the Working Directory**:
   - The file will be saved in the current working directory of the notebook.
   - In Jupyter, you can check your working directory with:
     ```python
     import os
     print(os.getcwd())
     ```
   - The file should appear in that directory.

4. **Re-run the Final Cell**:
   - If you made changes or fixed errors, re-run the final cell to generate `submission.csv`.

5. **If You Still Don't See the File**:
   - Let me know if you see any error messages, or copy the output of the final cell here.
   - If you want, I can generate a standalone script (e.g., `make_submission.py`) that you can run to create the file.

**Let me know what you see, or if you want a ready-to-run script for generating `submission.csv`!** 

## Running the Script

To run the script, you can use the following command:

```bash
python make_submission.py
```

This script will generate the submission file based on the predictions from your model.

**If you want, you can also upload a screenshot of your folder structure or the error message.**

I'm here to help you until it works—just let me know what you see after running `dir data`! 