# Open Polymer Prediction Challenge

This project provides a complete framework for tackling the NeurIPS 2025 Open Polymer Prediction competition. The primary goal is to predict five key polymer properties—Glass Transition Temperature (Tg), Fractional Free Volume (FFV), Thermal Conductivity (Tc), Density, and Radius of Gyration (Rg)—directly from their SMILES representations using advanced machine learning models.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Making Submissions](#making-submissions)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Overview

Polymers are fundamental materials in modern science and industry, yet predicting their properties remains a complex challenge. This project leverages graph-based neural networks and ensemble methods to learn the intricate relationship between a polymer's molecular structure (represented as a SMILES string) and its macroscopic properties. Our solution is designed to be modular, extensible, and easy to use, providing a strong baseline for the competition.

## Project Structure

The repository is organized as follows:

```
.
├── config/
│   └── training_config.json    # Configuration for training parameters
├── data/
│   ├── processed/              # Processed data for modeling
│   └── raw/                    # Raw competition data
├── models/                     # Saved model checkpoints
├── notebooks/
│   └── 01_starter.ipynb        # Exploratory data analysis and baseline models
├── src/
│   ├── ensemble.py             # Code for ensembling models
│   ├── gnn_model.py            # Graph Neural Network model definition
│   ├── polymer_features.py     # Feature engineering from SMILES
│   ├── train_orchestrator.py   # Main training script
│   └── utils.py                # Utility functions
├── .gitignore                  # Files to be ignored by Git
├── LICENSE                     # Project license
├── make_submission.py          # Script to generate the final submission file
├── README.md                   # This readme file
└── requirements.txt            # Python dependencies
```

## Getting Started

Follow these instructions to set up the project environment and start running experiments.

### Prerequisites

- Python 3.8 or higher
- Conda or venv for environment management (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Cansu126/open-polymer-prediction.git
    cd open-polymer-prediction
    ```

2.  **Create and activate a virtual environment:**
    - Using `conda`:
      ```bash
      conda create -n polymer-env python=3.8
      conda activate polymer-env
      ```
    - Using `venv`:
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
      ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the data:**
    Download the official competition data and place the `train.csv` and `test.csv` files into the `data/raw/` directory.

## Usage

This project is designed to be run from the command line for both training and submission generation.

### Training Models

To train the models, run the main training orchestrator script:

```bash
python src/train_orchestrator.py
```

This script will handle feature engineering, model training, and saving the best model checkpoints to the `models/` directory. You can customize the training process by modifying `config/training_config.json`.

### Making Submissions

Once you have a trained model, you can generate a `submission.csv` file for the competition leaderboard.

Run the submission script:

```bash
python make_submission.py
```

This will load the test data, apply the trained model to generate predictions, and format the output into the required `submission.csv` file in the root directory.

## Models

This project explores several models, including:
- **Graph Neural Networks (GNNs)**: To capture the graph structure of molecules.
- **Transformers**: To learn from the sequential nature of SMILES strings.
- **Ensemble Methods**: To combine the strengths of different models for robust predictions.

The best-performing models are saved and used for the final submission.

## Contributing

Contributions are welcome! If you have ideas for improvements, please open an issue to discuss your suggestions. Pull requests are also appreciated.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details. 