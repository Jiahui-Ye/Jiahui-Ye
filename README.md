# Online PU Learning (LINEX / BLINEX)

A simple experimental framework for Positive-Unlabeled (PU) Learning with multiple models and missing-data settings.

------

## Features

- Support for PU Learning with only positive + unlabeled data
- Multiple models:
  - LINEX / BLINEX
- Multiple missing mechanisms:
  - PIC
- GPU acceleration (via PyTorch)

------

## Model Hyperparameter Ranges

### BLINEX
- lambda : 0.4 – 0.7
- b      : 1e⁻³ – 1
- a      : 0.3 – 0.6

### LINEX
- a      : -5 – -1

------

## Project Structure

.
├── model.py       # Main script (run experiments)
├── data.py        # Data loading & preprocessing
├── linex.py       # LINEX / BLINEX models
├── evaluate.py    # Evaluation functions

------

## Requirements

### Python Version

- Python 3.8+ (recommended 3.9 / 3.10)

------

### Install Dependencies

pip install numpy pandas torch

------

### Optional (recommended)

pip install scikit-learn tqdm

------

### requirements.txt (optional)

numpy>=1.20
pandas>=1.3
torch>=1.10
scikit-learn>=1.0
tqdm>=4.60

Install with:

pip install -r requirements.txt

------

### GPU Support (Optional)

- Automatically uses GPU if available (via PyTorch)
- Falls back to CPU otherwise

------

## Data Format

Two formats are supported:

### 1. CSV format

label, feature1, feature2, ...

### 2. NPY + CSV

dataset_X_array.npy
dataset_label.csv

------

## How to Run

### Step 1: Set dataset path

Edit in model.py:

DATA_DIR_RELATIVE = '../../data'
# or
DATA_DIR_ABSOLUTE = '/your/data/path'

------

### Step 2: Configure experiment

In model.py:

DATASET_LIST = ['your_dataset']
LABELING_RATIO = 0.3
N_EPOCHS = 50

------

### Step 3: Run

python model.py

------

## Notes

- GPU will be used automatically if available
- Default train/test split: 80/20
- Features are normalized automatically
- Supports imbalanced datasets (optional sampling)

------

## Tips

- Start with small datasets to verify setup
- Check data path carefully if loading fails
- Use fewer epochs for quick testing
