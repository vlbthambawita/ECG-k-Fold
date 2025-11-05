# ECG Deepfake K-Fold Stratification

This repository contains scripts and tools for creating stratified k-fold splits of the ECG deepfake dataset.

## Scripts

### `make_k_fold_deepfake_ecg.py`

Main script to create stratified k-fold splits. It:
- Identifies significant columns for stratification
- Creates stratified folds (~1000 records per fold)
- Adds a 'fold' column to the dataset
- Generates visualization of the stratification

**Usage:**
```bash
python scripts/make_k_fold_deepfake_ecg.py
```

**Output:**
- `deep_fake_ecg_k_fold.csv`: CSV file with fold assignments
- `deep_fake_ecg_k_fold_stratification.png`: Visualization of stratification

## Hugging Face Space

The `HF_space/` directory contains a Gradio application for visualizing and analyzing the stratification interactively.

**To run locally:**
```bash
cd HF_space
pip install -r requirements.txt
python app.py
```

**To deploy to Hugging Face:**
1. Push the `HF_space/` directory to a Hugging Face Space
2. Configure the space to use Python 3.11
3. Set the environment to "search"

## Stratification Method

The stratification uses:
- **Key ECG parameters**: VentRate, AtrialRate, pr, qrs, qt, avgrrinterval, paxis, raxis, taxis
- **Binning strategy**: Continuous values are binned into 10 bins per parameter
- **Round-robin assignment**: Records are assigned to folds in a round-robin fashion within each stratification group
- **Balanced folds**: Ensures approximately equal distribution across folds

## Environment

Uses the "search" environment as specified.

