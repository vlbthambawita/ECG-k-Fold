# ECG Deepfake K-Fold Stratification Analyzer

This Hugging Face Space provides an interactive interface to visualize and analyze the stratified k-fold splitting of the ECG deepfake dataset.

## Features

- **Fold Distribution Visualization**: See how records are distributed across different folds
- **Parameter Analysis**: Analyze key ECG parameters (Ventricular Rate, Atrial Rate, etc.) across folds
- **Stratification Statistics**: View detailed statistics for each fold
- **Interactive Interface**: Easy-to-use Gradio interface

## Usage

1. Upload or provide the path to your CSV file with fold assignments
2. Click "Analyze Stratification" to generate visualizations
3. View statistics in the table below

## Requirements

- CSV file must contain a 'fold' column with fold assignments
- File should be in the same format as the original ECG dataset

## Environment

This space uses the "search" environment as specified.

