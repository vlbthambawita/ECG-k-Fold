#!/usr/bin/env python3
"""
Script to create stratified k-fold splits for ECG deepfake dataset.

This script:
1. Loads the ECG CSV file
2. Identifies significant columns for stratification
3. Creates stratified folds (~1000 records per fold)
4. Adds a 'fold' column to the dataset
5. Saves the new CSV file with fold assignments
6. Generates visualization of the stratification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import warnings

warnings.filterwarnings("ignore")


def identify_significant_columns(df: pd.DataFrame, min_non_null: float = 0.8, min_unique: int = 10) -> List[str]:
    """
    Identify columns with significant information for stratification.
    
    Args:
        df: Input dataframe
        min_non_null: Minimum fraction of non-null values (default: 0.8)
        min_unique: Minimum number of unique values (default: 10)
        
    Returns:
        List of significant column names
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    significant_cols = []
    
    for col in numeric_cols:
        non_null_pct = df[col].notna().sum() / len(df)
        unique_count = df[col].nunique()
        
        if non_null_pct >= min_non_null and unique_count >= min_unique:
            significant_cols.append(col)
    
    return significant_cols


def create_stratification_labels(df: pd.DataFrame, significant_cols: List[str], n_bins: int = 10) -> pd.Series:
    """
    Create stratification labels by binning significant columns.
    
    Args:
        df: Input dataframe
        significant_cols: List of significant columns to use
        n_bins: Number of bins for each column (default: 10)
        
    Returns:
        Series of stratification labels
    """
    # Select top significant columns that are most informative
    # Use key ECG parameters: VentRate, AtrialRate, pr, qrs, qt, avgrrinterval
    priority_cols = ['VentRate', 'AtrialRate', 'pr', 'qrs', 'qt', 'avgrrinterval', 
                     'paxis', 'raxis', 'taxis']
    
    # Use priority columns that exist, fallback to other significant columns
    cols_to_use = []
    for col in priority_cols:
        if col in significant_cols:
            cols_to_use.append(col)
    
    # If we don't have enough priority cols, add more significant ones
    if len(cols_to_use) < 3:
        remaining = [c for c in significant_cols if c not in cols_to_use]
        cols_to_use.extend(remaining[:5])
    
    # Create bins for each column and combine them
    stratification_labels = []
    
    for idx, row in df.iterrows():
        label_parts = []
        for col in cols_to_use[:5]:  # Use top 5 columns
            if pd.notna(row[col]):
                # Create bins
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max > col_min:
                    bin_num = int((row[col] - col_min) / (col_max - col_min) * n_bins)
                    bin_num = min(bin_num, n_bins - 1)  # Ensure within range
                    label_parts.append(f"{col}_{bin_num}")
                else:
                    label_parts.append(f"{col}_0")
            else:
                label_parts.append(f"{col}_nan")
        
        stratification_labels.append("_".join(label_parts))
    
    return pd.Series(stratification_labels, index=df.index)


def create_stratified_folds(df: pd.DataFrame, records_per_fold: int = 1000) -> pd.DataFrame:
    """
    Create stratified k-folds based on significant columns.
    
    Args:
        df: Input dataframe
        records_per_fold: Target number of records per fold (default: 1000)
        
    Returns:
        Dataframe with added 'fold' column
    """
    print(f"Creating stratified folds for {len(df):,} records...")
    print(f"Target records per fold: {records_per_fold:,}")
    
    # Identify significant columns
    print("\nIdentifying significant columns...")
    significant_cols = identify_significant_columns(df)
    print(f"Found {len(significant_cols)} significant columns")
    
    # Create stratification labels
    print("\nCreating stratification labels...")
    strat_labels = create_stratification_labels(df, significant_cols)
    print(f"Created {strat_labels.nunique()} unique stratification groups")
    
    # Calculate number of folds needed
    n_folds = max(1, len(df) // records_per_fold)
    print(f"\nCreating {n_folds} folds...")
    
    # Create fold assignments
    df_with_folds = df.copy()
    df_with_folds['stratification_label'] = strat_labels
    
    # Use StratifiedKFold approach
    fold_assignments = np.zeros(len(df), dtype=int)
    
    # Group by stratification label and assign folds
    for strat_label in strat_labels.unique():
        mask = strat_labels == strat_label
        indices = df_with_folds[mask].index.values
        
        if len(indices) > 0:
            # Shuffle indices within each stratification group
            np.random.seed(42)
            shuffled_indices = np.random.permutation(indices)
            
            # Assign folds in round-robin fashion
            for i, idx in enumerate(shuffled_indices):
                fold_assignments[df.index.get_loc(idx)] = i % n_folds
    
    df_with_folds['fold'] = fold_assignments + 1  # 1-indexed folds
    
    # Ensure folds are balanced
    fold_counts = df_with_folds['fold'].value_counts().sort_index()
    print(f"\nFold distribution:")
    print(fold_counts)
    print(f"\nFold size statistics:")
    print(fold_counts.describe())
    
    return df_with_folds


def visualize_stratification(df: pd.DataFrame, output_path: Path):
    """
    Create visualizations of the stratification.
    
    Args:
        df: Dataframe with fold assignments
        output_path: Path to save visualizations
    """
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Fold distribution
    ax1 = axes[0, 0]
    fold_counts = df['fold'].value_counts().sort_index()
    ax1.bar(fold_counts.index, fold_counts.values, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Fold Number', fontsize=12)
    ax1.set_ylabel('Number of Records', fontsize=12)
    ax1.set_title('Distribution of Records Across Folds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. VentRate distribution by fold
    ax2 = axes[0, 1]
    if 'VentRate' in df.columns:
        for fold in sorted(df['fold'].unique()):
            fold_data = df[df['fold'] == fold]['VentRate']
            ax2.hist(fold_data.dropna(), bins=30, alpha=0.5, label=f'Fold {fold}')
        ax2.set_xlabel('Ventricular Rate (bpm)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Ventricular Rate Distribution by Fold', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Key parameters by fold (boxplot)
    ax3 = axes[1, 0]
    if 'VentRate' in df.columns and 'pr' in df.columns:
        sample_folds = sorted(df['fold'].unique())[:10]  # Show first 10 folds
        plot_data = []
        plot_labels = []
        for fold in sample_folds:
            fold_data = df[df['fold'] == fold]['VentRate'].dropna()
            if len(fold_data) > 0:
                plot_data.append(fold_data.values)
                plot_labels.append(f'F{fold}')
        if plot_data:
            ax3.boxplot(plot_data, labels=plot_labels)
            ax3.set_xlabel('Fold Number', fontsize=12)
            ax3.set_ylabel('Ventricular Rate (bpm)', fontsize=12)
            ax3.set_title('Ventricular Rate Distribution Across Folds (Boxplot)', 
                         fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
    
    # 4. Stratification label distribution
    ax4 = axes[1, 1]
    if 'stratification_label' in df.columns:
        top_labels = df['stratification_label'].value_counts().head(20)
        ax4.barh(range(len(top_labels)), top_labels.values, alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(top_labels)))
        ax4.set_yticklabels([l[:30] + '...' if len(l) > 30 else l for l in top_labels.index], 
                            fontsize=8)
        ax4.set_xlabel('Number of Records', fontsize=12)
        ax4.set_title('Top 20 Stratification Labels', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def main():
    """Main function to create k-fold splits."""
    # File paths
    input_csv = Path('/global/D1/homes/vajira/data/SEARCH/deepfake_ecgs/filtered_all_normal_ECGs/filtered_all_normals_121977_ground_truth.csv')
    output_dir = input_csv.parent
    output_csv = output_dir / 'deep_fake_ecg_k_fold.csv'
    output_viz = output_dir / 'deep_fake_ecg_k_fold_stratification.png'
    
    print("=" * 80)
    print("ECG Deepfake K-Fold Stratification Script")
    print("=" * 80)
    print(f"\nInput file: {input_csv}")
    print(f"Output file: {output_csv}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Create stratified folds
    df_with_folds = create_stratified_folds(df, records_per_fold=1000)
    
    # Remove temporary stratification_label column before saving
    df_output = df_with_folds.drop(columns=['stratification_label'])
    
    # Save output CSV
    print(f"\nSaving output CSV to {output_csv}...")
    df_output.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df_output):,} records with fold assignments")
    
    # Create visualization
    visualize_stratification(df_with_folds, output_viz)
    
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - CSV: {output_csv}")
    print(f"  - Visualization: {output_viz}")
    print(f"\nFold statistics:")
    print(df_with_folds['fold'].value_counts().sort_index().describe())


if __name__ == "__main__":
    main()

