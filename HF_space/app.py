import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(csv_path: str):
    """Load the CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df, f"✓ Loaded {len(df):,} records with {len(df.columns)} columns"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def identify_significant_columns(df: pd.DataFrame, min_non_null: float = 0.8, min_unique: int = 10):
    """Identify significant columns for stratification."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    significant_cols = []
    
    for col in numeric_cols:
        non_null_pct = df[col].notna().sum() / len(df)
        unique_count = df[col].nunique()
        
        if non_null_pct >= min_non_null and unique_count >= min_unique:
            significant_cols.append(col)
    
    return significant_cols


def create_stratification_labels(df: pd.DataFrame, significant_cols: list, n_bins: int = 10):
    """Create stratification labels."""
    priority_cols = ['VentRate', 'AtrialRate', 'pr', 'qrs', 'qt', 'avgrrinterval', 
                     'paxis', 'raxis', 'taxis']
    
    cols_to_use = []
    for col in priority_cols:
        if col in significant_cols:
            cols_to_use.append(col)
    
    if len(cols_to_use) < 3:
        remaining = [c for c in significant_cols if c not in cols_to_use]
        cols_to_use.extend(remaining[:5])
    
    stratification_labels = []
    
    for idx, row in df.iterrows():
        label_parts = []
        for col in cols_to_use[:5]:
            if pd.notna(row[col]):
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max > col_min:
                    bin_num = int((row[col] - col_min) / (col_max - col_min) * n_bins)
                    bin_num = min(bin_num, n_bins - 1)
                    label_parts.append(f"{col}_{bin_num}")
                else:
                    label_parts.append(f"{col}_0")
            else:
                label_parts.append(f"{col}_nan")
        
        stratification_labels.append("_".join(label_parts))
    
    return pd.Series(stratification_labels, index=df.index)


def analyze_stratification(csv_path: str, records_per_fold: int = 1000):
    """Analyze and visualize stratification."""
    if csv_path is None or csv_path == "":
        return None, "Please provide a CSV file path"
    
    # Check if file exists
    if not Path(csv_path).exists():
        return None, f"File not found: {csv_path}\n\nPlease ensure the CSV file exists or run the k-fold script first."
    
    # Load data
    try:
        df, msg = load_data(csv_path)
    except Exception as e:
        return None, f"Error loading file: {str(e)}"
    
    if df is None:
        return None, msg
    
    # Check if fold column exists
    if 'fold' not in df.columns:
        return None, "CSV file does not contain 'fold' column. Please run the k-fold script first:\n\npython scripts/make_k_fold_deepfake_ecg.py"
    
    # Identify significant columns
    significant_cols = identify_significant_columns(df)
    
    # Create stratification labels
    strat_labels = create_stratification_labels(df, significant_cols)
    df['stratification_label'] = strat_labels
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Fold distribution
    ax1 = axes[0, 0]
    fold_counts = df['fold'].value_counts().sort_index()
    ax1.bar(fold_counts.index, fold_counts.values, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.set_xlabel('Fold Number', fontsize=12)
    ax1.set_ylabel('Number of Records', fontsize=12)
    ax1.set_title('Distribution of Records Across Folds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. VentRate distribution by fold
    ax2 = axes[0, 1]
    if 'VentRate' in df.columns:
        sample_folds = sorted(df['fold'].unique())[:10]
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_folds)))
        for i, fold in enumerate(sample_folds):
            fold_data = df[df['fold'] == fold]['VentRate']
            ax2.hist(fold_data.dropna(), bins=30, alpha=0.5, label=f'Fold {fold}', color=colors[i])
        ax2.set_xlabel('Ventricular Rate (bpm)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Ventricular Rate Distribution by Fold', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # 3. Boxplot of key parameters
    ax3 = axes[1, 0]
    if 'VentRate' in df.columns:
        sample_folds = sorted(df['fold'].unique())[:15]
        plot_data = []
        plot_labels = []
        for fold in sample_folds:
            fold_data = df[df['fold'] == fold]['VentRate'].dropna()
            if len(fold_data) > 0:
                plot_data.append(fold_data.values)
                plot_labels.append(f'F{fold}')
        if plot_data:
            bp = ax3.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            ax3.set_xlabel('Fold Number', fontsize=12)
            ax3.set_ylabel('Ventricular Rate (bpm)', fontsize=12)
            ax3.set_title('Ventricular Rate Distribution Across Folds (Boxplot)', 
                         fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
    
    # 4. Stratification statistics
    ax4 = axes[1, 1]
    fold_stats = df.groupby('fold').agg({
        'VentRate': ['mean', 'std'],
        'AtrialRate': ['mean', 'std'] if 'AtrialRate' in df.columns else []
    }).reset_index()
    
    if 'VentRate' in df.columns:
        ax4_twin = ax4.twinx()
        x_pos = np.arange(len(fold_stats))
        width = 0.35
        
        ax4.bar(x_pos - width/2, fold_stats[('VentRate', 'mean')], 
               width, label='Mean', alpha=0.7, color='steelblue')
        ax4_twin.bar(x_pos + width/2, fold_stats[('VentRate', 'std')], 
                    width, label='Std Dev', alpha=0.7, color='coral')
        
        ax4.set_xlabel('Fold Number', fontsize=12)
        ax4.set_ylabel('Mean Ventricular Rate (bpm)', fontsize=12, color='steelblue')
        ax4_twin.set_ylabel('Std Dev Ventricular Rate (bpm)', fontsize=12, color='coral')
        ax4.set_title('Mean and Std Dev of Ventricular Rate by Fold', 
                     fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(fold_stats['fold'].values)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_statistics_table(csv_path: str):
    """Create statistics table."""
    if csv_path is None or csv_path == "":
        return None
    
    if not Path(csv_path).exists():
        return None
    
    try:
        df, msg = load_data(csv_path)
    except Exception as e:
        return None
    
    if df is None or 'fold' not in df.columns:
        return None
    
    # Create statistics table
    stats_list = []
    
    if 'VentRate' in df.columns:
        for fold in sorted(df['fold'].unique()):
            fold_data = df[df['fold'] == fold]
            stats_list.append({
                'Fold': fold,
                'Count': len(fold_data),
                'Mean VentRate': fold_data['VentRate'].mean(),
                'Std VentRate': fold_data['VentRate'].std(),
                'Min VentRate': fold_data['VentRate'].min(),
                'Max VentRate': fold_data['VentRate'].max(),
            })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df


# Create Gradio interface
def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="ECG Deepfake K-Fold Stratification Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # ECG Deepfake K-Fold Stratification Analyzer
        
        This tool visualizes the stratified k-fold splitting of the ECG deepfake dataset.
        Upload or provide the path to the CSV file with fold assignments to analyze the stratification.
        
        **Features:**
        - Visualize fold distribution
        - Analyze parameter distributions across folds
        - View stratification statistics
        """)
        
        with gr.Row():
            csv_input = gr.Textbox(
                label="CSV File Path",
                value="/global/D1/homes/vajira/data/SEARCH/deepfake_ecgs/filtered_all_normal_ECGs/deep_fake_ecg_k_fold.csv",
                placeholder="Enter path to CSV file with fold column"
            )
            records_per_fold = gr.Slider(
                minimum=500,
                maximum=2000,
                value=1000,
                step=100,
                label="Records per Fold (for reference)"
            )
        
        with gr.Row():
            analyze_btn = gr.Button("Analyze Stratification", variant="primary")
        
        with gr.Row():
            plot_output = gr.Plot(label="Stratification Visualization")
        
        with gr.Row():
            stats_output = gr.Dataframe(
                label="Fold Statistics",
                headers=["Fold", "Count", "Mean VentRate", "Std VentRate", "Min VentRate", "Max VentRate"]
            )
        
        info_output = gr.Textbox(label="Information", lines=3)
        
        analyze_btn.click(
            fn=analyze_stratification,
            inputs=[csv_input, records_per_fold],
            outputs=[plot_output, info_output]
        )
        
        analyze_btn.click(
            fn=create_statistics_table,
            inputs=[csv_input],
            outputs=[stats_output]
        )
        
        # Auto-load on start
        demo.load(
            fn=analyze_stratification,
            inputs=[csv_input, records_per_fold],
            outputs=[plot_output, info_output]
        )
        
        demo.load(
            fn=create_statistics_table,
            inputs=[csv_input],
            outputs=[stats_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)

