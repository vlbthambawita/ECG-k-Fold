# ECG Deepfake K-Fold Stratification

This project provides tools for creating and visualizing stratified k-fold splits of the ECG deepfake dataset.

## Files Created

### Scripts
- **`scripts/make_k_fold_deepfake_ecg.py`**: Main script to create stratified k-fold splits
  - Identifies significant columns with >80% non-null values and >10 unique values
  - Uses key ECG parameters (VentRate, AtrialRate, pr, qrs, qt, etc.) for stratification
  - Creates approximately 1000 records per fold
  - Generates visualization PNG file

### Hugging Face Space
- **`HF_space/app.py`**: Gradio application for interactive visualization
- **`HF_space/requirements.txt`**: Python dependencies
- **`HF_space/README.md`**: Space configuration (uses "search" environment)
- **`HF_space/DESCRIPTION.md`**: Space description

## Usage

### Step 1: Create K-Fold Splits

```bash
cd /work/vajira/SEARCH/ECG-k-Fold
python scripts/make_k_fold_deepfake_ecg.py
```

**Output:**
- `deep_fake_ecg_k_fold.csv`: CSV file with fold assignments (same location as input)
- `deep_fake_ecg_k_fold_stratification.png`: Visualization of stratification

### Step 2: Visualize with Hugging Face Space

**Option A: Run Locally**
```bash
cd HF_space
pip install -r requirements.txt
python app.py
```

**Option B: Deploy to Hugging Face**
1. Push `HF_space/` directory to a Hugging Face Space
2. Space will automatically use Python 3.11 and "search" environment
3. Access the interactive interface through the Hugging Face Space

## Stratification Method

The stratification ensures balanced distribution across folds by:

1. **Identifying Significant Columns**: 
   - Filters columns with >80% non-null values
   - Requires >10 unique values per column
   - Prioritizes key ECG parameters: VentRate, AtrialRate, pr, qrs, qt, avgrrinterval, paxis, raxis, taxis

2. **Creating Stratification Labels**:
   - Bins continuous values into 10 bins per parameter
   - Combines bins from top 5 significant columns
   - Creates unique stratification groups

3. **Assigning Folds**:
   - Uses round-robin assignment within each stratification group
   - Ensures balanced distribution (~1000 records per fold)
   - Maintains stratification integrity across folds

## Features

- ✅ Stratified k-fold splitting (~1000 records per fold)
- ✅ Uses significant columns with meaningful values
- ✅ Visualizations of fold distribution
- ✅ Interactive Hugging Face Space interface
- ✅ Detailed statistics per fold
- ✅ Uses "search" environment as specified

## Output Format

The output CSV (`deep_fake_ecg_k_fold.csv`) contains all original columns plus:
- **`fold`**: Integer fold number (1-indexed)

Example:
```csv
TestID,patid,VentRate,AtrialRate,...,fold
140000,0,60,60,...,1
140001,1,65,65,...,2
...
```

## Visualization

The visualization includes:
1. **Fold Distribution**: Bar chart showing record counts per fold
2. **Parameter Distribution**: Histograms of VentRate across folds
3. **Boxplots**: Statistical distribution of key parameters
4. **Stratification Statistics**: Mean and standard deviation per fold

## Notes

- Input CSV: `/global/D1/homes/vajira/data/SEARCH/deepfake_ecgs/filtered_all_normal_ECGs/filtered_all_normals_121977_ground_truth.csv`
- Output CSV: Same directory as input, named `deep_fake_ecg_k_fold.csv`
- Environment: Uses "search" environment as specified for Hugging Face Space

