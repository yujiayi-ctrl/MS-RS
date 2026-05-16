import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def create_stratified_cv_splits(input_csv, output_dir, n_folds=5, random_state=42):
    """
    Create stratified sampling K-fold cross-validation data splits (without separate test set)
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Output directory
        n_folds: Number of cross-validation folds
        random_state: Random seed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    print(f"Reading data: {input_csv}")
    
    # --- Key change ---
    # Use dtype parameter to explicitly specify the 'label' column as integer type.
    # 'Int64' (capital I) is Pandas' nullable integer type, which can properly handle integer columns containing null values.
    df = pd.read_csv(input_csv, dtype={'label': 'Int64'})
    # --------------------
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Perform K-fold stratified cross-validation on all data
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # StratifiedKFold cannot handle nullable integer type 'Int64', need to convert to regular numpy array
    # .fillna(-1) is a temporary measure to fill invalid labels with -1 so that skf can work.
    # Since your input data should be clean, there should be few or no null values here.
    labels_for_skf = df['label'].fillna(-1).values
    
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(df, labels_for_skf)):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx}")
        print(f"{'='*60}")
        
        # Get training set and validation set
        train_fold = df.iloc[train_indices]
        val_fold = df.iloc[val_indices]
        
        print(f"Training set: {len(train_fold)}, Label distribution: {train_fold['label'].value_counts().to_dict()}")
        print(f"Validation set: {len(val_fold)}, Label distribution: {val_fold['label'].value_counts().to_dict()}")
        
        # Create fold CSV - format: train, train_label, val, val_label
        fold_df = pd.DataFrame()
        
        # Add training set columns
        fold_df['train'] = pd.Series(train_fold['slide_id'].values)
        fold_df['train_label'] = pd.Series(train_fold['label'].values)
        
        # Add validation set columns
        fold_df['val'] = pd.Series(val_fold['slide_id'].values)
        fold_df['val_label'] = pd.Series(val_fold['label'].values)
        
        # Save fold
        fold_csv_path = os.path.join(output_dir, f'fold{fold_idx}.csv')
        # When saving, Pandas will correctly format nullable integer 'Int64' as integers, null values will be written as empty strings
        fold_df.to_csv(fold_csv_path, index=False)
        print(f"Saved: {fold_csv_path}")
    
    print(f"\n{'='*60}")
    print("Data splitting completed!")
    print(f"Output directory: {output_dir}")
    print(f"Generated files: fold0.csv ~ fold{n_folds-1}.csv")
    print(f"{'='*60}")
    
    # Generate statistical summary
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n\n")
        f.write(f"Input file: {input_csv}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Label distribution: {df['label'].value_counts().to_dict()}\n\n")
        
        f.write(f"Number of cross-validation folds: {n_folds}\n")
        f.write(f"Validation set ratio per fold: {100/n_folds:.1f}%\n")
        f.write(f"Training set ratio per fold: {100*(n_folds-1)/n_folds:.1f}%\n")
        f.write(f"Random seed: {random_state}\n\n")
        
        # Iterate again to generate summary
        skf_summary = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for fold_idx, (train_indices, val_indices) in enumerate(skf_summary.split(df, labels_for_skf)):
            train_fold = df.iloc[train_indices]
            val_fold = df.iloc[val_indices]
            
            f.write(f"\nFold {fold_idx}:\n")
            f.write(f"  Training set: {len(train_fold)} samples\n")
            f.write(f"    Label distribution: {train_fold['label'].value_counts().to_dict()}\n")
            f.write(f"  Validation set: {len(val_fold)} samples\n")
            f.write(f"    Label distribution: {val_fold['label'].value_counts().to_dict()}\n")
    
    print(f"\nStatistical summary saved: {summary_path}")


if __name__ == '__main__':
    # Configuration parameters
    input_csv = '/data/home/scxj642/run/yujy/MS-RS_github/csv_data/CAMS_label.csv'
    output_dir = '/data/home/scxj642/run/yujy/MS-RS_github/csv_data/5fold'
    
    # Create stratified cross-validation splits
    create_stratified_cv_splits(
        input_csv=input_csv,
        output_dir=output_dir,
        n_folds=5,           # 5-fold cross-validation
        random_state=42      # Random seed
    )