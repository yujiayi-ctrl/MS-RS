import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def create_stratified_cv_splits_by_subtype(
    input_csv, 
    clinical_csv,
    output_dir, 
    n_folds=5, 
    random_state=42
):
    """
    Create K-fold cross-validation data splitting based on subtype + label stratified sampling
    
    Args:
        input_csv: Path to input CSV file (containing slide_id and label)
        clinical_csv: Path to clinical information CSV file (containing slide_id and subtype)
        output_dir: Output directory
        n_folds: Number of cross-validation folds
        random_state: Random seed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load main data
    print(f"Loading main data: {input_csv}")
    df = pd.read_csv(input_csv, dtype={'label': 'Int64'})
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Read clinical information (subtype)
    print(f"\nLoading clinical information: {clinical_csv}")
    clinical_df = pd.read_csv(clinical_csv)
    print(f"Number of clinical samples: {len(clinical_df)}")
    print(f"Subtype distribution:\n{clinical_df['subtype'].value_counts()}")
    
    # Merge data: Match subtype information by slide_id
    df = df.merge(
        clinical_df[['slide_id', 'subtype']], 
        on='slide_id', 
        how='left'
    )
    
    # Check merge results
    missing_subtype = df['subtype'].isna().sum()
    if missing_subtype > 0:
        print(f"\n⚠️ Warning: {missing_subtype} samples are missing subtype information, will be marked as 'unknown'")
        df['subtype'] = df['subtype'].fillna('unknown')
    
    print(f"\nMerged data:")
    print(f"  Total samples: {len(df)}")
    print(f"  Subtype distribution: {df['subtype'].value_counts().to_dict()}")
    
    df['stratify_key'] = df['subtype'].astype(str) + '_label' + df['label'].astype(str)
    
    print(f"\nStratify key distribution:")
    stratify_counts = df['stratify_key'].value_counts().sort_index()
    for key, count in stratify_counts.items():
        print(f"  {key}: {count}")
    
    min_samples_per_key = stratify_counts.min()
    if min_samples_per_key < n_folds:
        print(f"\n⚠️ Warning: Some stratify keys have fewer samples ({min_samples_per_key}) than the number of folds ({n_folds})")
        print("   Will merge rare subtypes to ensure stratification is valid...")
        
        rare_keys = stratify_counts[stratify_counts < n_folds].index.tolist()
        df.loc[df['stratify_key'].isin(rare_keys), 'stratify_key'] = \
            'rare_label' + df.loc[df['stratify_key'].isin(rare_keys), 'label'].astype(str)
        
        print(f"\nAdjusted stratify key distribution:")
        for key, count in df['stratify_key'].value_counts().sort_index().items():
            print(f"  {key}: {count}")
    
    # Perform K-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    stratify_labels = df['stratify_key'].values
    
    all_fold_stats = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(df, stratify_labels)):
        print(f"\n{'='*60}")
        print(f"处理 Fold {fold_idx}")
        print(f"{'='*60}")
        
        # Get training set and validation set
        train_fold = df.iloc[train_indices]
        val_fold = df.iloc[val_indices]
        
        # Print statistics
        print(f"Training set: {len(train_fold)}")
        print(f"  Label distribution: {train_fold['label'].value_counts().to_dict()}")
        print(f"  Subtype distribution: {train_fold['subtype'].value_counts().to_dict()}")
        
        print(f"Validation set: {len(val_fold)}")
        print(f"  Label distribution: {val_fold['label'].value_counts().to_dict()}")
        print(f"  Subtype distribution: {val_fold['subtype'].value_counts().to_dict()}")
        
        # Print detailed distribution in validation set by subtype
        print(f"  Validation set detailed distribution:")
        for subtype in sorted(val_fold['subtype'].unique()):
            subtype_data = val_fold[val_fold['subtype'] == subtype]
            pos = (subtype_data['label'] == 1).sum()
            neg = (subtype_data['label'] == 0).sum()
            print(f"    {subtype}: Positive samples={pos}, Negative samples={neg}")
        
        # Save fold statistics for summary
        all_fold_stats.append({
            'fold': fold_idx,
            'train_size': len(train_fold),
            'val_size': len(val_fold),
            'val_pos': (val_fold['label'] == 1).sum(),
            'val_neg': (val_fold['label'] == 0).sum()
        })
        
        fold_df = pd.DataFrame()
        
        # Add training set columns
        fold_df['train'] = pd.Series(train_fold['slide_id'].values)
        fold_df['train_label'] = pd.Series(train_fold['label'].values)
        
        # Add validation set columns
        fold_df['val'] = pd.Series(val_fold['slide_id'].values)
        fold_df['val_label'] = pd.Series(val_fold['label'].values)
        
        # Save fold
        fold_csv_path = os.path.join(output_dir, f'fold{fold_idx}.csv')
        fold_df.to_csv(fold_csv_path, index=False)
        print(f"Adjusted fold CSV saved: {fold_csv_path}")
    
    # Verifying no overlap between validation sets of different folds
    print(f"\n{'='*60}")
    print("Verifying no overlap between validation sets of different folds:")
    print(f"{'='*60}")
    
    all_val_sets = {}
    for fold_idx in range(n_folds):
        fold_df = pd.read_csv(os.path.join(output_dir, f'fold{fold_idx}.csv'))
        all_val_sets[fold_idx] = set(fold_df['val'].dropna().tolist())
    
    overlap_found = False
    for i in range(n_folds):
        for j in range(i+1, n_folds):
            overlap = all_val_sets[i] & all_val_sets[j]
            if len(overlap) > 0:
                print(f"  ✗ Fold {i} ∩ Fold {j}: {len(overlap)} overlapping samples!")
                overlap_found = True
            else:
                print(f"  ✓ Fold {i} ∩ Fold {j}: No overlap")
    
    if not overlap_found:
        print("\n✓ All validation sets have no overlap, the split is correct!")
    
    # Generate statistical summary
    print(f"\n{'='*60}")
    print("Data splitting completed!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Generated files: fold0.csv ~ fold{n_folds-1}.csv")
    
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Stratified K-fold cross-validation split based on subtypes + labels\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Input file: {input_csv}\n")
        f.write(f"Clinical information: {clinical_csv}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Label distribution: {df['label'].value_counts().to_dict()}\n")
        f.write(f"Subtype distribution: {df['subtype'].value_counts().to_dict()}\n\n")
        
        f.write(f"Cross-validation folds: {n_folds}\n")
        f.write(f"Validation set比例: {100/n_folds:.1f}%\n")
        f.write(f"Training set ratio per fold: {100*(n_folds-1)/n_folds:.1f}%\n")
        f.write(f"Random seed: {random_state}\n\n")

        f.write("Stratification key distribution:\n")
        for key, count in df['stratify_key'].value_counts().sort_index().items():
            f.write(f"  {key}: {count}\n")
        f.write("\n")
        
        for fold_idx in range(n_folds):
            fold_df = pd.read_csv(os.path.join(output_dir, f'fold{fold_idx}.csv'))
            train_labels = fold_df['train_label'].dropna()
            val_labels = fold_df['val_label'].dropna()
            
            f.write(f"\nFold {fold_idx}:\n")
            f.write(f"  Training set: {len(train_labels)} samples\n")
            f.write(f"    Positive: {(train_labels == 1).sum()}\n")
            f.write(f"    Negative: {(train_labels == 0).sum()}\n")
            f.write(f"  Validation set: {len(val_labels)} samples\n")
            f.write(f"    Positive: {(val_labels == 1).sum()}\n")
            f.write(f"    Negative: {(val_labels == 0).sum()}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Validation set overlap check: ")
        f.write("Passed ✓\n" if not overlap_found else "Overlap found ✗\n")
        f.write("="*60 + "\n")
    
    print(f"\nStatistical summary saved: {summary_path}")
    
    return all_fold_stats


if __name__ == '__main__':

    input_csv = '/data/home/scxj642/run/yujy/MS-RS_github/csv_data/CAMS_label.csv'
    clinical_csv = '/data/home/scxj642/run/yujy/MS-RS_github/csv_data/CAMS_clinical_processed.csv'
    output_dir = '/data/home/scxj642/run/yujy/MS-RS_github/csv_data/5fold'
    
    # Create subtype-based stratified cross-validation split
    stats = create_stratified_cv_splits_by_subtype(
        input_csv=input_csv,
        clinical_csv=clinical_csv,
        output_dir=output_dir,
        n_folds=5,
        random_state=42
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Fold Validation Set Statistics Summary:")
    print("="*60)
    print(f"{'Fold':<6} {'Training Set':<10} {'Validation Set':<10} {'Positive':<10} {'Negative':<10}")
    print("-"*46)
    for s in stats:
        print(f"{s['fold']:<6} {s['train_size']:<10} {s['val_size']:<10} {s['val_pos']:<10} {s['val_neg']:<10}")