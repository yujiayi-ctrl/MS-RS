```markdown
# Breast Cancer Pathology Image Feature Processing and Cross-Validation Pipeline

This directory contains scripts and documentation for breast cancer pathology image feature processing, patch extraction, CHIEF feature extraction, PCA dimensionality reduction, and generating k-fold cross-validation datasets.

## Overall Pipeline Overview

1.  **Data Preprocessing and Merging (Manual or Scripted)**
    *   Merge clinical information with cell nucleus features extracted from pathology slides.
    *   Handle missing values, encode categorical variables, etc.
    *   Output a complete data file for PCA analysis.

2.  **Patch Extraction (`patch_extraction/main.py`)**
    *   Extract patches from pathology slides based on multiple magnifications (5x, 10x, 20x).
    *   Supports XML annotation region extraction, with automatic fallback to traditional tissue segmentation.
    *   Optional stain normalization to unify color distribution.
    *   Output patch folder structure: `<save_folder>/<slide_id>/<magnification>/patches`.

3.  **CHIEF Feature Extraction (`features_extraction/Get_CHIEF_patch_feature.py`)**
    *   Extract 768-dimensional features from patches using a pre-trained CTransPath model.
    *   Supports multiple magnifications (5x, 10x, 20x).
    *   Outputs feature files (`.pkl` or `.pt`) for each sample at each magnification.

4.  **Feature Dimensionality Reduction (`csv_data/pca.py`)**
    *   Reads the merged complete data.
    *   Applies Principal Component Analysis (PCA) to highly correlated features to reduce dimensionality.
    *   Outputs the reduced data file (`csv_data/all_pca.csv`) and related analysis plots.

5.  **K-Fold Cross-Validation Data Splitting (`csv_data/split_subtype.py` or `csv_data/split.py`)**
    *   Reads the reduced data file (`csv_data/all_pca.csv`).
    *   Splits the dataset into K mutually exclusive training/validation subsets based on a specified stratification strategy (e.g., by subtype and label).
    *   Outputs K `fold_{i}.csv` files for subsequent model training and evaluation.

---
## Patch Extraction and Stain Normalization
**Script:** `patch_extraction/main.py`

This script performs multi-magnification patch extraction from Whole Slide Images (WSIs):

1.  **XML Annotation Handling**
    *   Extracts patches from tumor regions using XML annotations if available.
    *   Automatically falls back to traditional tissue segmentation if XML files are missing.
2.  **Stain Normalization (Optional)**
    *   Unifies patch staining distribution using the Reinhard method.
    *   Supports specifying a reference image (`stain_norm_reference_image`).
3.  **Multi-Magnification Patch Extraction**
    *   Supports 5x, 10x, 20x magnifications.
    *   Patch size and stride can be configured per magnification (refer to `config.py`).
4.  **Output Directory Structure**
    ```
    save_folder_dir/
    ├── <slide_id_1>/
    │   ├── 5x/
    │   │   ├── patch_0_0.jpg
    │   │   └── patch_0_1.jpg
    │   ├── 10x/
    │   └── 20x/
    ├── <slide_id_2>/
    │   └── ...
    ```
5.  **Logging Statistics**
    *   Number of patches per slide.
    *   Whether XML annotations were used.
    *   Whether stain normalization was applied.
    *   Saved to `processing_log_xml_annotation_stain_norm.csv`.

**Configuration Example (`config.py`):**
```python
num_thread = 4
magnifications = {'5x': {'patch_size': 256, 'stride': 256}, '10x': {'patch_size': 256, 'stride': 256}, '20x': {'patch_size': 256, 'stride': 256}}
tissue_mask_threshold = 0.5
enable_stain_normalization = True
stain_norm_reference_image = '/path/to/reference_image.jpg'
dataset_csv = '/path/to/CAMS_clinical.csv'
save_folder_dir = '/path/to/output/patches'
use_xml_annotations = True
jpeg_quality = 95
```

---

## K-Fold Cross-Validation Data Splitting

This step divides the final dataset into K folds for cross-validation. Using a strategy stratified by subtype and label (`csv_data/split_subtype.py`) is recommended to ensure consistent class proportions across folds, especially for imbalanced datasets.

### Script: `csv_data/split_subtype.py`

This script reads CSV files containing `slide_id`, `label`, and `subtype` information and creates a K-fold cross-validation split. It ensures that the proportion of positive/negative samples for different subtypes in the validation set of each fold matches the overall distribution.

#### Input File Format (`csv_data/split_subtype.py`)

The script requires **two** input files.

**1. Main Data File (e.g., `csv_data/CAMS_label.csv`)**

Contains sample IDs and labels.

| slide_id        | label |
| :-------------- | :---- |
| 586543-5        | 0     |
| 702520-3        | 0     |
| 655040-3        | 1     |
| ...             | ...   |

**2. Clinical Information File (e.g., `csv_data/All_data_processed.csv`)**

Contains sample IDs and subtype information used for stratification (must include at least `slide_id` and `subtype` columns).

| slide_id        | subtype    |
| :-------------- | :--------- |
| 358439-6        | HR-HER2+   |
| 361847-3        | HR+HER2+   |
| 373486-4        | HR-HER2+   |
| 395452-6        | HR+HER2+   |
| ...             | ...        |

*Note: The script merges the two files based on `slide_id`.*

#### Output File Format (`csv_data/split_subtype.py`)

The script generates `fold0.csv` through `fold4.csv` (for 5 folds) in the specified output directory, along with a summary file `split_summary.txt`.

**Example: `fold0.csv`**

| train           | train_label | val             | val_label |
| :-------------- | :---------- | :-------------- | :-------- |
| 586543-5        | 0           | 534124-3        | 0         |
| 702520-3        | 0           | 531459-3        | 0         |
| 558695-4        | 0           | 560530-2        | 1         |
| ...             | ...         | ...             | ...       |

*   `train`: `slide_id` of samples in the training set.
*   `train_label`: Label of the training samples (0/1).
*   `val`: `slide_id` of samples in the validation set.
*   `val_label`: Label of the validation samples (0/1).

---

## CHIEF Feature Extraction

**Script:** `features_extraction/Get_CHIEF_patch_feature.py`

This script extracts CHIEF features (768-dimensional vectors) from patch images generated from WSIs, supporting multiple magnifications.

**Input Arguments**

| Argument           | Type   | Default                                                                       | Description                                                     |
| :----------------- | :----- | :---------------------------------------------------------------------------- | :-------------------------------------------------------------- |
| `--base_data_dir`  | str    | `/252_node_user_storage/yujy/CAMS_data/patches`                               | Root directory for patch images, subdirs: `slide_id/magnification` |
| `--log_dir`        | str    | `/252_node_user_storage/yujy/CAMS_data/features_768`                          | Output directory for features                                    |
| `--model_weight`   | str    | `/data/home/scxj642/run/yujy/MS-RS/features_extraction/CHIEF_CTransPath.pth` | Path to pre-trained CTransPath model weights                   |
| `--dataset_csv`    | str    | `/home/yujy/CAMS_data/CAMS_clinical_processed585.csv`                         | CSV file listing samples                                         |
| `--magnifications` | list   | `['5x','10x','20x']`                                                          | Supported magnifications                                         |
| `--batch_size`     | int    | 256                                                                           | Batch size for DataLoader                                        |
| `--num_workers`    | int    | 8                                                                             | Number of worker threads for DataLoader                          |
| `--gpu_ids`        | str    | '0'                                                                           | GPU IDs to use                                                  |
| `--use_amp`        | bool   | True                                                                          | Enable Automatic Mixed Precision                                 |
| `--skip_existing`  | bool   | True                                                                          | Skip processing if output file already exists                    |
| `--save_format`    | str    | 'pt'                                                                          | Format to save features, either 'pt' or 'pkl'                    |

**Output Files**

*   One `.pkl` or `.pt` file per slide, containing patch features for different magnifications.
*   Example file paths:
    ```
    /features_768/5x/358439-6.pt
    /features_768/10x/358439-6.pt
    /features_768/20x/358439-6.pt
    ```

**Core Workflow**
1.  Load patch images using `PatchDataset`.
2.  Extract features using the pre-trained `SwinTransformerForCTransPath` model.
3.  Save features as `.pt` or `.pkl` files.
4.  Supports skipping existing feature files to speed up batch processing.

---

## Feature Dimensionality Reduction (PCA)

This step aims to reduce feature redundancy and dimensionality while preserving most of the variance.

### Script: `csv_data/pca.py`

This script performs the following core operations:
1.  **Data Standardization**: Scales all features to zero mean and unit variance using `StandardScaler`.
2.  **Correlation Analysis**: Computes the correlation matrix between features. Identifies feature pairs with an absolute correlation coefficient greater than a threshold (default 0.8).
3.  **PCA Dimensionality Reduction**: Applies PCA only to the highly correlated features. Automatically selects the number of principal components needed to explain 95% of the variance.
4.  **Feature Merging**: Combines the new PCA-generated features (`PCA_1`, `PCA_2`, ...) with the original low-correlation features to form the final feature set.
5.  **Result Saving**: Saves the final data and generates visualization files like correlation heatmaps and PCA variance explained plots.

#### Input File Format (`csv_data/pca.py`)

The script requires a CSV file containing merged clinical information and all features. **It assumes that the merging and cleaning of clinical information and pathology features have been completed.**

**Example: `merged_all_samples_with_dfs_inner.csv`**

| slide_id     | slide_path                  | subtype   | ... | eccentricity | orientation | T_stage_T3 | ... | label |
| :----------- | :-------------------------- | :-------- | :-: | :----------- | :---------- | :--------- | :-: | :---- |
| 358439-6     | /252_node.../358439-6.ndpi  | HR-HER2+  | ... | 0.703505577  | -0.7331214  | -0.2963932 | ... | 1     |
| ...          | ...                         | ...       | ... | ...          | ...         | ...        | ... | ...   |

*   `slide_id`: Unique identifier for the sample.
*   `slide_path`: File path to the pathology slide.
*   `subtype`: Breast cancer subtype, used for subsequent stratified sampling.
*   `label`: **(Optional, but highly recommended)** Classification label for the sample (e.g., 0=non-recurrence, 1=recurrence). This column is not processed by PCA but is retained in the output file for downstream tasks.
*   Other columns: All feature columns to be considered for PCA, which can be continuous values (e.g., `eccentricity`) or one-hot encoded categorical variables (e.g., `T_stage_T3`).

#### Output File Format (`csv_data/pca.py`)

The most important output file is the PCA-processed data.

**Example: `csv_data/all_pca.csv`**

| slide_id     | slide_path                  | subtype   | area_filled | PCA_1      | PCA_2      | ... | eccentricity | orientation | T_stage_T3 | label |
| :----------- | :-------------------------- | :-------- | :---------- | :--------- | :--------- | :-: | :----------- | :---------- | :--------- | :---- |
| 358439-6     | /252_node.../358439-6.ndpi  | HR-HER2+  | 37.13433952 | -0.2090112 | -3.7512428 | ... | 0.703505577  | -0.7331214  | -0.2963932 | 1     |
| 361847-3     | /252_node.../361847-3.ndpi  | HR+HER2+  | 43.03241621 | 2.7067263  | 1.8609939  | ... | 0.192777885  | -0.6741015  | -0.2963932 | 1     |
| TCGA-3C-AALI | /home/.../TCGA-3C-AALI.svs  | HR+HER2+  | 34.46272156 | -2.7691269 | 1.3640637  | ... | -0.419074822 | -0.1572768  | -0.2963932 | 0     |
| ...          | ...                         | ...       | ...         | ...        | ...        | ... | ...          | ...         | ...        | ...   |

*   Non-feature columns like `slide_id`, `slide_path`, `subtype`, `label` are preserved intact.
*   `PCA_1`, `PCA_2`, ...: New features generated from the highly correlated feature group via PCA.
*   Other feature columns: Original features (e.g., `eccentricity`, `orientation`, `T_stage_T3`) that were not part of the highly correlated group are retained as they were.

This `csv_data/all_pca.csv` file is the direct input for subsequent machine learning modeling, including k-fold cross-validation.

---

## Training and Testing Pipeline

This section details how to use the `train/MAIN.py` script for training, validation, and testing the multi-scale pathology image classification model based on Graph Neural Networks and contrastive learning. The script integrates the complete workflow of data loading, model construction, training loops, validation evaluation, and final testing.

### Core Script: `train/MAIN.py`

`train/MAIN.py` is the main entry point for the training and testing pipeline. It controls all aspects of the experiment by parsing command-line arguments defined in `train/config.py`.

#### 1. Training Mode (K-Fold Cross-Validation)

This is the default mode, used to perform K-fold cross-validation on a given dataset to evaluate model stability and performance.

**Example Command:**
```bash
# Recommended to submit via SLURM script (e.g., run.sh), especially for long-running tasks
python MAIN.py \
    --name                        "my_experiment" \
    --device                      cuda \
    --EPOCH                       200 \
    --magnifications              5x 10x 20x \
    --batch_size                  4 \
    --patience                    15 \
    --min_epochs                  80 \
    --lr                          0.00065 \
    --warmup_epochs               20 \
    --warmup_lr                   0.000325 \
    --min_lr                      0.0001 \
    --dropout                     0.5 \
    --weight_decay                0.00035 \
    --use_contrastive             True \
    --use_instance_branch         True \
    --contrastive_weight          0.15 \
    --instance_weight             0.1 \
    --contrast_temperature        0.7 \
    --proj_dim                    256 \
    --memory_bank_size            128 \
    --contrastive_ramp_epochs     10 \
    --in_dim                      768 \
    --gcn_hidden                  512 \
    --use_clinical                True \
    --clinical_csv                /path/to/all_pca.csv \
    --csv_dir                     /path/to/5fold_splits \
    --features_dir                /path/to/features_768 \
    --log_dir                     /path/to/results \
    --n_folds                     5
    ... (other arguments)
```

**Key Arguments (Training Related):**
*   `--n_folds`: (int) Total number of folds for cross-validation.
*   `--run_fold`: (int) Specifies which fold to run (-1 runs all folds, default: -1).
*   `--test_mode`: (bool) Must be `False` (default) to start training mode.
*   `--csv_dir`: (str) Directory containing `fold0.csv`, `fold1.csv`, etc., from the k-fold split.
*   `--features_dir`: (str) Root directory where CHIEF feature files are stored.
*   `--log_dir`: (str) Root directory for saving model weights, TensorBoard logs, and evaluation results.
*   `--use_cache`: (bool) Whether to cache processed data to speed up subsequent runs.

**Detailed Training Workflow:**
1.  **Argument Parsing & Initialization**: Parse command-line arguments, set random seeds, create logging directories.
2.  **Data Loading & Splitting**: Iterate over each fold (`fold_idx` from 0 to `n_folds-1`):
    *   Load sample IDs and labels for the training and validation sets from `csv_dir/fold{fold_idx}.csv`.
    *   Call `load_and_organize_data_with_cache` function from `data_loader.py`:
        *   Use multi-threading to load multi-scale feature files (`.pt` or `.pkl`) from `features_dir` for the given `slide_id`s.
        *   Generate spatial coordinates from features and construct KNN graphs (`edge_index`).
        *   Load and preprocess clinical features from `clinical_csv`.
        *   Return organized features, graph structures, labels, and subtype information.
    *   (Optional) Apply data augmentation strategies (e.g., pseudo-bag augmentation `use_pseudo_bag_aug`) to balance training data.
3.  **Model Initialization**:
    *   Instantiate the `CHIEF_MultiScale_Model`. This model includes:
        *   Multiple `CHIEFScaleBranch` modules, each processing graph data from a different magnification (e.g., 5x, 10x, 20x).
        *   An optional `ClinicalBranch` for processing clinical features.
        *   A `DynamicAttentionFuser` for dynamically fusing predictions from multi-scale and clinical branches.
        *   A `MemoryBank` module for WSI-level contrastive learning.
    *   (Optional) Load pre-trained weights from `pretrained_path`.
4.  **Training Preparation**:
    *   Instantiate the `CHIEFLoss` criterion, supporting class balancing, Focal Loss, and label smoothing.
    *   Create an optimizer (`create_optimized_optimizer`) that supports independent learning rates for different model branches.
    *   Initialize the learning rate scheduler (`OptimizedWarmupCosineScheduler`).
    *   Initialize the `EarlyStopping` mechanism.
5.  **Training & Validation Loop**:
    *   For each epoch, call `train_one_epoch_chief` to perform training. This function:
        *   Performs forward pass to compute classification, instance branch, and contrastive learning losses.
        *   Executes gradient accumulation and backpropagation.
        *   Updates the `MemoryBank`.
    *   After each epoch, call `validate_chief` to evaluate model performance on the validation set.
    *   Save the best model based on the validation `AUC` metric.
    *   Terminate training early if the early stopping condition is met.
6.  **Result Logging**:
    *   After training each fold, save the best model (`best_model.pth`) and detailed validation predictions to `log_dir/experiment_name/fold_{fold_idx}/`.
    *   After all folds are complete, calculate and save average performance metrics to `summary_results.csv`.

#### 2. Test Mode (Final Training & Test)

In this mode, the model combines all training/validation data, performs final training, and evaluates on a held-out independent test set.

**Example Command:**
```bash
python MAIN.py \
    --test_mode True \
    --skip_training False \
    --test_csv /path/to/test.csv \
    ... (other arguments similar to training mode)
```

**Key Arguments (Testing Related):**
*   `--test_mode`: (bool) Must be set to `True`.
*   `--skip_training`: (bool) If `True`, skips final training on all data and only loads an existing checkpoint via `load_checkpoint` for testing.
*   `--load_checkpoint`: (str) Path to a pre-trained model checkpoint for testing. Required if `skip_training` is `True`.
*   `--test_csv`: (str) Path to the CSV file for the independent test set.

**Detailed Testing Workflow (`train_final_and_test` in `test_functions.py`):**
1.  **Merge Training/Validation Data**:
    *   Load and combine data from all folds' training and validation sets into a single complete training set.
2.  **Load Test Data**:
    *   Load the independent test set data based on `test_csv`.
3.  **Final Model Training** (if `skip_training` is `False`):
    *   Train a model from scratch using the merged complete training set. The training process is similar to a single fold but without cross-validation.
    *   The best model during this final training phase is saved as `final_best_model.pth`.
4.  **Model Evaluation**:
    *   Load the best final trained model (`final_best_model.pth` or `load_checkpoint`).
    *   Run the `test_model` function on the test set to compute final performance metrics (AUC, F1-score, Accuracy, etc.).
    *   Generate and save confusion matrix and ROC curve plots.

#### 3. Key Auxiliary Files Description

*   **`run.sh`**: A job submission script for High-Performance Computing clusters (like SLURM). It contains preset training parameters, environment setup, error retry logic, and temporary file cleanup. It is the recommended way to execute large-scale experiments.
*   **`train/config.py`**: The project's configuration hub, defining all available command-line arguments and their defaults. Modifying this file changes the default behavior of experiments.
*   **`data_loader.py`**: Handles data reading, preprocessing, and loading. It manages multi-scale features, graph structure generation, and clinical feature integration.
*   **`models.py`**: Defines the core model architecture, including Graph Convolutional Networks, attention mechanisms, contrastive learning modules, and the dynamic ensemble module.
*   **`trainers.py`**: Contains the concrete implementations of the training and validation loops, as well as the definitions of training components like optimizers and learning rate schedulers.
*   **`test_functions.py`**: Specifically handles the data preparation, model training, and evaluation workflow for the final test mode.
*   **`utils.py`**: Provides a suite of utility functions for calculating evaluation metrics and creating visualizations, such as plotting ROC curves and confusion matrices.

By understanding the above workflow, you can flexibly use the `train/MAIN.py` script for model development, validation, and final testing. It is recommended to start with a smaller experiment (e.g., `--n_folds 1 --EPOCH 10`) to ensure the entire pipeline runs correctly.
```