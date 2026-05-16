import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import os
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


GLOBAL_GENERATOR = set_seed(42)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ---------------------------------------------------------------------------
# Per-scale config: parsing & validation
# ---------------------------------------------------------------------------
# Supported per-scale keys and their Python types.
_SCALE_KEY_TYPES = {
    'gcn_hidden':    int,
    'gcn_out':       int,
    'k_neighbors':   int,
    'gcn_layers':    int,
    'dropout':       float,
    'temperature':   float,
    'lr':            float,
}

def parse_scale_configs(args):
    """Parse --scale_configs JSON and return a typed dict or None.

    Each scale entry may override any subset of the keys listed in
    _SCALE_KEY_TYPES.  Unspecified keys fall back to the corresponding
    global arg value at model-construction time.

    Also accepts a 'clinical' key to set the LR for the clinical branch.

    Example JSON (single-quoted for shell):
        '{"5x":  {"gcn_hidden": 256, "gcn_out": 256, "dropout": 0.3,
                  "k_neighbors": 6,  "gcn_layers": 2, "lr": 5e-5},
          "10x": {"gcn_hidden": 384, "gcn_out": 384, "dropout": 0.4,
                  "k_neighbors": 8,  "gcn_layers": 2, "lr": 8e-5},
          "20x": {"gcn_hidden": 512, "gcn_out": 512, "dropout": 0.5,
                  "k_neighbors": 10, "gcn_layers": 3, "lr": 1e-4},
          "clinical": {"lr": 3e-5}}'

    Note: when different scales use different gcn_out values the
    fusion_method must be 'attention'.  'concatenation' requires all
    image-scale gcn_outs to be identical (enforced in the model __init__).
    """
    import json

    raw_str = getattr(args, 'scale_configs', None)
    if not raw_str:
        return None

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--scale_configs is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("--scale_configs must be a JSON object mapping scale names to dicts.")

    valid_scales = set(getattr(args, 'magnifications', [])) | {'clinical'}
    valid_keys   = set(_SCALE_KEY_TYPES.keys())

    parsed = {}
    for scale, cfg in raw.items():
        if scale not in valid_scales:
            raise ValueError(
                f"scale_configs: unknown scale '{scale}'. "
                f"Valid scales: {sorted(valid_scales)}"
            )
        if not isinstance(cfg, dict):
            raise ValueError(f"scale_configs['{scale}'] must be a dict.")
        unknown = set(cfg.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"scale_configs['{scale}'] has unknown keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_keys)}"
            )
        typed = {}
        for k, v in cfg.items():
            try:
                typed[k] = _SCALE_KEY_TYPES[k](v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"scale_configs['{scale}']['{k}']: cannot cast {v!r} "
                    f"to {_SCALE_KEY_TYPES[k].__name__}"
                ) from exc
        parsed[scale] = typed

    # Print summary
    print("\nPer-scale configs:")
    for scale in sorted(parsed.keys()):
        print(f"   {scale}: {parsed[scale]}")
    print()

    return parsed

def create_parser():
    parser = argparse.ArgumentParser(description='CHIEF-Style WSI Classification')

    parser.add_argument('--use_clinical', default=False, type=str2bool,
                       help='Whether to use clinical data')
    parser.add_argument('--clinical_csv',
                       default='/home/yujy/MS-RS/csv_data/CAMS_PCA_result.csv', type=str,
                       help='Clinical information CSV file path')
    parser.add_argument('--clinical_hidden', default=256, type=int,
                       help='Hidden dimension for clinical feature processing')
    parser.add_argument('--clinical_dropout', default=0.3, type=float,
                       help='Dropout rate for clinical feature branch')
    
    parser.add_argument('--fusion_method', default='concatenation', type=str,
                       choices=['concatenation', 'attention', 'weighted'],
                       help='Image and clinical feature fusion method: concatenation, attention, weighted')

    parser.add_argument('--use_contrastive', default=True, type=str2bool,
                       help='Whether to use WSI-level contrastive learning')
    parser.add_argument('--use_instance_branch', default=True, type=str2bool,
                       help='Whether to use Instance branch')
    parser.add_argument('--contrastive_weight', default=0.1, type=float,
                       help='Contrastive learning loss weight')
    parser.add_argument('--instance_weight', default=0.1, type=float,
                       help='Instance loss weight')
    parser.add_argument('--contrast_temperature', default=0.07, type=float,
                       help='Contrastive learning temperature parameter')
    parser.add_argument('--proj_dim', default=256, type=int,
                       help='Contrastive learning projection dimension')
    parser.add_argument('--memory_bank_size', default=128, type=int,
                       help='Memory Bank size (per subtype per class)')
    # [FIX 4b] MemoryBank 激活后对比损失渐进生效的 epoch 数
    parser.add_argument('--contrastive_ramp_epochs', default=10, type=int,
                       help='Epochs to linearly ramp up contrastive loss after MemoryBank warmup')

    parser.add_argument('--pretrained_path', default=None, type=str,
                       help='Self-supervised pretrained weights path')

    parser.add_argument('--use_pseudo_bag_aug', default=True, type=str2bool,
                       help='Whether to use pseudo bag augmentation')
    parser.add_argument('--pseudo_bag_ratio', default=6, type=float,
                       help='Pseudo bag expansion target ratio')
    parser.add_argument('--pseudo_bag_sample_ratio', default=0.7, type=float,
                       help='Patch sampling ratio within bag')
    parser.add_argument('--pseudo_bag_mix_prob', default=0.3, type=float,
                       help='Probability of mixing between bags')
    parser.add_argument('--pseudo_bag_noise_std', default=0.02, type=float,
                       help='Feature noise standard deviation')
    parser.add_argument('--pseudo_bag_min_patches', default=2, type=int,
                       help='Minimum number of patches in pseudo bag')

    parser.add_argument('--name', default='CHIEF_WSI', type=str)
    parser.add_argument('--EPOCH', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_cls', default=2, type=int)
    parser.add_argument('--num_subtypes', default=4, type=int,
                       help='Number of subtypes (corresponds to cancer type in CHIEF)')

    parser.add_argument('--in_dim', default=768, type=int,
                       help='Input feature dimension (768 for CHIEF CTransPath)')
    parser.add_argument('--gcn_hidden', default=512, type=int)
    parser.add_argument('--gcn_out', default=512, type=int)


    parser.add_argument('--subtype_embed_dim', default=256, type=int)
    parser.add_argument('--use_subtype_in_classifier', default=True, type=str2bool)
    parser.add_argument('--k_neighbors', default=8, type=int)
    parser.add_argument('--gcn_layers', default=2, type=int)

    parser.add_argument('--magnifications', default=['5x', '10x', '20x'], nargs='+', type=str)
    parser.add_argument('--use_multiscale', default=True, type=str2bool)

    parser.add_argument('--ensemble_method', default='attention', type=str,
                       choices=['weighted', 'random_forest', 'mlp', 'voting', 'attention'],
                       help='Ensemble method: attention=dynamic attention ensemble (recommended)')
    parser.add_argument('--scale_weights', default=[0.3, 0.3, 0.4], nargs='+', type=float)
    parser.add_argument('--ensemble_hidden_dim', default=128, type=int)
    parser.add_argument('--weight_lr_multiplier', default=1.5, type=float)
    # [FIX 4c] ensemble fuser 独立 LR 倍率（与 weight_lr_multiplier 解耦）
    # 建议不超过 2.0；过高会导致 ensemble 权重在分支参数稳定前剧烈震荡
    parser.add_argument('--ensemble_lr_multiplier', default=1.0, type=float,
                       help='LR multiplier for DynamicAttentionFuser only')
    parser.add_argument('--weight_regularization', default=0.01, type=float)
    parser.add_argument('--adaptive_weight_init', default=True, type=str2bool)

    parser.add_argument('--attention_type', default='chief', type=str,
                       choices=['chief', 'cross', 'traditional', 'simple'],
                       help='Attention module type')
    parser.add_argument('--attention_temperature', default=1.5, type=float)

    parser.add_argument('--use_focal_loss', default=True, type=str2bool)
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--use_weighted_loss', default=False, type=str2bool)
    parser.add_argument('--use_weighted_sampling', default=False, type=str2bool)
    parser.add_argument('--weight_strategy', default='temperature', type=str,
                       choices=['inverse', 'effective', 'balanced', 'balanced_smooth', 'temperature'])
    parser.add_argument('--effective_beta', default=0.9999, type=float)
    parser.add_argument('--sampling_temperature', default=1.0, type=float)
    parser.add_argument('--sampling_smooth_factor', default=0.7, type=float)
    parser.add_argument('--dynamic_sampling_warmup', default=10, type=int)

    parser.add_argument('--use_undersampling', default=False, type=str2bool)
    parser.add_argument('--undersample_ratio', default=3.0, type=float)
    parser.add_argument('--minority_aug_ratio', default=1.5, type=float)
    parser.add_argument('--use_dynamic_threshold', default=True, type=str2bool)
    parser.add_argument('--threshold_search_metric', default='f1', type=str)

    parser.add_argument('--use_lr_scheduler', default=True, type=str2bool)
    parser.add_argument('--scheduler_type', default='cosine_warmup', type=str)
    parser.add_argument('--warmup_epochs', default=8, type=int)
    parser.add_argument('--warmup_lr', default=1e-4, type=float)

    parser.add_argument('--use_label_smoothing', default=True, type=str2bool)
    parser.add_argument('--label_smoothing', default=0.05, type=float)

    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)

    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--min_epochs', default=40, type=int)
    parser.add_argument('--early_stop_metric', default='val_auc', type=str,
                       choices=['val_loss', 'val_auc', 'val_f1'])

    parser.add_argument('--dropout', default=0.4, type=float)

    parser.add_argument('--features_dir',
                       default='/252_node_user_storage/yujy/CAMS_data/features_768', type=str,
                       help='CHIEF 768-dimensional features directory')
    parser.add_argument('--csv_dir',
                       default='/home/yujy/PC-TMD/splits/CAMS/5fold_580_stratified', type=str)
    parser.add_argument('--log_dir',
                       default='/home/yujy/PC-TMD/results/results_CAMS', type=str)

    parser.add_argument('--test_mode', default=False, type=str2bool)
    parser.add_argument('--test_csv',
                       default='/home/yujy/PC-TMD/splits/CAMS/5fold_580_stratified/test.csv',
                       type=str)
    parser.add_argument('--load_checkpoint', default=None, type=str)
    parser.add_argument('--skip_training', default=False, type=str2bool)

    parser.add_argument('--n_folds', default=5, type=int)
    parser.add_argument('--run_fold', default=-1, type=int)
    parser.add_argument('--n_workers', default=6, type=int)

    parser.add_argument('--use_cache', default=False, type=str2bool)
    parser.add_argument('--cache_dir',
                       default='/data/home/scxj642/run/yujy/cache', type=str)
    parser.add_argument('--force_reload', default=False, type=str2bool)

    parser.add_argument('--seed', default=42, type=int, help='Base random seed')

    # Per-scale configuration (JSON string).  See parse_scale_configs() for
    # the full spec.  Unspecified keys fall back to the global args above.
    parser.add_argument(
        '--scale_configs', default=None, type=str,
        help=(
            'Per-scale JSON config. Supported keys per scale: '
            'gcn_hidden, gcn_out, k_neighbors, gcn_layers, dropout, temperature, lr. '
            'Also accepts "clinical" key (only lr is meaningful there). '
            'Example (single-quoted for shell): '
            '\'{"5x":{"gcn_hidden":256,"gcn_out":256,"dropout":0.3,"lr":5e-5},'
            '"20x":{"gcn_hidden":512,"gcn_out":512,"dropout":0.5,"lr":1e-4},'
            '"clinical":{"lr":3e-5}}\''
        )
    )

    return parser