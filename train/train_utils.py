import torch
import os


def load_pretrained_weights(model, pretrained_path, device='cuda', strict=False):
    if not os.path.exists(pretrained_path):
        print(f"⚠️ Pretrained weights file does not exist: {pretrained_path}")
        return model, [], []
    
    print(f"\n{'='*60}")
    print(f"📥 Loading pretrained weights: {pretrained_path}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['model_state_dict']
    else:
        pretrained_state_dict = checkpoint
    
    model_state_dict = model.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    
    filtered_state_dict = {}
    for key, value in pretrained_state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
        else:
            skipped_keys.append(f"{key} (not in model)")
    
    model.load_state_dict(filtered_state_dict, strict=False)
    
    missing_keys = [k for k in model_state_dict.keys() if k not in pretrained_state_dict]
    
    print(f"\n📊 Weight Loading Statistics:")
    print(f"   ✓ Successfully loaded: {len(loaded_keys)} parameters")
    print(f"   ⚠️ Skipped: {len(skipped_keys)} parameters")
    print(f"   ❓ Missing: {len(missing_keys)} parameters")
    
    print(f"\n📋 Loaded Parameter Categories:")
    
    module_counts = {}
    for key in loaded_keys:
        module = key.split('.')[0]
        if module not in module_counts:
            module_counts[module] = 0
        module_counts[module] += 1
    
    for module, count in sorted(module_counts.items()):
        print(f"   {module}: {count} parameters")
    
    mb_loaded = sum(1 for k in loaded_keys if 'memory_bank' in k.lower() or 'bank_' in k)
    print(f"\n   Memory Bank related parameters: {mb_loaded}")
    
    if 'epoch' in checkpoint:
        print(f"\n📅 Pretrained epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"📉 Pretrained loss: {checkpoint['loss']:.4f}")
    
    print(f"{'='*60}\n")
    
    return model, loaded_keys, missing_keys


def verify_memory_bank_loaded(model):
    print("\n🔍 Verifying Memory Bank Status:")
    
    stats = {}
    for mag in model.magnifications:
        if mag in model.memory_banks:
            mb = model.memory_banks[mag]
            mag_stats = {}
            
            total_filled = 0
            for s in range(mb.num_subtypes):
                for label in [0, 1]:
                    bank_key = f'bank_{s}_{label}'
                    bank = getattr(mb, bank_key, None)
                    if bank is not None:
                        # Check number of non-zero entries
                        filled = (bank.norm(dim=1) > 0.5).sum().item()
                        mag_stats[f'subtype{s}_label{label}'] = filled
                        total_filled += filled
            
            stats[mag] = mag_stats
            print(f"   {mag}: total filled {total_filled} samples")
    
    return stats




def create_model_with_pretrained(args, class_counts, fold_idx):
    from models import CHIEF_MultiScale_Model

    model_seed = args.seed
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    print(f"\n🔥 Model initialization seed: {model_seed} (same for all folds)")

    initial_scale_aucs = None
    if getattr(args, 'adaptive_weight_init', False):
        initial_scale_aucs = [0.35, 0.43, 0.68]

    # 推断 clinical_dim
    use_clinical = getattr(args, 'use_clinical', False)
    clinical_dim = getattr(args, 'clinical_dim', None)
    if use_clinical and clinical_dim is None:
        try:
            from clinical_feature_CN import get_clinical_processor
            processor = get_clinical_processor()
            clinical_dim = processor.feature_dim if processor is not None else 25
        except Exception:
            clinical_dim = 25

    # [FIX 3] 解析 per-scale 配置（已在 args 中以 JSON 字符串存储）
    scale_configs = None
    if getattr(args, 'scale_configs', None):
        from config import parse_scale_configs
        scale_configs = parse_scale_configs(args)

    model = CHIEF_MultiScale_Model(
        in_dim=args.in_dim,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
        num_cls=args.num_cls,
        num_subtypes=args.num_subtypes,
        magnifications=args.magnifications,
        k_neighbors=args.k_neighbors,
        gcn_layers=getattr(args, 'gcn_layers', 2),
        dropout=args.dropout,
        temperature=getattr(args, 'attention_temperature', 1.0),
        subtype_embed_dim=args.subtype_embed_dim,
        ensemble_method=args.ensemble_method,
        scale_weights=getattr(args, 'scale_weights', None),
        ensemble_hidden_dim=getattr(args, 'ensemble_hidden_dim', 128),
        adaptive_weight_init=getattr(args, 'adaptive_weight_init', False),
        initial_scale_aucs=initial_scale_aucs,
        use_contrastive=args.use_contrastive,
        use_instance_branch=args.use_instance_branch,
        contrast_temperature=args.contrast_temperature,
        proj_dim=args.proj_dim,
        memory_bank_size=args.memory_bank_size,
        use_clinical=use_clinical,
        clinical_dim=clinical_dim if use_clinical else None,
        clinical_hidden=getattr(args, 'clinical_hidden', 128),
        clinical_dropout=getattr(args, 'clinical_dropout', 0.3),
        fusion_method=getattr(args, 'fusion_method', 'attention'),
        # [FIX 3] per-scale 超参配置
        scale_configs=scale_configs,
        # [FIX 6] 透传 warmup_epochs 保证 MemoryBank 与 trainer 配置一致
        memory_bank_warmup_epochs=getattr(args, 'warmup_epochs', 10),
        # [FIX 7] 降低 MemoryBank 动量：0.999→0.9，bank 特征与当前 backbone 同步
        memory_bank_momentum=0.9,
    ).to(args.device)
    
    pretrained_path = getattr(args, 'pretrained_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\n🔥 Fold {fold_idx + 1}: Loading pretrained weights...")
        model, loaded_keys, missing_keys = load_pretrained_weights(
            model, 
            pretrained_path, 
            device=args.device,
            strict=False
        )
        
        verify_memory_bank_loaded(model)
    else:
        print(f"\n⚠️ No pretrained weights used, starting from random initialization")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"📊 Using CHIEF style: contrastive learning={args.use_contrastive}, instance branch={args.use_instance_branch}")
    
    return model


if __name__ == '__main__':
    print("Pretrained Weights Loading Utility")
    print("Usage:")
    print("  from pretrain_utils import load_pretrained_weights")