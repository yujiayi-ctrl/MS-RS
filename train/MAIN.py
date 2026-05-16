import torch
import os
import csv
import numpy as np 
from torch.utils.tensorboard import SummaryWriter

from config import create_parser, set_seed, worker_init_fn, parse_scale_configs
from test_functions import train_final_and_test
from models import (
    CHIEF_MultiScale_Model,
    CHIEFLoss
)
from data_loader import (
    load_fold_data,
    load_and_organize_data_with_cache,
    MultiscaleDataset,
    custom_collate,
    calculate_class_weights,
    calculate_dynamic_sample_weights,
    BalancedUnderSampler
)
from trainers import (
    OptimizedWarmupCosineScheduler,
    EarlyStopping,
    WeightEvolutionTracker,
    train_one_epoch_chief,
    validate_chief,
    create_optimized_optimizer
)
from utils import (
    eval_metric,
    plot_confusion_matrix,
    plot_roc_curve
)


def save_validation_predictions(predictions, fold_log_dir, fold_idx, threshold):

    import csv
    import os
    
    slide_names = predictions['slide_names']
    true_labels = predictions['true_labels']
    pred_probs = predictions['pred_probs']
    
    os.makedirs(fold_log_dir, exist_ok=True)
    
    pred_labels = [1 if prob >= threshold else 0 for prob in pred_probs]
    
    csv_path = os.path.join(fold_log_dir, f'fold{fold_idx}_val_predictions.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['slide_name', 'true_label', 'pred_prob', 'pred_label'])
        
        for i, (slide_name, true_label, pred_prob, pred_label) in enumerate(zip(slide_names, true_labels, pred_probs, pred_labels)):
            writer.writerow([slide_name, true_label, f"{pred_prob:.16f}", pred_label])
    
    save_detailed_metrics(true_labels, pred_probs, pred_labels, threshold, fold_log_dir, fold_idx)



def save_detailed_metrics(true_labels, pred_probs, pred_labels, threshold, fold_log_dir, fold_idx):

    import csv
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score
    
    auc = roc_auc_score(true_labels, pred_probs) if len(set(true_labels)) > 1 else 0.0
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    
    tn = sum((np.array(pred_labels) == 0) & (np.array(true_labels) == 0))
    fp = sum((np.array(pred_labels) == 1) & (np.array(true_labels) == 0))
    specificity = tn / (tn + fp + 1e-6)
    
    kappa = cohen_kappa_score(true_labels, pred_labels) if len(set(true_labels)) > 1 else 0.0
    
    tp = sum((np.array(pred_labels) == 1) & (np.array(true_labels) == 1))
    tn = sum((np.array(pred_labels) == 0) & (np.array(true_labels) == 0))
    fp = sum((np.array(pred_labels) == 1) & (np.array(true_labels) == 0))
    fn = sum((np.array(pred_labels) == 0) & (np.array(true_labels) == 1))
    
    metrics_path = os.path.join(fold_log_dir, f'evaluation_results.csv')
    
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'auc', 'f1_score', 'accuracy', 'precision', 'recall', 
                         'specificity', 'kappa', 'threshold', 'tp', 'tn', 'fp', 'fn'])
        
        writer.writerow([
            fold_idx,
            f"{auc:.16f}",
            f"{f1:.16f}",
            f"{accuracy:.16f}",
            f"{precision:.16f}",
            f"{recall:.16f}",
            f"{specificity:.16f}",
            f"{kappa:.16f}",
            f"{threshold:.16f}",
            tp, tn, fp, fn
        ])
    
def train_fold_chief(args, fold_idx):

    fold_seed = args.seed + fold_idx
    set_seed(fold_seed)
    print(f"\nFold {fold_idx} seed: {fold_seed} (base_seed={args.seed} + fold_idx={fold_idx})")
    
    print(f"\n{'='*80}")
    print(f"Training Fold {fold_idx + 1}/{args.n_folds}")
    print(f"WSI-level contrastive learning + Memory Bank + multi-branch attention")
    print(f"Ensemble method: {args.ensemble_method}")
    print(f"{'='*80}")
    
    fold_log_dir = os.path.join(args.log_dir, args.name, f'fold_{fold_idx}')
    os.makedirs(fold_log_dir, exist_ok=True)
    
    writer = SummaryWriter(fold_log_dir)
    
    csv_path = os.path.join(args.csv_dir, f'fold{fold_idx}.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found {csv_path}")
        return None, None
    
    train_slide_to_label, val_slide_to_label = load_fold_data(csv_path)
    
    print(f"\nTrain set: {len(train_slide_to_label)} slides")
    print(f"Val set: {len(val_slide_to_label)} slides")
    
    print("\n" + "="*50)
    print("Loading training data...")
    print("="*50)
    train_cache = load_and_organize_data_with_cache(args, fold_idx, train_slide_to_label, 'train')
    
    train_slide_names = train_cache['slide_names']
    train_features = train_cache['features']
    train_edge_indices = train_cache['edge_indices']
    train_labels = train_cache['labels']
    train_subtypes = train_cache['subtypes']
    train_coords = train_cache.get('coords', None)
    train_patch_names = train_cache.get('patch_names', None)
    train_clinical_features = train_cache.get('clinical_features', [None] * len(train_slide_names))
    if args.use_undersampling:
        undersampler = BalancedUnderSampler(...)
        (train_slide_names, train_features, train_edge_indices,
        train_coords, train_patch_names, train_labels, train_subtypes,
        train_clinical_features) = \
            undersampler.balance_dataset(
                train_slide_names, train_features, train_edge_indices,
                train_coords, train_patch_names, train_labels, train_subtypes,
                train_clinical_features,
                minority_label=1
            )
    
    val_cache = load_and_organize_data_with_cache(args, fold_idx, val_slide_to_label, 'val')
    
    val_slide_names = val_cache['slide_names']
    val_features = val_cache['features']
    val_edge_indices = val_cache['edge_indices']
    val_labels = val_cache['labels']
    val_subtypes = val_cache['subtypes']
    val_clinical_features = val_cache.get('clinical_features', [None] * len(val_slide_names))
    class_weights, class_counts = calculate_class_weights(
        train_labels,
        num_classes=args.num_cls,
        strategy=args.weight_strategy,
        beta=args.effective_beta,
        temperature=args.sampling_temperature,
        smooth_factor=args.sampling_smooth_factor
    )
    class_weights = class_weights.to(args.device)
    
    train_dataset = MultiscaleDataset(
        train_slide_names,
        train_features,
        train_edge_indices,
        train_labels,
        train_subtypes,
        train_clinical_features
    )
    val_dataset = MultiscaleDataset(
        val_slide_names,
        val_features,
        val_edge_indices,
        val_labels,
        val_subtypes,
        val_clinical_features
    )
    
    if args.use_weighted_sampling:
        print(f"\nUsing dynamic weighted sampling:")
        sample_weights = calculate_dynamic_sample_weights(
            train_labels,
            class_weights.cpu(),
            epoch=0,
            warmup_epochs=args.dynamic_sampling_warmup,
            temperature=args.sampling_temperature,
            smooth_factor=args.sampling_smooth_factor
        )
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(fold_seed)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            collate_fn=custom_collate, num_workers=0,
            worker_init_fn=worker_init_fn
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=custom_collate, num_workers=0,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(fold_seed)
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=custom_collate, num_workers=0,
        worker_init_fn=worker_init_fn
    )
    
    clinical_dim = None
    if getattr(args, 'use_clinical', False):
        from clinical_feature_CN import get_clinical_processor
        clinical_processor = get_clinical_processor(
            getattr(args, 'clinical_csv', None),
            cache_dir=getattr(args, 'cache_dir', None)
        )
        if clinical_processor:
            clinical_dim = clinical_processor.feature_dim

    # [FIX 3] 解析 --scale_configs JSON，得到 per-scale 超参 dict
    scale_configs = parse_scale_configs(args)

    initial_scale_aucs = None
    if args.adaptive_weight_init:
        num_scales = len(args.magnifications)
        if getattr(args, 'use_clinical', False):
            num_scales += 1
        initial_scale_aucs = None

    model = CHIEF_MultiScale_Model(
        in_dim=args.in_dim,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
        num_cls=args.num_cls,
        num_subtypes=args.num_subtypes,
        magnifications=args.magnifications,
        k_neighbors=args.k_neighbors,
        gcn_layers=args.gcn_layers,
        dropout=args.dropout,
        temperature=args.attention_temperature,
        subtype_embed_dim=args.subtype_embed_dim,
        ensemble_method=args.ensemble_method,
        scale_weights=args.scale_weights,
        ensemble_hidden_dim=args.ensemble_hidden_dim,
        adaptive_weight_init=args.adaptive_weight_init,
        initial_scale_aucs=initial_scale_aucs,
        use_contrastive=args.use_contrastive,
        use_instance_branch=args.use_instance_branch,
        contrast_temperature=args.contrast_temperature,
        proj_dim=args.proj_dim,
        memory_bank_size=args.memory_bank_size,
        use_clinical=getattr(args, 'use_clinical', False),
        clinical_dim=clinical_dim,
        clinical_hidden=getattr(args, 'clinical_hidden', 256),
        clinical_dropout=getattr(args, 'clinical_dropout', 0.3),
        fusion_method=getattr(args, 'fusion_method', 'attention'),
        # [FIX 3] 透传 per-scale 超参配置
        scale_configs=scale_configs,
        # [FIX 6] 透传 warmup_epochs 保证 MemoryBank 与 trainer 配置一致
        memory_bank_warmup_epochs=getattr(args, 'warmup_epochs', 10),
        # [FIX 7] 降低 MemoryBank 动量：0.999→0.9，bank 特征与当前 backbone 同步
        memory_bank_momentum=0.9,
    ).to(args.device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        checkpoint = torch.load(args.pretrained_path, map_location=args.device)
        pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
        model_dict = model.state_dict()
        
        filtered_dict = {}
        loaded_count = 0
        skipped_count = 0
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
                loaded_count += 1
            else:
                skipped_count += 1
        
        model.load_state_dict(filtered_dict, strict=False)
        # [FIX 2] 预训练权重加载后 current_epoch buffer 可能被覆盖为非零值，
        # 重置为 0 确保冷启动保护从第 1 epoch 正确计数。
        if hasattr(model, 'set_memory_bank_epoch'):
            model.set_memory_bank_epoch(0)
    elif args.pretrained_path:
        print(f"Pretrained weights not found: {args.pretrained_path}")
    
    criterion = CHIEFLoss(
        class_counts=class_counts if args.use_weighted_loss else None,
        num_classes=args.num_cls,
        gamma=args.focal_gamma,
        beta=args.effective_beta,
        smoothing=args.label_smoothing,
        instance_weight=args.instance_weight,
        contrastive_weight=args.contrastive_weight
    )
    
    optimizer = create_optimized_optimizer(
        model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        weight_lr_multiplier=args.weight_lr_multiplier,
        # [FIX 4c] ensemble fuser 独立 LR 倍率
        ensemble_lr_multiplier=getattr(args, 'ensemble_lr_multiplier', 1.0),
        # [FIX 3] 透传 per-scale LR 配置
        scale_configs=scale_configs,
    )
    
    scheduler = None
    if args.use_lr_scheduler and args.scheduler_type == 'cosine_warmup':
        scheduler = OptimizedWarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.EPOCH,
            warmup_lr=args.warmup_lr,
            base_lr=args.lr,
            min_lr=args.min_lr
        )
    
    patience = args.patience
    min_epochs = args.min_epochs
    patience_counter = 0
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_threshold = 0.5
    best_scale = 'ensemble'
    best_scale_metrics = {}
    best_ensemble_analysis = {}
    
    print(f"\nEarly stopping config:")
    print(f"   - Monitor: {args.early_stop_metric}")
    print(f"   - Patience: {patience}")
    print(f"   - Min epochs: {min_epochs}")
    
    weight_tracker = WeightEvolutionTracker(args.magnifications)
    
    for epoch in range(1, args.EPOCH + 1):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx + 1} - Epoch {epoch}/{args.EPOCH}")
        
        current_lr = optimizer.param_groups[0]['lr']
        weight_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else current_lr
        print(f"LR: base={current_lr:.2e}, weight={weight_lr:.2e}")
        print(f"{'='*80}")
        
        # [FIX 2] 每个 epoch 开始时将当前 epoch 号同步到所有 MemoryBank。
        # 这是冷启动保护生效的必要条件：MemoryBank._is_ready() 内部读取
        # current_epoch buffer 与 warmup_epochs 比较，若此处不调用则
        # current_epoch 永远为 0，get_positives 永远返回 None，
        # 对比损失在整个训练过程中始终为 0.0000（即日志中观察到的现象）。
        if hasattr(model, 'set_memory_bank_epoch'):
            model.set_memory_bank_epoch(epoch)

        if args.use_weighted_sampling and epoch > 1:
            sample_weights = calculate_dynamic_sample_weights(
                train_labels,
                class_weights.cpu(),
                epoch=epoch-1,
                warmup_epochs=args.dynamic_sampling_warmup,
                temperature=args.sampling_temperature,
                smooth_factor=args.sampling_smooth_factor
            )
            
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=torch.Generator().manual_seed(fold_seed + epoch)
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, sampler=sampler,
                collate_fn=custom_collate, num_workers=0,
                worker_init_fn=worker_init_fn
            )
        
        train_loss, train_metrics, train_scale_metrics = train_one_epoch_chief(
            model, train_loader, criterion, optimizer, args.device,
            epoch, args.EPOCH, threshold=best_threshold,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_regularization=args.weight_regularization,
            instance_loss_weight=args.instance_weight,
            contrastive_loss_weight=args.contrastive_weight,
            # [FIX 4b] 对比损失渐进激活参数
            contrastive_ramp_epochs=getattr(args, 'contrastive_ramp_epochs', 10),
            memory_bank_warmup_epochs=getattr(args, 'warmup_epochs', 10),
        )
        
        val_loss, val_metrics, val_scale_metrics, ensemble_analysis, all_labels, all_probs, val_predictions = validate_chief(
            model, val_loader, criterion, args.device, "Val",
            use_dynamic_threshold=args.use_dynamic_threshold,
            threshold_metric=args.threshold_search_metric,
            weight_tracker=weight_tracker,
            epoch=epoch,
            save_predictions=True,
            fold_idx=fold_idx
        )
        
        if scheduler is not None:
            scheduler.step()
        
        best_scale = val_metrics.get('best_scale', 'ensemble')
        
        overfit_gap = train_metrics['auc'] - val_metrics['auc']
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('AUC/train', train_metrics['auc'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics.get('acc', 0.0), epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('Overfit_Gap', overfit_gap, epoch)
        
        if 'instance_loss' in train_metrics:
            writer.add_scalar('CHIEF/instance_loss', train_metrics['instance_loss'], epoch)
            writer.add_scalar('CHIEF/contrastive_loss', train_metrics['contrastive_loss'], epoch)
        
        improved = False
        if val_metrics['auc'] > best_val_auc:
            best_val_loss = val_loss
            best_val_f1 = val_metrics['f1']
            best_val_auc = val_metrics['auc']
            best_val_acc = val_metrics.get('acc', 0.0)
            best_epoch = epoch
            best_threshold = val_metrics['threshold']
            best_scale = val_metrics.get('best_scale', 'ensemble')
            best_scale_metrics = val_scale_metrics.copy()
            best_ensemble_analysis = ensemble_analysis.copy() if ensemble_analysis else {}
            
            if val_predictions:
                save_validation_predictions(val_predictions, fold_log_dir, fold_idx, best_threshold)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'best_threshold': best_threshold,
                'best_scale': best_scale,
                'scale_metrics': best_scale_metrics,
                'ensemble_analysis': best_ensemble_analysis,
                'fold_seed': fold_seed,
                'args': vars(args),
                'val_predictions': val_predictions
            }
            
            checkpoint_path = os.path.join(fold_log_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"\nBest model saved")
            
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1
        
        if improved:
            print(f"New best AUC: {val_metrics['auc']:.4f}")
        else:
            print(f"No AUC improvement ({patience_counter}/{patience})")
        
        if epoch >= min_epochs:
            if patience_counter >= patience:
                print(f"\n{'='*80}")
                print(f"Early stopping at epoch {epoch}")
                print(f"   Best model from epoch {best_epoch}")
                print(f"   Best AUC: {best_val_auc:.4f}")
                print(f"   Best Acc: {best_val_acc:.4f}")
                print(f"{'='*80}\n")
                break
        else:
            print(f"   Min epochs protection: {epoch}/{min_epochs}")
    
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx + 1} training complete!")
    print(f"Best: Loss={best_val_loss:.4f}, F1={best_val_f1:.4f}, "
          f"AUC={best_val_auc:.4f}, Acc={best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Fold seed: {fold_seed}")
    print(f"{'='*80}\n")
    
    eval_csv_path = os.path.join(fold_log_dir, 'evaluation_results.csv')
    with open(eval_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer_csv = csv.DictWriter(f, fieldnames=['fold', 'fold_seed', 'val_loss', 'val_auc', 'val_f1', 
                                                   'val_acc', 'threshold', 'attention_type', 'best_scale'])
        writer_csv.writeheader()
        writer_csv.writerow({
            'fold': fold_idx,
            'fold_seed': fold_seed,
            'val_loss': best_val_loss,
            'val_auc': best_val_auc,
            'val_f1': best_val_f1,
            'val_acc': best_val_acc,
            'threshold': best_threshold,
            'attention_type': 'chief',
            'best_scale': best_scale
        })
    
    result = {
        'f1': best_val_f1,
        'auc': best_val_auc,
        'loss': best_val_loss,
        'acc': best_val_acc,
        'threshold': best_threshold,
        'attention_type': 'chief',
        'best_scale': best_scale,
        'fold_seed': fold_seed
    }
    
    writer.close()
    
    return result, best_epoch


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.test_mode:
        train_final_and_test(args)
        return
    
    print(f"\n{'='*80}")
    print(f"Starting WSI classification training")
    print(f"Cross-validation: {args.n_folds} folds")
    print(f"Base seed: {args.seed}")
    print(f"{'='*80}")
    
    all_fold_results = []
    
    folds_to_run = range(args.n_folds) if args.run_fold == -1 else [args.run_fold]
    
    for fold_idx in folds_to_run:
        fold_result, best_epoch = train_fold_chief(args, fold_idx)
        if fold_result:
            all_fold_results.append(fold_result)
    
    if all_fold_results:

        all_fold_metrics = []
        
        for i, result in enumerate(all_fold_results):
            metrics_path = os.path.join(args.log_dir, args.name, f'fold_{i}', f'fold{i}_detailed_metrics.csv')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_fold_metrics.append(row)
        
        avg_auc = np.mean([float(r['auc']) for r in all_fold_metrics])
        avg_f1 = np.mean([float(r['f1_score']) for r in all_fold_metrics])
        avg_acc = np.mean([float(r['accuracy']) for r in all_fold_metrics])
        avg_precision = np.mean([float(r['precision']) for r in all_fold_metrics])
        avg_recall = np.mean([float(r['recall']) for r in all_fold_metrics])
        avg_specificity = np.mean([float(r['specificity']) for r in all_fold_metrics])
        avg_kappa = np.mean([float(r['kappa']) for r in all_fold_metrics])
        avg_tp = np.mean([int(r['tp']) for r in all_fold_metrics])
        avg_tn = np.mean([int(r['tn']) for r in all_fold_metrics])
        avg_fp = np.mean([int(r['fp']) for r in all_fold_metrics])
        avg_fn = np.mean([int(r['fn']) for r in all_fold_metrics])
        
        print(f"\n{'='*80}")
        print(f"All folds complete - Average results")
        print(f"   Avg AUC: {avg_auc:.4f}")
        print(f"   Avg F1: {avg_f1:.4f}")
        print(f"   Avg Accuracy: {avg_acc:.4f}")
        print(f"   Avg Precision: {avg_precision:.4f}")
        print(f"   Avg Recall: {avg_recall:.4f}")
        print(f"   Avg Specificity: {avg_specificity:.4f}")
        print(f"   Avg Kappa: {avg_kappa:.4f}")
        print(f"{'='*80}")
        
        summary_csv_path = os.path.join(args.log_dir, args.name, 'summary_results.csv')
        os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
        
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'fold', 'auc', 'f1_score', 'accuracy', 'precision', 'recall',
                'specificity', 'kappa', 'threshold', 'tp', 'tn', 'fp', 'fn'
            ])
            writer.writeheader()
            
            for i, metrics in enumerate(all_fold_metrics):
                writer.writerow({
                    'fold': i,
                    'auc': metrics['auc'],
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'specificity': metrics['specificity'],
                    'kappa': metrics['kappa'],
                    'threshold': metrics['threshold'],
                    'tp': metrics['tp'],
                    'tn': metrics['tn'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn']
                })
            
            writer.writerow({
                'fold': 'average',
                'auc': f"{avg_auc:.16f}",
                'f1_score': f"{avg_f1:.16f}",
                'accuracy': f"{avg_acc:.16f}",
                'precision': f"{avg_precision:.16f}",
                'recall': f"{avg_recall:.16f}",
                'specificity': f"{avg_specificity:.16f}",
                'kappa': f"{avg_kappa:.16f}",
                'threshold': '',
                'tp': f"{avg_tp:.2f}",
                'tn': f"{avg_tn:.2f}",
                'fp': f"{avg_fp:.2f}",
                'fn': f"{avg_fn:.2f}"
            })


if __name__ == '__main__':
    main()