import torch
import os
import csv
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

from models import (
    PC_TMB_MultiScale_Ensemble_Optimized_Model,
    ClassBalancedFocalLossWithSmoothing
)
from data_loader import (
    load_test_data,
    load_all_folds_combined,
    load_and_organize_data_with_cache,
    load_and_organize_test_data,
    MultiscaleDataset,
    custom_collate,
    BalancedUnderSampler,
    calculate_class_weights,
    calculate_dynamic_sample_weights
)
from trainers import (
    OptimizedWarmupCosineScheduler,
    create_optimized_optimizer,
    test_model  # 确保导入更新后的test_model
)
from config import set_seed, worker_init_fn


def train_final_and_test(args):
    print("\n" + "="*80)
    print("="*80)
    
    test_log_dir = os.path.join(args.log_dir, args.name, 'final_test')
    os.makedirs(test_log_dir, exist_ok=True)
    
    writer = SummaryWriter(test_log_dir)
    
    if not args.skip_training:
        print("\n" + "="*50)
        print("📦 Step 1: Merging training and validation sets")
        print("="*50)
        
        combined_slide_to_label = load_all_folds_combined(args.csv_dir, args.n_folds)
        
        print(f"\n📦 Loading merged training data features...")
        train_cache = load_and_organize_data_with_cache(
            args, fold_idx=999,
            slide_to_label=combined_slide_to_label, 
            phase='train'
        )
        
        train_slide_names = train_cache['slide_names']
        train_features = train_cache['features']
        train_edge_indices = train_cache['edge_indices']
        train_labels = train_cache['labels']
        train_subtypes = train_cache['subtypes']
        
        if args.use_undersampling:
            undersampler = BalancedUnderSampler(
                target_ratio=args.undersample_ratio,
                minority_aug_ratio=args.minority_aug_ratio
            )
            train_coords = train_cache.get('coords', None)
            train_patch_names = train_cache.get('patch_names', None)
            
            (train_slide_names, train_features, train_edge_indices,
             train_coords, train_patch_names, train_labels, train_subtypes) = \
                undersampler.balance_dataset(
                    train_slide_names, train_features, train_edge_indices,
                    train_coords, train_patch_names, train_labels, train_subtypes
                )
        
        print(f"\n📊 Merged training set: {len(train_slide_names)} slides")
    
    print("\n" + "="*50)
    print("📦 Step 2: Loading test set")
    print("="*50)
    
    test_slide_to_label = load_test_data(args.test_csv)
    test_cache = load_and_organize_test_data(args, test_slide_to_label)
    
    test_slide_names = test_cache['slide_names']
    test_features = test_cache['features']
    test_edge_indices = test_cache['edge_indices']
    test_labels = test_cache['labels']
    test_subtypes = test_cache['subtypes']
    
    print(f"📊 Test set: {len(test_slide_names)} slides")
    
    test_label_list = [l.item() if isinstance(l, torch.Tensor) else l for l in test_labels]
    test_class_counts = Counter(test_label_list)
    print(f"   Class distribution:")
    for cls, count in sorted(test_class_counts.items()):
        print(f"   - Class {cls}: {count} ({count/len(test_label_list)*100:.1f}%)")
    
    if not args.skip_training:
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
            train_subtypes
        )
        
        if args.use_weighted_sampling:
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
                replacement=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, sampler=sampler, 
                collate_fn=custom_collate, num_workers=0
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, 
                collate_fn=custom_collate, num_workers=0
            )
    
    test_dataset = MultiscaleDataset(
        test_slide_names, 
        test_features, 
        test_edge_indices, 
        test_labels, 
        test_subtypes
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=custom_collate, num_workers=0
    )
    
    print("\n" + "="*50)
    print("🔧 Step 3: Creating model")
    print("="*50)
    
    initial_scale_aucs = None
    if args.adaptive_weight_init:
        initial_scale_aucs = [0.35, 0.43, 0.68]
    
    model = PC_TMB_MultiScale_Ensemble_Optimized_Model(
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
        attention_type=args.attention_type,
        subtype_embed_dim=args.subtype_embed_dim,
        use_subtype_in_classifier=args.use_subtype_in_classifier,
        ensemble_method=args.ensemble_method,
        scale_weights=args.scale_weights,
        ensemble_hidden_dim=args.ensemble_hidden_dim,
        adaptive_weight_init=args.adaptive_weight_init,
        initial_scale_aucs=initial_scale_aucs,
        # [FIX 7] 与 MAIN.py 保持一致：降低 bank 动量，并透传 warmup_epochs
        memory_bank_warmup_epochs=getattr(args, 'warmup_epochs', 10),
        memory_bank_momentum=0.9,
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    if args.load_checkpoint:
        print(f"\n📥 Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model weights loaded successfully")
    
    if not args.skip_training:
        print("\n" + "="*50)
        print("🏋️ Step 4: Full data training")
        print("="*50)
        
        if args.use_focal_loss and args.use_label_smoothing:
            criterion = ClassBalancedFocalLossWithSmoothing(
                class_counts=class_counts if args.use_weighted_loss else None,
                num_classes=args.num_cls,
                gamma=args.focal_gamma,
                beta=args.effective_beta,
                smoothing=args.label_smoothing
            )
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = create_optimized_optimizer(
            model, 
            base_lr=args.lr,
            weight_decay=args.weight_decay, 
            weight_lr_multiplier=args.weight_lr_multiplier
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
        
        best_train_loss = float('inf')
        best_train_auc = 0.0
        
        for epoch in range(1, args.EPOCH + 1):
            print(f"\n{'='*60}")
            print(f"Final Training - Epoch {epoch}/{args.EPOCH}")
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
            print(f"{'='*60}")
            
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
                    generator=torch.Generator().manual_seed(args.seed + epoch)
                )
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, sampler=sampler,
                    collate_fn=custom_collate, num_workers=0,
                    worker_init_fn=worker_init_fn
                )
            
            from trainers import train_one_epoch_chief
            train_loss, train_metrics, _ = train_one_epoch_chief(
                model, train_loader, criterion, optimizer, args.device,
                epoch, args.EPOCH, threshold=0.5,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                weight_regularization=args.weight_regularization,
                instance_loss_weight=getattr(args, 'instance_weight', 0.1),
                contrastive_loss_weight=getattr(args, 'contrastive_weight', 0.1),
                memory_bank_warmup_epochs=getattr(args, 'warmup_epochs', 10),
            )
            
            if scheduler is not None:
                scheduler.step()
            
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if train_metrics['auc'] > best_train_auc:
                best_train_auc = train_metrics['auc']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_auc': train_metrics['auc'],
                    'args': vars(args)
                }
                torch.save(checkpoint, os.path.join(test_log_dir, 'final_best_model.pth'))
            
            print(f"Epoch {epoch} training completed - Loss: {train_loss:.4f}, AUC: {train_metrics['auc']:.4f}")
            print(f"Best training Loss: {best_train_loss:.4f}, Best training AUC: {best_train_auc:.4f}")
        
        final_checkpoint = {
            'epoch': args.EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_train_loss': best_train_loss,
            'best_train_auc': best_train_auc
        }
        torch.save(final_checkpoint, os.path.join(test_log_dir, 'final_model.pth'))
    
    print("\n" + "="*50)
    print("📊 Step 5: Test set evaluation")
    print("="*50)
    
    if not args.skip_training and os.path.exists(os.path.join(test_log_dir, 'final_best_model.pth')):
        checkpoint = torch.load(os.path.join(test_log_dir, 'final_best_model.pth'), map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best trained model (Epoch {checkpoint['epoch']})")
    
    # 使用更新后的test_model，它已经包含了分支比较逻辑
    test_results = test_model(
        model, test_loader, criterion, args.device, threshold=0.5
    )
    
    test_metrics = test_results['metrics']
    
    from utils import plot_confusion_matrix, plot_roc_curve
    
    plot_confusion_matrix(test_results['all_labels'], 
                         [1 if p >= test_metrics['threshold'] else 0 for p in test_results['all_probs']],
                         class_names=['Class 0', 'Class 1'],
                         title='Test Confusion Matrix',
                         save_path=os.path.join(test_log_dir, 'confusion_matrix.png'))
    
    plot_roc_curve(test_results['all_labels'], test_results['all_probs'],
                   title='Test ROC Curve',
                   save_path=os.path.join(test_log_dir, 'roc_curve.png'))
    
    test_result_csv = os.path.join(test_log_dir, 'test_results.csv')
    with open(test_result_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
        writer.writeheader()
        writer.writerow(test_metrics)
    
    writer.close()
    
    return test_metrics