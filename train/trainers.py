import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import math


class OptimizedWarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_lr, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = -1
        self._set_lr(warmup_lr)
        print(f"LR scheduler initialized - base lr: {warmup_lr:.2e}")
    
    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            if 'weight_lr_multiplier' in param_group:
                # ensemble fuser 参数：固定倍数缩放
                param_group['lr'] = lr * param_group['weight_lr_multiplier']
            elif 'lr_ratio' in param_group:
                # [FIX 3] per-scale 分支参数：保持各尺度间相对比例，随 scheduler 等比衰减
                param_group['lr'] = lr * param_group['lr_ratio']
            else:
                param_group['lr'] = lr
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            progress = (self.current_epoch + 1) / self.warmup_epochs
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * progress
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        self._set_lr(lr)
        return lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, mode='min', min_epochs=10, metric_name='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.metric_name = metric_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        print(f"Early stopping: metric={metric_name}, mode={mode}, patience={patience}, min_epochs={min_epochs}")
    
    def __call__(self, score, epoch):
        if epoch < self.min_epochs:
            if self.best_score is None:
                self.best_score = score
                self.best_epoch = epoch
            elif (self.mode == 'min' and score < self.best_score) or \
                 (self.mode == 'max' and score > self.best_score):
                self.best_score = score
                self.best_epoch = epoch
            return False
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            print(f"  ✓ {self.metric_name} improved! New best: {score:.4f}")
        else:
            self.counter += 1
            print(f"  ⚠️ {self.metric_name} no improvement ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered!")
                print(f"   Best {self.metric_name}: {self.best_score:.4f} (Epoch {self.best_epoch})")
                return True
        
        return False


class WeightEvolutionTracker:
    def __init__(self, magnifications):
        self.magnifications = magnifications
        self.weight_history = []
        
    def update(self, weights, scale_aucs, epoch):
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        self.weight_history.append({
            'epoch': epoch,
            'weights': weights.copy() if hasattr(weights, 'copy') else np.array(weights),
            'scale_aucs': scale_aucs.copy() if isinstance(scale_aucs, list) else scale_aucs
        })
        
    def get_weight_correlation(self):
        if len(self.weight_history) < 2:
            return 0.0
        
        current = self.weight_history[-1]
        corr = np.corrcoef(current['weights'], current['scale_aucs'])[0, 1]
        return corr if not np.isnan(corr) else 0.0
        
    def get_weight_stability(self, window=8):
        if len(self.weight_history) < window:
            return np.zeros(len(self.magnifications))
        
        recent_weights = [h['weights'] for h in self.weight_history[-window:]]
        return np.std(recent_weights, axis=0)
        
    def print_evolution_summary(self, epoch):
        if len(self.weight_history) < 2:
            return
            
        current = self.weight_history[-1]
        initial = self.weight_history[0]
        
        print(f"\nWeight evolution summary (Epoch {epoch}):")
        print(f"   Initial weights: ", end="")
        for i, mag in enumerate(self.magnifications):
            print(f"{mag}={initial['weights'][i]:.3f} ", end="")
        print()
        
        print(f"   Current weights: ", end="")
        for i, mag in enumerate(self.magnifications):
            print(f"{mag}={current['weights'][i]:.3f} ", end="")
        print()


def compute_metrics(probs, labels, preds=None, threshold=0.5):
    probs = np.array(probs)
    labels = np.array(labels)
    
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    if preds is None:
        preds = (probs >= threshold).astype(int)
    else:
        preds = np.array(preds)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    balanced_acc = (sensitivity + specificity) / 2
    
    return {
        'auc': auc,
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_acc': balanced_acc,
        'threshold': threshold,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def find_optimal_threshold(probs, labels, metric='youden', threshold_range=(0.05, 0.95)):
    probs = np.array(probs)
    labels = np.array(labels)
    
    min_thresh, max_thresh = threshold_range
    thresholds = np.arange(min_thresh, max_thresh, 0.005)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        elif metric == 'youden':
            tp = np.sum((preds == 1) & (labels == 1))
            tn = np.sum((preds == 0) & (labels == 0))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            sensitivity = tp / (tp + fn + 1e-6)
            specificity = tn / (tn + fp + 1e-6)
            score = sensitivity + specificity - 1
        elif metric == 'balanced_acc':
            tp = np.sum((preds == 1) & (labels == 1))
            tn = np.sum((preds == 0) & (labels == 0))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            sensitivity = tp / (tp + fn + 1e-6)
            specificity = tn / (tn + fp + 1e-6)
            score = (sensitivity + specificity) / 2
        else:
            score = f1_score(labels, preds, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    if best_score == 0.0:
        best_threshold = np.median(probs)
    
    return best_threshold, best_score


def _contrastive_scale(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """
    [FIX 6] cont_scale 仅保留用于进度条显示，不再参与 loss 计算。
    backbone 已由 stop-gradient 保护，此函数仅做信息展示用。
    """
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / max(1, ramp_epochs))
    return float(progress)


def train_one_epoch_chief(model, train_loader, criterion, optimizer, device, epoch, total_epochs,
                          threshold=0.5, gradient_accumulation_steps=1, weight_regularization=0.05,
                          instance_loss_weight=0.1, contrastive_loss_weight=0.1,
                          # [FIX 4b] MemoryBank 激活后对比损失渐进生效的 epoch 数。
                          # 设为 0 等价于原来的硬切换行为。
                          # 建议设为 warmup_epochs 的 50%，例如 warmup=20 则 ramp=10。
                          contrastive_ramp_epochs=10,
                          memory_bank_warmup_epochs=10):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_instance_loss = 0.0
    total_contrastive_loss = 0.0
    num_samples = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    scale_stats = {mag: {'preds': [], 'labels': [], 'probs': []}
                   for mag in model.magnifications}
    
    optimizer.zero_grad()

    # [FIX 5] cont_scale 只依赖 epoch，在每个 epoch 内为常量，提前计算一次
    cont_scale = _contrastive_scale(epoch, memory_bank_warmup_epochs, contrastive_ramp_epochs)

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", ncols=120)
    
    for batch_idx, batch_data in enumerate(progress_bar):
        slide_names, multiscale_features, multiscale_edge_indices, labels, subtypes, clinical_features = batch_data
        
        batch_loss = 0.0
        batch_size = len(slide_names)
        
        for i in range(batch_size):
            slide_features = {mag: feat.to(device) for mag, feat in multiscale_features[i].items()}
            slide_edge_indices = {mag: ei.to(device) for mag, ei in multiscale_edge_indices[i].items()}
            subtype = subtypes[i].to(device)
            label = labels[i].to(device)
            label_int = label.item()
            
            clinical_feat = clinical_features[i].to(device) if clinical_features[i] is not None else None
            
            model_output = model(
                slide_features, slide_edge_indices, subtype, 
                clinical_features=clinical_feat, label=label_int
            )
            
            if len(model_output) == 6:
                final_logits, scale_logits, scale_probs, attn_weights, ensemble_info, chief_losses = model_output
            else:
                final_logits, scale_logits, scale_probs, attn_weights, ensemble_info = model_output
                chief_losses = {}
            
            if hasattr(criterion, 'forward') and 'chief_losses' in criterion.forward.__code__.co_varnames:
                # [FIX 6] 移除 cont_scale 乘数：backbone 已由 stop-gradient 保护，
                # contrastive_loss 只通过 projector 反传，应始终以完整权重训练。
                # cont_scale 此前的零乘数导致 projector 在 epoch 10-19 接收零梯度，
                # warmup 结束时 projector 仍为随机初始化，对比损失无法收敛。
                cls_only_loss, _ = criterion(
                    final_logits.unsqueeze(0), label.unsqueeze(0), None
                )
                inst_loss = chief_losses.get('instance_loss', 0)
                cont_loss = chief_losses.get('contrastive_loss', 0)
                final_loss = (cls_only_loss
                              + instance_loss_weight * inst_loss
                              + contrastive_loss_weight * cont_loss)
            else:
                loss_output = criterion(final_logits.unsqueeze(0), label.unsqueeze(0))
                final_loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                if 'instance_loss' in chief_losses:
                    final_loss = final_loss + instance_loss_weight * chief_losses['instance_loss']
                if 'contrastive_loss' in chief_losses:
                    # [FIX 6] 同样移除 cont_scale 乘数
                    final_loss = (final_loss
                                  + contrastive_loss_weight
                                  * chief_losses['contrastive_loss'])
            
            # [FIX 2] 各分支独立监督：scale_loss 覆盖所有分支（含 clinical），
            # 确保每个分支只由自身的交叉熵损失驱动，配合 models.py 的 detach 一起
            # 彻底消除跨分支梯度耦合。
            scale_loss = 0.0
            num_scale_branches = 0
            for mag in model.magnifications:
                if mag in scale_logits:
                    scale_loss += F.cross_entropy(
                        scale_logits[mag].unsqueeze(0), label.unsqueeze(0)
                    )
                    num_scale_branches += 1
            # clinical 分支单独补入 scale_loss，否则 detach 后 clinical 无梯度
            if 'clinical' in scale_logits:
                scale_loss += F.cross_entropy(
                    scale_logits['clinical'].unsqueeze(0), label.unsqueeze(0)
                )
                num_scale_branches += 1
            
            ensemble_reg_loss = 0.0
            if hasattr(model.ensemble_fuser, 'weight_regularization'):
                try:
                    ensemble_reg_loss = model.ensemble_fuser.weight_regularization()
                except:
                    ensemble_reg_loss = 0.0
            
            total_loss_sample = (final_loss +
                               0.3 * scale_loss / max(num_scale_branches, 1) +
                               weight_regularization * ensemble_reg_loss)
            
            loss = total_loss_sample / gradient_accumulation_steps
            loss.backward()
            
            batch_loss += total_loss_sample.item()
            
            if isinstance(chief_losses.get('instance_loss'), torch.Tensor):
                total_instance_loss += chief_losses['instance_loss'].item()
            if isinstance(chief_losses.get('contrastive_loss'), torch.Tensor):
                total_contrastive_loss += chief_losses['contrastive_loss'].item()
            
            with torch.no_grad():
                final_prob = F.softmax(final_logits, dim=0)[1].item()
                final_pred = 1 if final_prob >= threshold else 0
                all_preds.append(final_pred)
                all_labels.append(label.item())
                all_probs.append(final_prob)
                
                for mag in model.magnifications:
                    if mag in scale_probs:
                        scale_prob = scale_probs[mag][1].item()
                        scale_pred = 1 if scale_prob >= threshold else 0
                        scale_stats[mag]['preds'].append(scale_pred)
                        scale_stats[mag]['labels'].append(label.item())
                        scale_stats[mag]['probs'].append(scale_prob)
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += batch_loss
        num_samples += batch_size
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        avg_inst = total_instance_loss / num_samples if num_samples > 0 else 0
        avg_cont = total_contrastive_loss / num_samples if num_samples > 0 else 0

        # [FIX 6] c_sc 改为显示 bank 数据就绪比例（filled slots / total slots），
        # 而非基于 epoch 的渐进系数（cont_scale 已从 loss 中移除）。
        # 值为 1.00 表示所有 subtype-class slot 均已积累足够样本。
        bank_ready_ratio = 0.0
        if hasattr(model, 'memory_banks') and model.memory_banks:
            total_slots = 0
            ready_slots = 0
            for mb in model.memory_banks.values():
                for s in range(mb.num_subtypes):
                    for lbl in range(2):
                        total_slots += 1
                        filled = getattr(mb, f'filled_{s}_{lbl}').item()
                        if filled >= mb.min_filled_count:
                            ready_slots += 1
            bank_ready_ratio = ready_slots / max(total_slots, 1)

        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'inst': f'{avg_inst:.4f}',
            'cont': f'{avg_cont:.4f}',
            'bnk': f'{bank_ready_ratio:.2f}'
        })
    
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    if len(all_probs) > 0:
        metrics = compute_metrics(all_probs, all_labels, all_preds, threshold)
    else:
        metrics = {'auc': 0.0, 'acc': 0.0, 'f1': 0.0}
    
    metrics['instance_loss'] = total_instance_loss / num_samples if num_samples > 0 else 0
    metrics['contrastive_loss'] = total_contrastive_loss / num_samples if num_samples > 0 else 0
    
    scale_metrics = {}
    for mag in model.magnifications:
        if len(scale_stats[mag]['probs']) > 0:
            scale_metrics[mag] = compute_metrics(
                scale_stats[mag]['probs'],
                scale_stats[mag]['labels'],
                scale_stats[mag]['preds'],
                threshold
            )
        else:
            scale_metrics[mag] = {'auc': 0.0, 'f1': 0.0, 'acc': 0.0}
    
    return avg_loss, metrics, scale_metrics



def validate_chief(model, val_loader, criterion, device, phase="Val", use_dynamic_threshold=True,
                   threshold_metric='youden', weight_tracker=None, epoch=None, 
                   save_predictions=False, fold_idx=None):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    all_labels = []
    all_ensemble_probs = []
    all_slide_names = []
    
    branch_names = model.branch_names if hasattr(model, 'branch_names') else model.magnifications
    
    scale_stats = {branch: {'labels': [], 'probs': []} for branch in branch_names}
    
    ensemble_weights_history = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"[{phase}]", ncols=120)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            slide_names, multiscale_features, multiscale_edge_indices, labels, subtypes, clinical_features = batch_data
            
            batch_size = len(slide_names)
            batch_loss = 0.0
            
            for i in range(batch_size):
                slide_features = {mag: feat.to(device) for mag, feat in multiscale_features[i].items()}
                slide_edge_indices = {mag: ei.to(device) for mag, ei in multiscale_edge_indices[i].items()}
                subtype = subtypes[i].to(device)
                label = labels[i].to(device)
                
                clinical_feat = clinical_features[i].to(device) if clinical_features[i] is not None else None
                
                model_output = model(
                    slide_features, slide_edge_indices, subtype, 
                    clinical_features=clinical_feat, label=None
                )
                
                if len(model_output) == 6:
                    final_logits, branch_logits, branch_probs, attn_weights, ensemble_info, chief_losses = model_output
                else:
                    final_logits, branch_logits, branch_probs, attn_weights, ensemble_info = model_output
                    chief_losses = {}
                
                if hasattr(criterion, 'forward') and 'chief_losses' in criterion.forward.__code__.co_varnames:
                    loss_output = criterion(final_logits.unsqueeze(0), label.unsqueeze(0), None)
                    final_loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                else:
                    loss_output = criterion(final_logits.unsqueeze(0), label.unsqueeze(0))
                    final_loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                
                batch_loss += final_loss.item()
                
                ensemble_prob = F.softmax(final_logits, dim=0)[1].item()
                all_labels.append(label.item())
                all_ensemble_probs.append(ensemble_prob)
                all_slide_names.append(slide_names[i])
                
                for branch in branch_names:
                    if branch in branch_probs:
                        branch_prob = branch_probs[branch][1].item()
                        scale_stats[branch]['labels'].append(label.item())
                        scale_stats[branch]['probs'].append(branch_prob)
                
                if 'learned_weights' in ensemble_info:
                    weights = ensemble_info['learned_weights']
                    if isinstance(weights, torch.Tensor):
                        ensemble_weights_history.append(weights.cpu().numpy())
                    else:
                        ensemble_weights_history.append(np.array(weights))
                
                if 'attention_weights' in ensemble_info:
                    attn_w = ensemble_info['attention_weights']
                    if isinstance(attn_w, torch.Tensor):
                        ensemble_weights_history.append(attn_w.cpu().numpy())
            
            total_loss += batch_loss
            num_samples += batch_size
            
            avg_loss = total_loss / num_samples
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    scale_metrics = {}
    scale_aucs = []
    
    print(f"\nBranch {phase}results:") 
    
    for branch in branch_names:
        if len(scale_stats[branch]['probs']) > 0:
            try:
                branch_auc = roc_auc_score(scale_stats[branch]['labels'], scale_stats[branch]['probs'])
            except:
                branch_auc = 0.0
            
            scale_aucs.append(branch_auc)
            
            if use_dynamic_threshold:
                best_thresh, _ = find_optimal_threshold(
                    scale_stats[branch]['probs'],
                    scale_stats[branch]['labels'],
                    metric=threshold_metric,
                    threshold_range=(0.05, 0.95)
                )
            else:
                best_thresh = 0.5
            
            branch_preds = [(1 if p >= best_thresh else 0) for p in scale_stats[branch]['probs']]
            scale_metrics[branch] = compute_metrics(
                scale_stats[branch]['probs'],
                scale_stats[branch]['labels'],
                branch_preds,
                best_thresh
            )
            
            branch_label = f"   {branch}"
            print(f"{branch_label}: AUC={scale_metrics[branch]['auc']:.4f}, "
                  f"Acc={scale_metrics[branch]['acc']:.4f}")
        else:
            scale_aucs.append(0.0)
            scale_metrics[branch] = {'auc': 0.0, 'acc': 0.0, 'threshold': 0.5}
    
    if hasattr(model.ensemble_fuser, 'update_performance_history') and len(scale_aucs) > 0:
        model.ensemble_fuser.update_performance_history(scale_aucs)

    # [FIX 7] 根据本轮验证 AUC 更新分支剪枝掩码
    if hasattr(model, 'update_branch_mask') and scale_metrics:
        branch_aucs_dict = {b: scale_metrics[b].get('auc', 0.5)
                            for b in branch_names if b in scale_metrics}
        model.update_branch_mask(branch_aucs_dict, threshold=0.58, window=20, min_active=3)
    
    if len(all_ensemble_probs) > 0:
        try:
            ensemble_auc = roc_auc_score(all_labels, all_ensemble_probs)
        except:
            ensemble_auc = 0.0
        
        if use_dynamic_threshold:
            best_threshold, _ = find_optimal_threshold(
                all_ensemble_probs, all_labels,
                metric=threshold_metric,
                threshold_range=(0.05, 0.95)
            )
        else:
            best_threshold = 0.5
        
        all_preds = [(1 if p >= best_threshold else 0) for p in all_ensemble_probs]
        metrics = compute_metrics(all_ensemble_probs, all_labels, all_preds, best_threshold)
        metrics['best_scale'] = 'ensemble'
        
        print(f"\nEnsemble results:")
        print(f"      Ensemble: AUC={ensemble_auc:.4f}, Acc={metrics['acc']:.4f}")
    else:
        metrics = {'auc': 0.0, 'acc': 0.0, 'threshold': 0.5, 'best_scale': 'ensemble'}
    
    ensemble_analysis = {}
    if ensemble_weights_history:
        avg_weights = np.mean(ensemble_weights_history, axis=0)
        std_weights = np.std(ensemble_weights_history, axis=0)
        ensemble_analysis = {
            'avg_weights': avg_weights,
            'std_weights': std_weights,
            'branch_names': branch_names,
            'scale_aucs': scale_aucs
        }
        
        print(f"\nDynamic attention ensemble weights:")
        for i, branch in enumerate(branch_names):
            branch_label = f"      {branch}"
            # [FIX 7] 显示分支激活状态
            if hasattr(model.ensemble_fuser, 'branch_active'):
                active = model.ensemble_fuser.branch_active[i].item()
                status = "     " if active > 0.5 else " [MASKED]"
            else:
                status = ""
            print(f"{branch_label}: {avg_weights[i]:.3f} ± {std_weights[i]:.3f}{status}")
        
        if weight_tracker is not None and epoch is not None:
            weight_tracker.update(avg_weights, scale_aucs, epoch)
    
    if save_predictions and fold_idx is not None:
        all_pred_labels = [(1 if prob >= best_threshold else 0) for prob in all_ensemble_probs]
        
        predictions = {
            'slide_names': all_slide_names,
            'true_labels': all_labels,
            'pred_probs': all_ensemble_probs,
            'pred_labels': all_pred_labels
        }
        
        return avg_loss, metrics, scale_metrics, ensemble_analysis, all_labels, all_ensemble_probs, predictions
    else:
        return avg_loss, metrics, scale_metrics, ensemble_analysis, all_labels, all_ensemble_probs, None

def create_optimized_optimizer(model, base_lr=2e-5, weight_decay=1e-3,
                               weight_lr_multiplier=10.0,
                               # [FIX 4c] ensemble fuser 独立 LR 倍率。
                               # 原来与 weight_lr_multiplier 共享，但 ensemble fuser
                               # 在分支已 detach 后需要更保守的 LR 才能稳定收敛。
                               # 默认 1.0 = 与 base_lr 相同，可按需调高但不超过 2.0。
                               ensemble_lr_multiplier=1.0,
                               per_scale_lr=None,
                               scale_configs=None):
    """
    构建 AdamW optimizer，支持各分支独立 param group 与独立 LR。

    LR 优先级（高 → 低）：
      1. scale_configs[mag/clinical]['lr']   — parse_scale_configs() 的返回值
      2. per_scale_lr[mag]                   — 旧接口，向后兼容
      3. base_lr                             — 全局默认

    param group 划分：
      - scale_branches.{mag}  — 每个图像尺度分支独立 group
      - clinical_branch /      — clinical 分支
        concat_classifier
      - ensemble_fuser.        — attention_net  (base_lr * weight_lr_multiplier)
        attention_net
      - ensemble_fuser.        — scale_weights  (base_lr * weight_lr_multiplier)
        scale/base_weights
      - other                  — 其余参数 (base_lr)

    scale_configs: parse_scale_configs() 的返回值，格式
        {'5x': {'gcn_hidden':..., 'lr':...}, 'clinical': {'lr':...}, ...}
    per_scale_lr: 旧接口，{'5x': float, ...}，与 scale_configs 同时存在时
        scale_configs 中的 lr 优先。
    """
    per_scale_lr    = per_scale_lr or {}
    scale_configs   = scale_configs or {}

    def _branch_lr(key):
        """从 scale_configs 或 per_scale_lr 中解析 LR，缺省用 base_lr。"""
        if key in scale_configs and 'lr' in scale_configs[key]:
            return scale_configs[key]['lr']
        if key in per_scale_lr:
            return per_scale_lr[key]
        return base_lr

    # ── 按参数名路由 ──────────────────────────────────────────────────────
    scale_branch_params = {mag: [] for mag in model.magnifications}
    clinical_params   = []
    ensemble_w_params = []
    ensemble_a_params = []
    other_params      = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        matched_scale = False
        for mag in model.magnifications:
            if name.startswith(f'scale_branches.{mag}.'):
                scale_branch_params[mag].append(param)
                matched_scale = True
                break
        if matched_scale:
            continue

        if name.startswith('clinical_branch.') or name.startswith('concat_classifier.'):
            clinical_params.append(param)
        elif 'scale_weights' in name or 'base_weights' in name:
            ensemble_w_params.append(param)
        elif 'ensemble_fuser' in name and 'attention_net' in name:
            ensemble_a_params.append(param)
        else:
            other_params.append(param)

    # ── 构建 param_groups ─────────────────────────────────────────────────
    param_groups = []

    # 其余参数（memory_bank buffer 不含梯度，不会出现在这里）
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'other'
        })

    # 每个图像尺度分支独立 group
    for mag in model.magnifications:
        params = scale_branch_params[mag]
        if not params:
            continue
        mag_lr = _branch_lr(mag)
        param_groups.append({
            'params': params,
            'lr': mag_lr,
            'lr_ratio': mag_lr / base_lr if base_lr > 0 else 1.0,
            'weight_decay': weight_decay,
            'name': f'scale_{mag}'
        })

    # clinical 分支
    if clinical_params:
        clinical_lr = _branch_lr('clinical')
        param_groups.append({
            'params': clinical_params,
            'lr': clinical_lr,
            'lr_ratio': clinical_lr / base_lr if base_lr > 0 else 1.0,
            'weight_decay': weight_decay,
            'name': 'clinical'
        })

    # ensemble fuser 权重参数
    if ensemble_w_params:
        param_groups.append({
            'params': ensemble_w_params,
            'lr': base_lr * ensemble_lr_multiplier,
            'weight_decay': 0,
            'weight_lr_multiplier': ensemble_lr_multiplier,   # scheduler 用
            'name': 'ensemble_weights'
        })

    # ensemble fuser attention_net
    if ensemble_a_params:
        param_groups.append({
            'params': ensemble_a_params,
            'lr': base_lr * ensemble_lr_multiplier,
            'weight_decay': weight_decay * 0.1,
            'weight_lr_multiplier': ensemble_lr_multiplier,   # scheduler 用
            'name': 'ensemble_attention'
        })

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr)

    # ── 日志 ─────────────────────────────────────────────────────────────
    print(f"\nOptimizer config (per-branch independent LR):")
    print(f"   base_lr:              {base_lr:.2e}")
    for mag in model.magnifications:
        print(f"   {mag} branch lr:    {_branch_lr(mag):.2e}")
    if clinical_params:
        print(f"   clinical branch lr: {_branch_lr('clinical'):.2e}")
    print(f"   ensemble fuser lr:    {base_lr * ensemble_lr_multiplier:.2e}")
    print(f"   weight_decay:         {weight_decay:.2e}")

    return optimizer


def test_model(model, test_loader, criterion, device, save_predictions=True):
    print(f"\n{'='*80}")
    print(f"Starting test set evaluation")
    print(f"{'='*80}")
    
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    all_labels = []
    all_probs = []
    all_slide_names = []
    
    branch_names = model.branch_names if hasattr(model, 'branch_names') else model.magnifications
    
    scale_stats = {branch: {'labels': [], 'probs': []} for branch in branch_names}
    
    ensemble_weights_history = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="[Test]", ncols=120)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            slide_names, multiscale_features, multiscale_edge_indices, labels, subtypes, clinical_features = batch_data
            
            batch_size = len(slide_names)
            batch_loss = 0.0
            
            for i in range(batch_size):
                slide_features = {mag: feat.to(device) for mag, feat in multiscale_features[i].items()}
                slide_edge_indices = {mag: ei.to(device) for mag, ei in multiscale_edge_indices[i].items()}
                subtype = subtypes[i].to(device)
                label = labels[i].to(device)
                
                clinical_feat = clinical_features[i].to(device) if clinical_features[i] is not None else None
                
                model_output = model(
                    slide_features, slide_edge_indices, subtype, 
                    clinical_features=clinical_feat
                )
                
                if len(model_output) == 6:
                    final_logits, branch_logits, branch_probs, attn_weights, ensemble_info, chief_losses = model_output
                else:
                    final_logits, branch_logits, branch_probs, attn_weights, ensemble_info = model_output
                
                loss_output = criterion(final_logits.unsqueeze(0), label.unsqueeze(0))
                if isinstance(loss_output, tuple):
                    final_loss = loss_output[0]
                else:
                    final_loss = loss_output
                batch_loss += final_loss.item()
                
                final_prob = F.softmax(final_logits, dim=0)[1].item()
                all_labels.append(label.item())
                all_probs.append(final_prob)
                all_slide_names.append(slide_names[i])
                
                for branch in branch_names:
                    if branch in branch_probs:
                        branch_prob = branch_probs[branch][1].item()
                        scale_stats[branch]['labels'].append(label.item())
                        scale_stats[branch]['probs'].append(branch_prob)
                
                if 'attention_weights' in ensemble_info:
                    attn_w = ensemble_info['attention_weights']
                    if isinstance(attn_w, torch.Tensor):
                        ensemble_weights_history.append(attn_w.cpu().numpy())
                elif 'learned_weights' in ensemble_info:
                    weights = ensemble_info['learned_weights']
                    if isinstance(weights, torch.Tensor):
                        ensemble_weights_history.append(weights.cpu().numpy())
                    else:
                        ensemble_weights_history.append(np.array(weights))
            
            total_loss += batch_loss
            num_samples += batch_size
            
            avg_loss = total_loss / num_samples
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    scale_metrics = {}
    print(f"\nBranch test results:")
    
    for branch in branch_names:
        if len(scale_stats[branch]['probs']) > 0:
            try:
                branch_auc = roc_auc_score(scale_stats[branch]['labels'], scale_stats[branch]['probs'])
            except:
                branch_auc = 0.0
            
            best_thresh, _ = find_optimal_threshold(
                scale_stats[branch]['probs'],
                scale_stats[branch]['labels'],
                metric='youden',
                threshold_range=(0.05, 0.95)
            )
            
            branch_preds = [(1 if p >= best_thresh else 0) for p in scale_stats[branch]['probs']]
            scale_metrics[branch] = compute_metrics(
                scale_stats[branch]['probs'],
                scale_stats[branch]['labels'],
                branch_preds,
                best_thresh
            )
            
            branch_label = f"   {branch}"
            print(f"{branch_label}: AUC={scale_metrics[branch]['auc']:.4f}, "
                  f"F1={scale_metrics[branch]['f1']:.4f}, "
                  f"Acc={scale_metrics[branch]['acc']:.4f}")
        else:
            scale_metrics[branch] = {'auc': 0.0, 'f1': 0.0, 'acc': 0.0, 'threshold': 0.5}
    
    if len(all_probs) > 0:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        best_thresh, _ = find_optimal_threshold(all_probs, all_labels, metric='youden')
        all_preds = [(1 if p >= best_thresh else 0) for p in all_probs]
        test_metrics = compute_metrics(all_probs, all_labels, all_preds, best_thresh)
        test_metrics['best_scale'] = 'ensemble'
    else:
        test_metrics = {'auc': 0.0, 'acc': 0.0, 'f1': 0.0, 'threshold': 0.5, 'best_scale': 'ensemble'}
    
    ensemble_analysis = {}
    if ensemble_weights_history:
        avg_weights = np.mean(ensemble_weights_history, axis=0)
        std_weights = np.std(ensemble_weights_history, axis=0)
        ensemble_analysis = {
            'avg_weights': avg_weights,
            'std_weights': std_weights,
            'branch_names': branch_names
        }
        
        print(f"\nDynamic attention ensemble weights:")
        for i, branch in enumerate(branch_names):
            branch_label = f"   {branch}"
            print(f"{branch_label}: {avg_weights[i]:.3f} ± {std_weights[i]:.3f}")
    
    print(f"\n{'='*80}")
    print(f"Final test results (ensemble)")
    print(f"{'='*80}")
    print(f"   Loss:        {avg_loss:.4f}")
    print(f"   AUC:         {test_metrics['auc']:.4f}")
    print(f"   Accuracy:    {test_metrics['acc']:.4f}")
    print(f"   F1-Score:    {test_metrics['f1']:.4f}")
    print(f"   Precision:   {test_metrics['precision']:.4f}")
    print(f"   Recall:      {test_metrics['recall']:.4f}")
    print(f"   Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {test_metrics['specificity']:.4f}")
    print(f"{'='*80}")
    
    test_results = {
        'loss': avg_loss,
        'metrics': test_metrics,
        'scale_metrics': scale_metrics,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'all_slide_names': all_slide_names,
        'ensemble_analysis': ensemble_analysis
    }
    
    return test_results


train_one_epoch_optimized = train_one_epoch_chief
validate_optimized = validate_chief