import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import math


class MemoryBank(nn.Module):
    def __init__(self, feature_dim, bank_size=256, num_subtypes=4, momentum=0.999,
                 warmup_epochs=10,
                 min_filled_ratio=0.1):
        super(MemoryBank, self).__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.num_subtypes = num_subtypes
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.min_filled_count = max(1, int(bank_size * min_filled_ratio))

        self.register_buffer('current_epoch', torch.zeros(1, dtype=torch.long))

        for s in range(num_subtypes):
            for label in range(2):
                self.register_buffer(
                    f'bank_{s}_{label}',
                    torch.zeros(bank_size, feature_dim)
                )
                self.register_buffer(
                    f'ptr_{s}_{label}',
                    torch.zeros(1, dtype=torch.long)
                )
                self.register_buffer(
                    f'filled_{s}_{label}',
                    torch.zeros(1, dtype=torch.long)
                )

    def set_epoch(self, epoch: int):
        """Called by the trainer at the beginning of each epoch to determine the epoch for cold-start protection."""
        self.current_epoch.fill_(epoch)

    def _is_ready(self, subtype_idx: int, label: int) -> bool:

        filled = getattr(self, f'filled_{subtype_idx}_{label}').item()
        return filled >= self.min_filled_count

    @torch.no_grad()
    def update(self, features, subtype_label, class_label, is_training=True):
        if not is_training:
            return
        if class_label is None:
            return

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.size(0) > 1:
            features = features[0:1]
        features = features.squeeze(0)

        if features.size(0) != self.feature_dim:
            if features.size(0) > self.feature_dim:
                features = features[:self.feature_dim]
            else:
                padding = torch.zeros(self.feature_dim - features.size(0), device=features.device)
                features = torch.cat([features, padding], dim=0)

        features = F.normalize(features, dim=0)

        bank_key = f'bank_{subtype_label}_{class_label}'
        ptr_key = f'ptr_{subtype_label}_{class_label}'
        filled_key = f'filled_{subtype_label}_{class_label}'

        bank = getattr(self, bank_key)
        ptr = getattr(self, ptr_key)
        filled = getattr(self, filled_key)
        ptr_val = ptr.item()

        if bank[ptr_val].norm() < 1e-6:
            bank[ptr_val] = features
        else:
            bank[ptr_val] = self.momentum * bank[ptr_val] + (1 - self.momentum) * features

        new_ptr = (ptr_val + 1) % self.bank_size
        getattr(self, ptr_key).fill_(new_ptr)

        new_filled = min(filled.item() + 1, self.bank_size)
        getattr(self, filled_key).fill_(new_filled)

    def get_positives(self, subtype_idx, label, exclude_idx=None):

        if label is None:
            return None

        if not self._is_ready(subtype_idx, label):
            return None
        bank = getattr(self, f'bank_{subtype_idx}_{label}')
        mask = bank.norm(dim=1) > 0.5
        valid = bank[mask]
        return valid if valid.size(0) > 0 else None

    def get_negatives(self, subtype_idx, label, max_negatives=128):

        if label is None:
            return None
        negatives = []
        other_label = 1 - label

        if self._is_ready(subtype_idx, other_label):
            bank = getattr(self, f'bank_{subtype_idx}_{other_label}')
            mask = bank.norm(dim=1) > 0.5
            if mask.sum() > 0:
                negatives.append(bank[mask])

        for s in range(self.num_subtypes):
            if s != subtype_idx:
                for l in range(2):
                    if self._is_ready(s, l):
                        bank = getattr(self, f'bank_{s}_{l}')
                        mask = bank.norm(dim=1) > 0.5
                        if mask.sum() > 0:
                            negatives.append(bank[mask])

        if len(negatives) == 0:
            return None

        all_negatives = torch.cat(negatives, dim=0)

        if all_negatives.size(0) > max_negatives:
            perm = torch.randperm(all_negatives.size(0), device=all_negatives.device)
            all_negatives = all_negatives[perm[:max_negatives]]

        return all_negatives

    def get_bank_stats(self):

        stats = {}
        epoch = self.current_epoch.item()
        for s in range(self.num_subtypes):
            for label in range(2):
                filled = getattr(self, f'filled_{s}_{label}').item()
                ready = self._is_ready(s, label)
                stats[f's{s}_l{label}'] = {
                    'filled': filled,
                    'ready': ready,
                    'epoch': epoch,
                    'warmup_epochs': self.warmup_epochs,
                    'min_filled_count': self.min_filled_count,
                }
        return stats


class WSIContrastiveBranch(nn.Module):
    def __init__(self, feature_dim, proj_dim=256, temperature=0.07):
        super(WSIContrastiveBranch, self).__init__()
        self.temperature = temperature

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim)
        )

    def forward(self, wsi_embedding):
        projected = self.projector(wsi_embedding)
        projected = F.normalize(projected, dim=1)
        return projected

    def infonce_loss(self, anchor, positives, negatives, use_all_positives=False, max_positives=8):
        if positives is None or positives.size(0) == 0:
            return torch.tensor(0.0, device=anchor.device)

        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)

        if use_all_positives and positives.size(0) > max_positives:
            perm = torch.randperm(positives.size(0), device=positives.device)
            positives = positives[perm[:max_positives]]

        if use_all_positives and positives.size(0) > 1:
            losses = []
            for i in range(positives.size(0)):
                positive = positives[i:i+1]
                pos_sim = torch.mm(anchor, positive.t()) / self.temperature

                if negatives is not None and negatives.size(0) > 0:
                    neg_sim = torch.mm(anchor, negatives.t()) / self.temperature
                    logits = torch.cat([pos_sim, neg_sim], dim=1)
                else:
                    logits = pos_sim

                labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)

                if negatives is None or negatives.size(0) == 0:
                    loss = -pos_sim.mean()
                else:
                    loss = F.cross_entropy(logits, labels)
                losses.append(loss)
            return torch.stack(losses).mean()
        else:
            if positives.size(0) > 1:
                pos_idx = torch.randint(0, positives.size(0), (1,), device=positives.device)
                positive = positives[pos_idx]
            else:
                positive = positives

            pos_sim = torch.mm(anchor, positive.t()) / self.temperature

            if negatives is not None and negatives.size(0) > 0:
                neg_sim = torch.mm(anchor, negatives.t()) / self.temperature
                logits = torch.cat([pos_sim, neg_sim], dim=1)
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
                loss = F.cross_entropy(logits, labels)
            else:
                loss = -pos_sim.mean()
            return loss

    def contrastive_loss(self, anchor, positives, negatives):
        return self.infonce_loss(anchor, positives, negatives, use_all_positives=True)


class InstanceBranch(nn.Module):
    def __init__(self, feature_dim, hidden_dim=512, num_classes=2, dropout=0.5):
        super(InstanceBranch, self).__init__()

        self.instance_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.instance_attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.k_ratio = 0.1

    def forward(self, patch_features, labels=None):
        N = patch_features.size(0)
        k = max(1, int(N * self.k_ratio))

        attn_scores = self.instance_attention(patch_features).squeeze(-1)
        _, top_indices = torch.topk(attn_scores, k)
        _, bottom_indices = torch.topk(attn_scores, k, largest=False)

        high_attn_features = patch_features[top_indices]
        low_attn_features = patch_features[bottom_indices]

        instance_loss = torch.tensor(0.0, device=patch_features.device)

        if labels is not None and self.training:
            high_logits = self.instance_classifier(high_attn_features)
            high_labels = torch.full((k,), labels, dtype=torch.long, device=patch_features.device)
            high_loss = F.cross_entropy(high_logits, high_labels)
            instance_loss = high_loss

        return instance_loss, high_attn_features, low_attn_features


class MainAttentionBranch(nn.Module):
    def __init__(self, feature_dim, subtype_dim=4, subtype_embed_dim=256,
                 dropout=0.5, temperature=1.0):
        super(MainAttentionBranch, self).__init__()
        self.temperature = temperature

        self.subtype_encoder = nn.Sequential(
            nn.Linear(subtype_dim, subtype_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(subtype_embed_dim, feature_dim)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Sigmoid()
        )
        self.attention_W = nn.Linear(feature_dim // 2, 1)

        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_features, subtype_onehot):
        N = patch_features.size(0)

        subtype_embedding = self.subtype_encoder(subtype_onehot)
        subtype_expanded = subtype_embedding.unsqueeze(0).expand(N, -1)

        concat_features = torch.cat([patch_features, subtype_expanded], dim=1)
        gate = self.fusion_gate(concat_features)
        fused_features = gate * patch_features + (1 - gate) * subtype_expanded

        V = self.attention_V(fused_features)
        U = self.attention_U(fused_features)
        attn_scores = self.attention_W(V * U)

        attn_weights = F.softmax(attn_scores / self.temperature, dim=0)
        attn_weights = self.dropout(attn_weights)

        aggregated = torch.sum(patch_features * attn_weights, dim=0, keepdim=True)

        return aggregated, attn_weights.squeeze()


class CHIEFAttentionModule(nn.Module):
    def __init__(self, feature_dim, subtype_dim=4, subtype_embed_dim=256,
                 proj_dim=256, num_classes=2, dropout=0.5, temperature=1.0,
                 contrast_temperature=0.07, use_contrastive=True,
                 use_instance_branch=True):
        super(CHIEFAttentionModule, self).__init__()

        self.use_contrastive = use_contrastive
        self.use_instance_branch = use_instance_branch

        self.main_attention = MainAttentionBranch(
            feature_dim=feature_dim,
            subtype_dim=subtype_dim,
            subtype_embed_dim=subtype_embed_dim,
            dropout=dropout,
            temperature=temperature
        )

        if use_instance_branch:
            self.instance_branch = InstanceBranch(
                feature_dim=feature_dim,
                hidden_dim=feature_dim // 2,
                num_classes=num_classes,
                dropout=dropout
            )

        if use_contrastive:
            self.contrastive_branch = WSIContrastiveBranch(
                feature_dim=feature_dim,
                proj_dim=proj_dim,
                temperature=contrast_temperature
            )

    def forward(self, patch_features, subtype_onehot, label=None,
                memory_bank=None, return_contrastive=True):
        wsi_embedding, attn_weights = self.main_attention(patch_features, subtype_onehot)

        instance_loss = torch.tensor(0.0, device=patch_features.device)
        if self.use_instance_branch and self.training and label is not None:
            instance_loss, _, _ = self.instance_branch(patch_features, label)

        contrastive_loss = torch.tensor(0.0, device=patch_features.device)
        if self.use_contrastive and memory_bank is not None and self.training and label is not None:
            subtype_idx = torch.argmax(subtype_onehot).item()

            # [FIX 5] Stop-gradient：在 wsi_embedding 进入对比投影头之前 detach。
            # 原因分析：
            #   - Warmup（epoch 1-9）：bank 未就绪，对比损失=0，backbone 仅由分类/实例损失优化，
            #     在 epoch 9 时集成 AUC 已达 0.7672。
            #   - Warmup 结束（epoch 10+）：bank 就绪，对比损失从随机投影器输出，
            #     值接近理论最大值 log(N_neg+1)≈6.8，cont_scale 线性从 0 渐增。
            #   - 若无 stop-gradient，哪怕 cont_scale=0.1 时，
            #     6.8 * 0.1 * 0.05 = 0.034 的损失也会通过
            #     wsi_embedding → attention → GCN 反传随机梯度，
            #     直接破坏已收敛的图像分支表示，导致集成 AUC 从 0.7672 跌至 0.64 以下。
            #   - Detach 后：对比损失仅通过 contrastive_branch.projector 反传，
            #     projector 独立学习判别性投影空间，backbone 不受干扰。
            #   - 代价：backbone 不能从对比监督中受益，但在本任务中
            #     保护已学到的分类表示远比对比头的额外增益更重要。
            wsi_projected = self.contrastive_branch(wsi_embedding.detach())

            # [FIX 1] get_positives 内部已包含冷启动保护，
            # 若 bank 未就绪则返回 None，此处 if 判断自然跳过对比损失，
            # 不再向图像分支传递任何噪声梯度。
            positives = memory_bank.get_positives(subtype_idx, label)
            negatives = memory_bank.get_negatives(subtype_idx, label)

            if positives is not None and positives.size(0) > 0:
                with torch.no_grad():
                    positives_proj = self.contrastive_branch(positives)
                    negatives_proj = self.contrastive_branch(negatives) if negatives is not None else None

                contrastive_loss = self.contrastive_branch.infonce_loss(
                    wsi_projected, positives_proj, negatives_proj, use_all_positives=True
                )

            # [FIX 1] 无论对比损失是否计算，都更新 bank（积累真实特征，推进就绪进度）
            memory_bank.update(wsi_embedding, subtype_idx, label, is_training=True)

        return wsi_embedding, attn_weights, instance_loss, contrastive_loss


class SpatialGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(SpatialGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels if _ < num_layers - 1 else out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class CHIEFScaleBranch(nn.Module):
    def __init__(self, in_dim, gcn_hidden, gcn_out, subtype_dim, subtype_embed_dim,
                 num_cls, k_neighbors=8, gcn_layers=2, dropout=0.5, temperature=1.0,
                 use_contrastive=True, use_instance_branch=True,
                 contrast_temperature=0.07, proj_dim=256):
        super(CHIEFScaleBranch, self).__init__()
        self.k_neighbors = k_neighbors
        self.gcn_out = gcn_out

        self.dim_reduction = nn.Sequential(
            nn.Linear(in_dim, gcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gcn = SpatialGCN(gcn_hidden, gcn_hidden, gcn_out,
                              num_layers=gcn_layers, dropout=dropout)

        self.chief_attention = CHIEFAttentionModule(
            feature_dim=gcn_out,
            subtype_dim=subtype_dim,
            subtype_embed_dim=subtype_embed_dim,
            proj_dim=proj_dim,
            num_classes=num_cls,
            dropout=dropout,
            temperature=temperature,
            contrast_temperature=contrast_temperature,
            use_contrastive=use_contrastive,
            use_instance_branch=use_instance_branch
        )

        self.subtype_embed = nn.Linear(subtype_dim, subtype_embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(gcn_out + subtype_embed_dim, gcn_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_out, num_cls)
        )

    def forward(self, features, edge_index, subtype_features, label=None,
                memory_bank=None):
        x = self.dim_reduction(features)
        x = self.gcn(x, edge_index)

        aggregated, attn_weights, instance_loss, contrastive_loss = self.chief_attention(
            x, subtype_features, label=label, memory_bank=memory_bank
        )

        subtype_embedded = self.subtype_embed(subtype_features)

        wsi_vec = aggregated.squeeze(0)
        concat_features = torch.cat([wsi_vec, subtype_embedded], dim=0)

        logits = self.classifier(concat_features.unsqueeze(0)).squeeze(0)

        return logits, aggregated, subtype_embedded, attn_weights, instance_loss, contrastive_loss


class ClinicalBranch(nn.Module):

    def __init__(self, clinical_dim, hidden_dim, num_cls, dropout=0.3):
        super(ClinicalBranch, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_cls)
        )

    def forward(self, clinical_features):
        if clinical_features.dim() == 2:
            clinical_features = clinical_features.squeeze(0)

        embedding = self.feature_extractor(clinical_features)
        logits = self.classifier(embedding)

        return logits, embedding


class DynamicAttentionFuser(nn.Module):
    def __init__(self, num_branches, num_classes, hidden_dim=128, momentum=0.9,
                 weight_ema=0.5,
                 mask_threshold=0.64,
                 mask_window=5,
                 min_active=2,
                 mask_weight=0.05):
        super(DynamicAttentionFuser, self).__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        self.momentum = momentum
        self.weight_ema = weight_ema
        self.mask_threshold = mask_threshold
        self.mask_window = mask_window
        self.min_active = min_active
        self.mask_weight = mask_weight

        self.input_norm = nn.LayerNorm(num_classes * num_branches)
        self.attention_net = nn.Sequential(
            nn.Linear(num_classes * num_branches, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_branches),
            nn.Softmax(dim=-1)
        )

        self.register_buffer('performance_history', torch.full((num_branches,), 0.5))
        self.register_buffer('ema_weights', torch.ones(num_branches) / num_branches)

        _MAX_HISTORY = 20
        self.register_buffer('auc_history',
                             torch.full((num_branches, _MAX_HISTORY), -1.0))
        self.register_buffer('auc_history_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('branch_active', torch.ones(num_branches))

    def forward(self, logits_list):
        probs_list = [F.softmax(l.detach() if not l.requires_grad else l, dim=0)
                      for l in logits_list]
        probs_concat = torch.cat(probs_list, dim=0)

        normed = self.input_norm(probs_concat)
        raw_weights = self.attention_net(normed)

        with torch.no_grad():
            perf = self.performance_history.to(raw_weights.device)
            perf_min = perf.min()
            perf_max = perf.max()
            if perf_max - perf_min > 1e-6:
                perf_norm = (perf - perf_min) / (perf_max - perf_min + 1e-8)
            else:
                perf_norm = torch.ones_like(perf) / self.num_branches
            perf_bias = perf_norm + 0.1

        biased_weights = raw_weights * perf_bias
        biased_weights = biased_weights / (biased_weights.sum() + 1e-8)

        if self.training:
            with torch.no_grad():
                self.ema_weights = (self.weight_ema * self.ema_weights
                                    + (1 - self.weight_ema) * biased_weights.detach())
            attention_weights = biased_weights
        else:
            blended = (self.weight_ema * self.ema_weights
                       + (1 - self.weight_ema) * biased_weights)
            attention_weights = blended / (blended.sum() + 1e-8)

        weighted_logits = torch.zeros(self.num_classes, device=probs_concat.device)

        if (self.branch_active < 0.5).any():
            masked_attn = attention_weights * self.branch_active
            denom = masked_attn.sum()
            if denom > 1e-8:
                attention_weights = masked_attn / denom

        for i, logits in enumerate(logits_list):
            weighted_logits += attention_weights[i] * logits

        probs = [F.softmax(logits, dim=0) for logits in logits_list]

        return weighted_logits, probs, attention_weights

    def update_performance_history(self, branch_aucs):
        if isinstance(branch_aucs, (list, tuple)):
            branch_aucs = torch.tensor(branch_aucs, device=self.performance_history.device)

        if self.performance_history.std() < 1e-6:
            self.performance_history = branch_aucs.clone().float()
        else:
            self.performance_history = (self.momentum * self.performance_history +
                                        (1 - self.momentum) * branch_aucs)

    def get_current_weights(self):
        return self.performance_history / self.performance_history.sum()

    def get_weight_entropy(self):
        weights = self.get_current_weights()
        entropy = -(weights * torch.log(weights + 1e-10)).sum()
        return entropy.item()

    def update_branch_mask(self, branch_aucs, threshold=None, window=None, min_active=None):

        if threshold is None:
            threshold = self.mask_threshold
        if window is None:
            window = self.mask_window
        if min_active is None:
            min_active = self.min_active

        if isinstance(branch_aucs, (list, tuple)):
            branch_aucs = torch.tensor(branch_aucs, dtype=torch.float32,
                                       device=self.branch_active.device)

        n = branch_aucs.size(0)
        if n != self.num_branches:
            return  # Dimension mismatch, skipping.

        # Write to circular buffer
        max_hist = self.auc_history.size(1)
        ptr = int(self.auc_history_ptr.item()) % max_hist
        self.auc_history[:n, ptr] = branch_aucs
        self.auc_history_ptr += 1

        n_filled = min(int(self.auc_history_ptr.item()), max_hist)
        if n_filled < window:
            return

        total_ptr = int(self.auc_history_ptr.item())
        indices = [(total_ptr - 1 - i) % max_hist for i in range(window)]
        recent = self.auc_history[:, indices]       # (num_branches, window)
        valid_mask = recent >= 0.0
        mean_aucs = (recent * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1).clamp(min=1))

        new_mask = torch.where(
            mean_aucs >= threshold,
            torch.ones(self.num_branches, device=mean_aucs.device),
            torch.full((self.num_branches,), self.mask_weight, device=mean_aucs.device)
        )

        active_cnt = int((new_mask > 0.5).sum().item())
        if active_cnt < min_active:
            _, rank = mean_aucs.sort(descending=True)
            restored = 0
            for idx in rank.tolist():
                if new_mask[idx] < 0.5:
                    new_mask[idx] = 0.5
                    restored += 1
                    if active_cnt + restored >= min_active:
                        break

        prev = self.branch_active.clone()
        self.branch_active.copy_(new_mask)
        if not torch.allclose(prev, new_mask, atol=0.01):
            auc_strs = ", ".join(f"{v:.3f}" for v in mean_aucs.tolist())
            mask_strs = ", ".join(f"{v:.2f}" for v in new_mask.tolist())
            print(f"\n[BranchMask] Rolling AUC ({window}-ep mean): [{auc_strs}]")
            print(f"[BranchMask] New mask: [{mask_strs}]  (threshold={threshold:.2f})")


class CHIEF_MultiScale_Model(nn.Module):
    def __init__(self, in_dim, gcn_hidden, gcn_out, num_cls, num_subtypes,
                 magnifications, k_neighbors=8, gcn_layers=2, dropout=0.5,
                 temperature=1.0, subtype_embed_dim=256,
                 ensemble_method='attention', scale_weights=None,
                 ensemble_hidden_dim=128, adaptive_weight_init=False,
                 initial_scale_aucs=None,
                 use_contrastive=True, use_instance_branch=True,
                 contrast_temperature=0.07, proj_dim=256, memory_bank_size=128,
                 use_clinical=False, clinical_dim=None,
                 clinical_hidden=256, clinical_dropout=0.3,
                 fusion_method='attention',
                 memory_bank_warmup_epochs=10,
                 memory_bank_min_filled_ratio=0.1,
                 memory_bank_momentum=0.9,
                 detach_branch_for_ensemble=True,
                 scale_configs=None):
        super(CHIEF_MultiScale_Model, self).__init__()

        self.num_cls = num_cls
        self.num_subtypes = num_subtypes
        self.magnifications = magnifications
        self.num_scales = len(magnifications)
        self.ensemble_method = ensemble_method
        self.use_contrastive = use_contrastive
        self.use_instance_branch = use_instance_branch
        self.use_clinical = use_clinical
        self.fusion_method = fusion_method
        self.detach_branch_for_ensemble = detach_branch_for_ensemble
        self.scale_configs = scale_configs or {}

        self.scale_gcn_out = {}

        self.scale_branches = nn.ModuleDict()
        for mag in magnifications:
            sc = self.scale_configs.get(mag, {})
            mag_gcn_hidden    = sc.get('gcn_hidden',   gcn_hidden)
            mag_gcn_out       = sc.get('gcn_out',      gcn_out)
            mag_k_neighbors   = sc.get('k_neighbors',  k_neighbors)
            mag_gcn_layers    = sc.get('gcn_layers',   gcn_layers)
            mag_dropout       = sc.get('dropout',      dropout)
            mag_temperature   = sc.get('temperature',  temperature)

            self.scale_gcn_out[mag] = mag_gcn_out

            self.scale_branches[mag] = CHIEFScaleBranch(
                in_dim=in_dim,
                gcn_hidden=mag_gcn_hidden,
                gcn_out=mag_gcn_out,
                subtype_dim=num_subtypes,
                subtype_embed_dim=subtype_embed_dim,
                num_cls=num_cls,
                k_neighbors=mag_k_neighbors,
                gcn_layers=mag_gcn_layers,
                dropout=mag_dropout,
                temperature=mag_temperature,
                use_contrastive=use_contrastive,
                use_instance_branch=use_instance_branch,
                contrast_temperature=contrast_temperature,
                proj_dim=proj_dim
            )

        self.memory_banks = nn.ModuleDict()
        if use_contrastive:
            for mag in magnifications:
                self.memory_banks[mag] = MemoryBank(
                    feature_dim=self.scale_gcn_out[mag],
                    bank_size=memory_bank_size,
                    num_subtypes=num_subtypes,
                    momentum=memory_bank_momentum,   # [FIX 7] was hardcoded 0.999
                    warmup_epochs=memory_bank_warmup_epochs,
                    min_filled_ratio=memory_bank_min_filled_ratio
                )

        self.clinical_branch = None
        self.concat_classifier = None
        if use_clinical and clinical_dim is not None:
            self.clinical_branch = ClinicalBranch(
                clinical_dim=clinical_dim,
                hidden_dim=clinical_hidden,
                num_cls=num_cls,
                dropout=clinical_dropout
            )
            if fusion_method == 'concatenation':
                unique_outs = set(self.scale_gcn_out.values())
                if len(unique_outs) > 1:
                    raise ValueError(
                        f"fusion_method='concatenation' requires identical gcn_out across "
                        f"all scales (got {dict(self.scale_gcn_out)}). "
                        f"Use fusion_method='attention' when scales have different gcn_out."
                    )
                concat_gcn_out = next(iter(unique_outs))
                concat_input_dim = concat_gcn_out + subtype_embed_dim + clinical_hidden
                self.concat_classifier = nn.Sequential(
                    nn.Linear(concat_input_dim, concat_gcn_out),
                    nn.ReLU(),
                    nn.Dropout(clinical_dropout),
                    nn.Linear(concat_gcn_out, num_cls)
                )
                self.num_total_branches = self.num_scales  # clinical已融合，不参与ensemble
                self.branch_names = list(magnifications)
            else:
                self.num_total_branches = self.num_scales + 1
                self.branch_names = list(magnifications) + ['clinical']
        else:
            self.num_total_branches = self.num_scales
            self.branch_names = list(magnifications)

        self.ensemble_fuser = DynamicAttentionFuser(
            num_branches=self.num_total_branches,
            num_classes=num_cls,
            hidden_dim=ensemble_hidden_dim,
            momentum=0.9
        )

    def set_memory_bank_epoch(self, epoch: int):

        for mag, bank in self.memory_banks.items():
            bank.set_epoch(epoch)

    def get_memory_bank_stats(self):

        all_stats = {}
        for mag, bank in self.memory_banks.items():
            all_stats[mag] = bank.get_bank_stats()
        return all_stats

    def update_branch_mask(self, branch_aucs_dict, threshold=0.58, window=20, min_active=3):

        aucs = [branch_aucs_dict.get(b, 0.5) for b in self.branch_names]
        if hasattr(self.ensemble_fuser, 'update_branch_mask'):
            self.ensemble_fuser.update_branch_mask(
                aucs, threshold=threshold, window=window, min_active=min_active
            )

    def forward(self, multiscale_features, multiscale_edge_indices, subtype_onehot,
                clinical_features=None, label=None):

        branch_logits = {}
        branch_probs = {}
        multiscale_attn_weights = {}
        logits_list = []

        total_instance_loss = torch.tensor(0.0, device=subtype_onehot.device)
        total_contrastive_loss = torch.tensor(0.0, device=subtype_onehot.device)

        for mag in self.magnifications:
            if mag in multiscale_features and mag in multiscale_edge_indices:
                features = multiscale_features[mag]
                edge_index = multiscale_edge_indices[mag]

                memory_bank = None
                if self.use_contrastive and mag in self.memory_banks and self.training and label is not None:
                    memory_bank = self.memory_banks[mag]

                logits, wsi_embedding, subtype_embedding, attn_weights, instance_loss, contrastive_loss = \
                    self.scale_branches[mag](
                        features, edge_index, subtype_onehot,
                        label=label, memory_bank=memory_bank
                    )

                branch_logits[mag] = logits
                branch_probs[mag] = F.softmax(logits, dim=0)
                multiscale_attn_weights[mag] = attn_weights
                logits_list.append(logits)

                total_instance_loss = total_instance_loss + instance_loss
                total_contrastive_loss = total_contrastive_loss + contrastive_loss
            else:
                zero_logits = torch.zeros(self.num_cls, device=subtype_onehot.device)
                branch_logits[mag] = zero_logits
                branch_probs[mag] = F.softmax(zero_logits, dim=0)
                multiscale_attn_weights[mag] = None
                logits_list.append(zero_logits)

        if self.use_clinical and self.clinical_branch is not None and clinical_features is not None:
            clinical_logits, clinical_embedding = self.clinical_branch(clinical_features)

            if self.fusion_method == 'concatenation' and self.concat_classifier is not None:

                image_embeddings = []
                subtype_embeds = []
                for mag in self.magnifications:
                    if mag in multiscale_features:
                        feat = multiscale_features[mag]
                        edge = multiscale_edge_indices[mag]
                        branch = self.scale_branches[mag]
                        x = branch.dim_reduction(feat)
                        x = branch.gcn(x, edge)
                        agg, _ = branch.chief_attention.main_attention(x, subtype_onehot)
                        image_embeddings.append(agg.squeeze(0))
                        subtype_embeds.append(branch.subtype_embed(subtype_onehot))

                img_emb = torch.stack(image_embeddings).mean(0)      # (gcn_out,)
                sub_emb = torch.stack(subtype_embeds).mean(0)        # (subtype_embed_dim,)
                concat_vec = torch.cat([img_emb, sub_emb, clinical_embedding], dim=0)
                fused_logits = self.concat_classifier(concat_vec.unsqueeze(0)).squeeze(0)

                logits_list = [fused_logits]
                for mag in self.magnifications:
                    branch_logits[mag] = fused_logits
                    branch_probs[mag] = F.softmax(fused_logits, dim=0)

                branch_logits['clinical'] = clinical_logits
                branch_probs['clinical'] = F.softmax(clinical_logits, dim=0)
                multiscale_attn_weights['clinical'] = None
            else:
                branch_logits['clinical'] = clinical_logits
                branch_probs['clinical'] = F.softmax(clinical_logits, dim=0)
                multiscale_attn_weights['clinical'] = None
                logits_list.append(clinical_logits)
        elif self.use_clinical:
            zero_logits = torch.zeros(self.num_cls, device=subtype_onehot.device)
            branch_logits['clinical'] = zero_logits
            branch_probs['clinical'] = F.softmax(zero_logits, dim=0)
            multiscale_attn_weights['clinical'] = None
            if self.fusion_method != 'concatenation':
                logits_list.append(zero_logits)

        if len(logits_list) > 0:

            if self.detach_branch_for_ensemble and self.training \
                    and self.fusion_method != 'concatenation':
                ensemble_input = [l.detach() for l in logits_list]
            else:
                ensemble_input = logits_list
            final_logits, ensemble_probs, attention_weights = self.ensemble_fuser(ensemble_input)
        else:
            final_logits = torch.zeros(self.num_cls, device=subtype_onehot.device)
            ensemble_probs = [torch.zeros(self.num_cls, device=subtype_onehot.device)] * self.num_total_branches
            attention_weights = torch.ones(self.num_total_branches, device=subtype_onehot.device) / self.num_total_branches

        ensemble_info = {
            'method': self.ensemble_method,
            'scale_contributions': ensemble_probs,
            'branch_names': self.branch_names,
            'attention_weights': attention_weights
        }

        if hasattr(self.ensemble_fuser, 'get_current_weights'):
            ensemble_info['learned_weights'] = self.ensemble_fuser.get_current_weights()

        if hasattr(self.ensemble_fuser, 'get_weight_entropy'):
            ensemble_info['weight_entropy'] = self.ensemble_fuser.get_weight_entropy()

        num_active_image_branches = len([m for m in self.magnifications if m in multiscale_features])
        chief_losses = {
            'instance_loss': total_instance_loss / max(num_active_image_branches, 1),
            'contrastive_loss': total_contrastive_loss / max(num_active_image_branches, 1)
        }

        return final_logits, branch_logits, branch_probs, multiscale_attn_weights, \
               ensemble_info, chief_losses


class CHIEFLoss(nn.Module):
    def __init__(self, class_counts=None, num_classes=2, gamma=2.0, beta=0.9999,
                 smoothing=0.1, instance_weight=0.1, contrastive_weight=0.1):
        super(CHIEFLoss, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.instance_weight = instance_weight
        self.contrastive_weight = contrastive_weight

        if class_counts is not None:
            effective_num = 1.0 - np.power(beta, class_counts)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes

            ratio = weights.max() / weights.min()
            if ratio > 3.0:
                alpha = np.log(3.0) / np.log(ratio)
                weights = np.power(weights, alpha)
                weights = weights / np.sum(weights) * num_classes

            self.class_weights = torch.FloatTensor(weights)
        else:
            self.class_weights = torch.ones(num_classes)

    def forward(self, inputs, targets, chief_losses=None):
        self.class_weights = self.class_weights.to(inputs.device)
        n_classes = inputs.size(-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(inputs, dim=-1)
        weighted_log_probs = log_probs * self.class_weights.unsqueeze(0)
        ce_loss = -(smooth_targets * weighted_log_probs).sum(dim=-1)

        probs = torch.exp(log_probs)
        pt = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        cls_loss = (focal_weight * ce_loss).mean()
        total_loss = cls_loss

        if chief_losses is not None:
            if 'instance_loss' in chief_losses:
                total_loss = total_loss + self.instance_weight * chief_losses['instance_loss']
            if 'contrastive_loss' in chief_losses:
                total_loss = total_loss + self.contrastive_weight * chief_losses['contrastive_loss']

        return total_loss, {
            'cls_loss': cls_loss,
            'instance_loss': chief_losses.get('instance_loss', 0) if chief_losses else 0,
            'contrastive_loss': chief_losses.get('contrastive_loss', 0) if chief_losses else 0
        }


class PC_TMB_MultiScale_Ensemble_Optimized_Model(CHIEF_MultiScale_Model):

    def __init__(self, *args, attention_type='chief', use_subtype_in_classifier=True, **kwargs):
        kwargs.pop('use_contrastive', None)
        kwargs.pop('use_instance_branch', None)

        use_contrastive = (attention_type == 'chief')
        use_instance_branch = (attention_type == 'chief')

        super().__init__(
            *args,
            use_contrastive=use_contrastive,
            use_instance_branch=use_instance_branch,
            **kwargs
        )
        self.attention_type = attention_type

    def forward(self, multiscale_features, multiscale_edge_indices, subtype_onehot,
                clinical_features=None, label=None):
        result = super().forward(
            multiscale_features, multiscale_edge_indices, subtype_onehot,
            clinical_features, label
        )

        final_logits, branch_logits, branch_probs, multiscale_attn_weights, \
            ensemble_info, chief_losses = result

        ensemble_info['chief_losses'] = chief_losses

        return final_logits, branch_logits, branch_probs, multiscale_attn_weights, ensemble_info


ClassBalancedFocalLossWithSmoothing = CHIEFLoss