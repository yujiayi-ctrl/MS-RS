from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score, confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import os

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label, fixed_threshold=None):
    if isinstance(label, torch.Tensor):
        label_np = label.cpu().numpy().astype(int)
    else:
        label_np = label.astype(int)
    
    if isinstance(oprob, torch.Tensor):
        oprob_np = oprob.detach().cpu().numpy()
    else:
        oprob_np = oprob
    
    if fixed_threshold is None:
        auc, threshold = roc_threshold(label_np, oprob_np)
    else:
        threshold = fixed_threshold
        try:
            auc = roc_auc_score(label_np, oprob_np)
        except:
            auc = 0.0
    
    if isinstance(oprob, torch.Tensor):
        prob = oprob > threshold
    else:
        prob = oprob_np > threshold
    y_pred = prob.cpu().numpy().astype(int) if isinstance(prob, torch.Tensor) else prob.astype(int)
    
    TP = (y_pred == 1) & (label_np == 1)
    TN = (y_pred == 0) & (label_np == 0)
    FP = (y_pred == 1) & (label_np == 0)
    FN = (y_pred == 0) & (label_np == 1)
    
    accuracy = (TP.sum() + TN.sum()) / (len(label_np) + 1e-12)
    precision = TP.sum() / (TP.sum() + FP.sum() + 1e-12) if (TP.sum() + FP.sum()) > 0 else 0.0
    recall = TP.sum() / (TP.sum() + FN.sum() + 1e-12) if (TP.sum() + FN.sum()) > 0 else 0.0
    specificity = TN.sum() / (TN.sum() + FP.sum() + 1e-12) if (TN.sum() + FP.sum()) > 0 else 0.0
    F1 = 2 * precision * recall / (precision + recall + 1e-12) if (precision + recall) > 0 else 0.0
    
    try:
        kappa = cohen_kappa_score(label_np, y_pred)
    except:
        kappa = 0.0
    tp = TP.sum()
    tn = TN.sum()
    fp = FP.sum()
    fn = FN.sum()
    result_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': F1,
        'auc': auc,
        'kappa': kappa,
        'threshold': threshold,
        'y_pred': y_pred,
        'label_np': label_np,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return result_dict

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Prediction Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return cm, cm_normalized

def plot_multiple_confusion_matrices(cm_list, fold_names, class_names, title, save_path=None):
    n_folds = len(cm_list)
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 5))
    
    if n_folds == 1:
        axes = [axes]
    
    for i, (cm, ax, fold_name) in enumerate(zip(cm_list, axes, fold_names)):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{fold_name}')
        ax.set_ylabel('True Label' if i == 0 else '')
        ax.set_xlabel('Prediction Label')
    
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_score, title, save_path=None, label=None, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc:.3f})' if label else f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    if save_path and ax is None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    elif ax is None:
        plt.show()
    
    return fpr, tpr, auc

def plot_multiple_roc_curves(roc_data_list, labels, title, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for (y_true, y_score), label, color in zip(roc_data_list, labels, colors):
        fpr, tpr, auc = plot_roc_curve(y_true, y_score, title, label=label, ax=ax)
        ax.plot(fpr, tpr, lw=2, color=color, label=f'{label} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()