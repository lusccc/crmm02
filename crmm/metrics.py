import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
from scipy.stats import ks_2samp
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, roc_curve, precision_recall_fscore_support,
    accuracy_score, recall_score, precision_score, f1_score,
)
from transformers import EvalPrediction
from transformers.utils import logging

logger = logging.get_logger('transformers')


def calc_classification_metrics_hf(p: EvalPrediction, save_cm_fig_dir=None):
    if isinstance(p.predictions, tuple):
        pred_labels = np.argmax(p.predictions[0], axis=1)
        pred_scores = softmax(p.predictions[0], axis=1)[:, 1]
    else:
        pred_labels = np.argmax(p.predictions, axis=1)
        pred_scores = softmax(p.predictions, axis=1)[:, 1]
    labels = p.label_ids
    if len(np.unique(labels)) == 2:  # binary classification
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(labels, pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        cm = confusion_matrix(labels, pred_labels)
        fpr, tpr, _ = roc_curve(labels, pred_scores)
        ks = np.max(tpr - fpr)
        # gmean = np.sqrt(recalls[ix] * precisions[ix])  # this is wrong!
        tn, fp, fn, tp = cm.ravel()
        acc = (pred_labels == labels).mean()

        # type1 acc (TN / (TN + FP))
        type1_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        # type2 acc (TP / (TP + FN))
        type2_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        gmean = np.sqrt(type1_acc * type2_acc)  # this is right!

        result = {
            'threshold': threshold,
            'pr_auc': pr_auc,
            'recall': recalls[ix].item(),
            'precision': precisions[ix].item(),
            'f1': fscore[ix].item(),
            'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(), 'tp': tp.item(),
            "acc": acc,
            'roc_auc': roc_auc_pred_score,
            'recall_.5threshold': recall_score(labels, pred_labels),
            'precision_.5threshold': precision_score(labels, pred_labels),
            'f1_.5threshold': f1_score(labels, pred_labels),
            'ks': ks,
            'gmean': gmean,
            'type1_acc': type1_acc,  # 加入type1_acc
            'type2_acc': type2_acc,  # 加入type2_acc
        }
    else:
        acc = (pred_labels == labels).mean()
        precision, recall, f1, support = precision_recall_fscore_support(labels, pred_labels)
        cm = confusion_matrix(labels, pred_labels, )
        result = {
            "acc": acc,
            "f1": str(list(f1)),
            "f1_mean": f1.mean(),
            "mcc": matthews_corrcoef(labels, pred_labels),
            "per_class_recall": str(recall.tolist()),
            "recall_mean": recall.mean(),
            "per_class_precision": str(precision.tolist()),
            "precision_mean": precision.mean(),
            "cm": str(cm.tolist())
        }
    logger.info(f'\n{cm}')
    logger.info(result)
    if save_cm_fig_dir:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot()
        plt.savefig(os.path.join(save_cm_fig_dir, 'cm.png'))

    return result


def calc_classification_metrics_benchmark(model_name, y_true, y_pred, y_pred_prob):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 计算基本指标
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_prob),
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

    # 计算KS统计量
    def calculate_ks_statistic(y_true, y_pred_prob):
        return ks_2samp(y_pred_prob[y_true == 1], y_pred_prob[y_true == 0]).statistic

    metrics['ks'] = calculate_ks_statistic(y_true, y_pred_prob)

    # 计算G-mean
    metrics['gmean'] = np.sqrt(metrics['recall'] * metrics['precision'])

    metrics['type1_acc'] = tn / (tn + fp)
    metrics['type2_acc'] = tp / (tp + fn)
    metrics.update({'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(), 'tp': tp.item()})

    return metrics
