import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, roc_curve, precision_recall_fscore_support,
)
from transformers import EvalPrediction
from transformers.utils import logging

logger = logging.get_logger('transformers')


def calc_classification_metrics(p: EvalPrediction, save_cm_fig_dir=None):
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
        gmean = np.sqrt(recalls[ix] * precisions[ix])
        tn, fp, fn, tp = cm.ravel()
        acc = (pred_labels == labels).mean()

        # type1 acc (TN / (TN + FP))
        type1_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        # type2 acc (TP / (TP + FN))
        type2_acc = tp / (tp + fn) if (tp + fn) > 0 else 0

        result = {"acc": acc,
                  'roc_auc': roc_auc_pred_score,
                  'threshold': threshold,
                  'pr_auc': pr_auc,
                  'recall': recalls[ix].item(),
                  'precision': precisions[ix].item(),
                  'f1': fscore[ix].item(),
                  'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(), 'tp': tp.item(),
                  'ks': ks,
                  'gmean': gmean,
                  'type1_acc': type1_acc,  # 加入type1_acc
                  'type2_acc': type2_acc,  # 加入type2_acc
                  'cm': str(cm.tolist())}
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
