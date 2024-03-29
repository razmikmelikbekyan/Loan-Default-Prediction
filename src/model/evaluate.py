from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, average_precision_score, precision_recall_curve
)


def bernoulli_conf_interval(p: float, n: int, confidence: float):
    """
    Calculates confidence interval for n i.i.d. bernoulli(p) random variables.
    It is using CLT: we approximating the average of n i.i.d. bernoulli(p) distributed random
    variables with normal distribution.
        confidence interval: p ± z * (p(1-p) / n)^(1/2)
        alpha = 1 - confidence
        z = 1 - alpha / 2 quantile for standard normal distribution
    :param p: the probability of 1
    :param n: number of i.i.d. bernoulli(p) random variables
    :param confidence: confidence value (0 < confidence < 1)
    :return: tuple, confidence interval
    """
    alpha = 1 - confidence  # target error rate
    z = stats.norm.ppf(1 - alpha / 2)  # 1-alpha/2 - quantile of a standard normal distribution
    se = z * np.sqrt(p * (1 - p) / n)  # standard error
    return p - se, p + se


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculates confusion matrix related metrics and plots confusion matrix.
    :param y_true: true label of the data, with shape (n_samples,), binary array
    :param y_pred: prediction of the data, with shape (n_samples,), binary array
    :return: dict of metrics
    """
    metrics = {}
    cm = confusion_matrix(y_true, y_pred)

    # true negatives (TN): we predicted N, and they don't have the disease (actual N)
    # false positives (FP): we predicted Y, but they don't have the disease (actual N)
    # false negatives (FN): we predicted N, but they do have the disease (actual Y)
    # true positives (TP): we predicted Y and they do have the disease (actual Y)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp})

    # normalzing matrix - getting rates
    cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    # tnr (specificity): probability that a test result will be negative when the disease is not
    # present (tn / actual_N)

    # fpr: probability that a test result will be positive when the disease is not present
    # (fp / actual_N)

    # fnr: probability that a test result will be negative when the disease is present
    # (fn / actual_Y)

    # tpr (recall): probability that a test result will be positive when the disease is present
    # (tp / actual_Y)

    tnr, fpr, fnr, tpr = cm_norm.ravel()
    metrics.update({'tpr': tpr, 'fnr': fnr, 'tnr': tnr, 'fpr': fpr})

    # ppv (precision): probability that the disease is present when the test is positive
    # (tp / (tp + fp))

    # npv: probability that the disease is not present when the test is  negative (tn / (tn + fn))
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    ppv_conf_interval = bernoulli_conf_interval(ppv, tp + fp, 0.95)
    npv_conf_interval = bernoulli_conf_interval(npv, tn + fn, 0.95)
    metrics.update(
        {
            'ppv': ppv,
            'npv': npv,
            'PPV 95% CI': ppv_conf_interval,
            'NPV 95% CI': npv_conf_interval
        }
    )

    # Overall, how often is the classifier correct?
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    metrics.update({'accuracy': accuracy})

    # The weighted average of recall and precision.
    f_score = 2 * tpr * ppv / (tpr + ppv)
    metrics.update({'f_score': f_score, 'cm': cm_norm})

    return metrics


def plot_confusion_matrix(confusion_matrix_metrics: Dict,
                          title: str = '',
                          class_labels: List[str] = ('N', 'Y'),
                          figsize: Tuple = (5, 5),
                          fsize: int = 12,
                          output_path: str = None):
    """
    Plots confusion matrix.

    :param confusion_matrix_metrics: the output of "calculate_confusion_matrix" function
    :param title: title
    :param class_labels: list of labels, first should be the name of negative classes
    :param figsize: the tuple, specifying figure size of plot
    :param fsize: font size for plot
    :param output_path: the path were image will be saved
    """

    cm_norm = confusion_matrix_metrics['cm']

    tn = confusion_matrix_metrics['tn']
    tnr = confusion_matrix_metrics['tnr']
    npv = confusion_matrix_metrics['npv']

    fp = confusion_matrix_metrics['fp']
    fpr = confusion_matrix_metrics['fpr']

    fn = confusion_matrix_metrics['fn']
    fnr = confusion_matrix_metrics['fnr']

    tp = confusion_matrix_metrics['tp']
    tpr = confusion_matrix_metrics['tpr']
    ppv = confusion_matrix_metrics['ppv']

    accuracy = confusion_matrix_metrics['accuracy']
    f_score = confusion_matrix_metrics['f_score']

    # annotation for heatmap
    annot = np.empty_like(cm_norm).astype(str)
    annot[0, 0] = f'TN={tn}\n\nTNR={tnr * 100 :.0f}%\n\nNPV={npv * 100 :.0f}%'
    annot[0, 1] = f'FP={fp} \n\nFPR={fpr * 100 :.0f}%'
    annot[1, 0] = f'FN={fn} \n\nFNR={fnr * 100 :.0f}%'
    annot[1, 1] = f'TP={tp} \n\nTPR={tpr * 100 :.0f}% \n\nPPV={ppv * 100 :.0f}%'

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        pd.DataFrame(cm_norm, index=class_labels, columns=class_labels),
        annot=annot,
        annot_kws={"size": fsize, 'color': 'w', 'fontstyle': 'oblique'},
        linewidths=0.1,
        ax=ax,
        cbar=False,
        linecolor='w',
        fmt=''
    )

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    facecolors[0] = [0.35, 0.8, 0.55, 1.0]  # green
    facecolors[3] = [0.35, 0.8, 0.55, 1.0]  # green
    facecolors[1] = [0.65, 0.1, 0.1, 1.0]  # red
    facecolors[2] = [0.65, 0.1, 0.1, 1.0]  # red

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=12)

    # set labels
    ax.axes.set_title(
        f"{title} \n Accuracy={accuracy * 100 :.2f}%, f_score={f_score :.2f}",
        fontsize=fsize
    )
    ax.set_xlabel("Predicted label", fontsize=15)
    ax.set_ylabel("Actual label", fontsize=15)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    plt.show()


def calculate_roc_curve_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """
    Calculates ROC curve related metrics: AUC and fpr's and tpr's for thresholds.
    :param y_true: true label of the data, with shape (n_samples,), binary array
    :param y_score: target scores, can either be probability estimates of the positive class,
                    confidence values, or non-thresholded measure of decisions
    :return: dict of metrics
    """
    fprs, tprs, _ = roc_curve(y_true, y_score)
    auc_score = auc(fprs, tprs)
    return {'fprs': fprs, 'tprs': tprs, 'auc': auc_score}


def plot_roc_curve(roc_curve_metrics: Dict,
                   title: str = None,
                   output_path: str = None):
    """
    Plots RUC curve.
    :param roc_curve_metrics: the output of "calculate_roc_curve_metrics" function
    :param title: the name of model
    :param output_path: the path were image will be saved
    """
    plt.figure(figsize=(10, 10))

    temp = 'ROC (AUC = {:.2f})'.format(roc_curve_metrics['auc'])
    title = '{}: {}'.format(title, temp) if title else temp

    plt.plot(roc_curve_metrics['fprs'], roc_curve_metrics['tprs'], lw=3, alpha=0.3, label=title)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC curve', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=18)

    if output_path:
        plt.savefig(output_path)

    plt.show()


def calculate_ppv_tpr_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """
    Calculates precision-recall curve related metrics: average_precision_score and
    precision's and recall's for thresholds.
    :param y_true: true label of the data, with shape (n_samples,), binary array
    :param y_score: target scores, can either be probability estimates of the positive class,
                    confidence values, or non-thresholded measure of decisions
    :return:dict of metrics
    """
    average_ppv = average_precision_score(y_true, y_score)
    ppvs, tprs, _ = precision_recall_curve(y_true, y_score)
    return {'average_ppv': average_ppv, 'precisions': ppvs, 'recalls': tprs}


def plot_ppv_tpr_curve(ppv_tpr_metric: Dict,
                       label: str = None,
                       image_path: str = None):
    """
    Plots Precision-Recall curve.
    :param ppv_tpr_metric: the output of "calculate_ppv_tpr_metrics" function
    :param label: the name of line
    :param image_path: the path were image will be saved
    """
    ppvs = ppv_tpr_metric['precisions']
    tprs = ppv_tpr_metric['recalls']
    average_ppv = ppv_tpr_metric['average_ppv']

    plt.figure(figsize=(10, 10))

    plt.step(tprs, ppvs, color='b', alpha=0.2, where='post', label=label)
    plt.fill_between(tprs, ppvs, step='post', alpha=0.2, color='b')

    plt.xlabel('TPR', fontsize=14)
    plt.ylabel('PPV', fontsize=14)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('PPV-TPR curve: Average PPV={0:0.2f}'.format(average_ppv), fontsize=14)
    plt.legend(loc="lower right", fontsize=14)

    if image_path:
        plt.savefig(image_path)

    plt.show()
