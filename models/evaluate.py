# -*- coding: utf-8 -*-

import numpy as np
import torch
from .utils import get_device


def acc_std(W, data_test, method):
    """
    Evaluate classification accuracy and standard deviation

    Uses Balanced Accuracy to handle class imbalance.
    Balanced Accuracy = (Recall + Specificity) / 2 = (TPR + TNR) / 2

    Parameters:
    -----------
    W : ndarray
        Weight matrix (n_runs x d), each row is one learned model
    data_test : DataFrame
        Test data with [true_label, ones, features...]
    method : str
        Method name for display

    Returns:
    --------
    acc : float
        Average balanced accuracy across runs
    std : float
        Standard deviation of balanced accuracy
    """
    device = get_device()

    cols = data_test.shape[1]
    X = data_test.iloc[:, 2:cols].values
    y = data_test.iloc[:, 0:1].values.flatten()

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    W_tensor = torch.tensor(W, dtype=torch.float32, device=device)

    accuracies = []
    for t in range(W.shape[0]):
        w_t = W_tensor[t]
        f_scores = torch.mv(X_tensor, w_t)
        predictions = torch.sign(f_scores)
        predictions = torch.where(predictions == 0, torch.ones_like(predictions), predictions)

        y_np = y_tensor.cpu().numpy()
        pred_np = predictions.cpu().numpy()

        TP = np.sum((y_np == 1) & (pred_np == 1))
        TN = np.sum((y_np == -1) & (pred_np == -1))
        FP = np.sum((y_np == -1) & (pred_np == 1))
        FN = np.sum((y_np == 1) & (pred_np == -1))

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        balanced_acc = (recall + specificity) / 2
        accuracies.append(balanced_acc)

    acc = np.mean(accuracies)
    std = np.std(accuracies)

    print(f'Model: {method}')
    print(f'  Accuracy (mean): {acc:.3f}')
    print(f'  Accuracy (std):  {std:.3f}')

    return acc, std


def evaluate_comprehensive(W, data_test, method):
    """
    Compute comprehensive evaluation metrics

    Includes:
    - Accuracy
    - Standard Deviation
    - Precision
    - Recall
    - F1-Score
    - Specificity

    Parameters:
    -----------
    W : ndarray
        Weight matrix (n_runs x d)
    data_test : DataFrame
        Test data [true_label, ones, features...]
    method : str
        Model name

    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics (mean and std)
    """
    device = get_device()

    cols = data_test.shape[1]
    X = data_test.iloc[:, 2:cols].values
    y_true = data_test.iloc[:, 0:1].values.flatten()

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device)
    W_tensor = torch.tensor(W, dtype=torch.float32, device=device)

    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': []
    }

    for t in range(W.shape[0]):
        w_t = W_tensor[t]
        f_scores = torch.mv(X_tensor, w_t)
        y_pred = torch.sign(f_scores)
        y_pred = torch.where(y_pred == 0, torch.ones_like(y_pred), y_pred)

        y_true_np = y_tensor.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        TP = np.sum((y_true_np == 1) & (y_pred_np == 1))
        TN = np.sum((y_true_np == -1) & (y_pred_np == -1))
        FP = np.sum((y_true_np == -1) & (y_pred_np == 1))
        FN = np.sum((y_true_np == 1) & (y_pred_np == -1))

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        all_metrics['accuracy'].append(accuracy)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        all_metrics['precision'].append(precision)

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        all_metrics['recall'].append(recall)

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        all_metrics['f1'].append(f1)

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        all_metrics['specificity'].append(specificity)

    metrics = {}
    for metric_name, values in all_metrics.items():
        metrics[f'{metric_name}_mean'] = np.mean(values)
        metrics[f'{metric_name}_std'] = np.std(values)

    print(f'Model: {method}')
    print(f'  Accuracy    = {metrics["accuracy_mean"]:.3f} +/- {metrics["accuracy_std"]:.3f}')
    print(f'  Precision   = {metrics["precision_mean"]:.3f} +/- {metrics["precision_std"]:.3f}')
    print(f'  Recall      = {metrics["recall_mean"]:.3f} +/- {metrics["recall_std"]:.3f}')
    print(f'  F1-Score    = {metrics["f1_mean"]:.3f} +/- {metrics["f1_std"]:.3f}')
    print(f'  Specificity = {metrics["specificity_mean"]:.3f} +/- {metrics["specificity_std"]:.3f}')

    return metrics


def acc_unlabeled(W, data_test, method):
    """
    Compute the ratio of unlabeled samples classified as positive

    Useful for diagnosing if the model is too conservative/aggressive

    Parameters:
    -----------
    W : ndarray
        Weight matrix (n_runs x d)
    data_test : DataFrame
        Test data
    method : str
        Method name

    Returns:
    --------
    ratio : float
        Average ratio of unlabeled (true negatives) classified as positive
    """
    cols = data_test.shape[1]
    X = data_test.iloc[:, 2:cols].values
    y = data_test.iloc[:, 0:1].values.flatten()

    ratios = []
    for t in range(W.shape[0]):
        w_t = W[t]

        count_pos = 0
        num_unlabeled = 0

        for i in range(len(X)):
            if y[i] == -1:
                num_unlabeled += 1
                pred = np.sign(np.dot(X[i], w_t))
                if pred == 1:
                    count_pos += 1

        if num_unlabeled > 0:
            ratio_t = count_pos / num_unlabeled
        else:
            ratio_t = 0

        ratios.append(ratio_t)

    avg_ratio = np.mean(ratios)
    print(f'Model: {method}')
    print(f'  Unlabeled->Positive ratio: {avg_ratio:.3f}')

    return avg_ratio