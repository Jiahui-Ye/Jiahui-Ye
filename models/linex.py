# -*- coding: utf-8 -*-

import numpy as np
import torch
from .utils import get_device

EPS = 1e-8

def _compute_linex_grad(f, y, a, pu_weights):
    """
    Paper LINEX gradient
    L = e^{a(z-f)} - a(z-f) - 1
    ∇_f L = - (1/a) * (exp(a*(z-f)) - 1)
    where z = y (+1 or -1)
    """
    exponent = a * (y - f)
    exp_term = torch.exp(torch.clamp(exponent, -50, 50))
    grad_f = -(1.0 / a) * (exp_term - 1)
    return pu_weights * grad_f

def _compute_blinex_grad(f, y, a, b, lam, pu_weights):
    """
    Paper BLINEX gradient (Eq.11)
    L = (1/lam) * [1 - 1/(1 + b*(exp(a*z*f)-a*z*f-1))]
    ∇_f L = a * b * z * (exp(a*z*f)-1) / [lam * (1 + b*(exp(a*z*f)-a*z*f-1))^2]
    """
    t = a * y * f
    exp_t = torch.exp(torch.clamp(t, -50, 50))
    numerator = a * b * y * (exp_t - 1)
    denominator = lam * (1.0 + b * (exp_t - t - 1) + EPS) ** 2
    grad_f = numerator / denominator
    return pu_weights * grad_f

def OPU_LINEX(data_train, pi, alpha, gamma, reg, a_linex,
              pos_weight_factor=2.5, unlabeled_weight_factor=1,
              n_epochs=1, batch_size=512, dataset_name='', seed=42):
    """
    Online PU Learning with LINEX Loss (paper version)
    """
    device = get_device()
    cols = data_train.shape[1]
    X = torch.tensor(np.asarray(data_train.iloc[:, 3:cols], dtype=np.float32), device=device)
    y = torch.tensor(np.asarray(data_train.iloc[:, 0:1].values.flatten(), dtype=np.float32), device=device)
    n_samples = len(X)

    torch.manual_seed(seed)
    w = torch.randn(X.shape[1], dtype=torch.float32, device=device) * 0.01

    use_balanced_sampling = ('YelpZip' in dataset_name) or ('dianping' in dataset_name)

    if a_linex < 0:
        asymmetry_compensation = 1.0 + 0.18 * abs(a_linex)
    else:
        asymmetry_compensation = 1.0

    if use_balanced_sampling:
        pos_indices = torch.where(y == 1)[0]
        unlabeled_indices = torch.where(y == -1)[0]
        n_pos = len(pos_indices)
        n_unlabeled = len(unlabeled_indices)
        half_batch = batch_size // 2

        pos_weight = torch.tensor(pos_weight_factor * asymmetry_compensation * pi / gamma,
                                  dtype=torch.float32, device=device)
        neg_weight = torch.tensor(1.0 / (1 - gamma), dtype=torch.float32, device=device)
        n_batches = max(n_pos, n_unlabeled) // half_batch
        if n_batches == 0:
            n_batches = 1

        for _ in range(n_epochs):
            for _ in range(n_batches):
                pos_sample = pos_indices[torch.randint(0, n_pos, (half_batch,), device=device)]
                if n_unlabeled >= half_batch:
                    unlabeled_sample = unlabeled_indices[torch.randperm(n_unlabeled, device=device)[:half_batch]]
                else:
                    unlabeled_sample = unlabeled_indices[torch.randint(0, n_unlabeled, (half_batch,), device=device)]
                batch_idx = torch.cat([pos_sample, unlabeled_sample])
                batch_idx = batch_idx[torch.randperm(len(batch_idx), device=device)]

                X_b = X[batch_idx]
                y_b = y[batch_idx]
                f = torch.mv(X_b, w)

                pu_weights = torch.where(y_b == 1, pos_weight, neg_weight)
                grad_coeffs = _compute_linex_grad(f, y_b, a_linex, pu_weights)
                gradient = torch.mv(X_b.T, grad_coeffs) / len(batch_idx)
                w = w - alpha * gradient
    else:
        pos_weight = torch.tensor(pos_weight_factor * asymmetry_compensation * (2.0 * pi) / gamma,
                                  dtype=torch.float32, device=device)
        neg_weight = torch.tensor(unlabeled_weight_factor / (1 - gamma), dtype=torch.float32, device=device)
        for _ in range(n_epochs):
            indices = torch.randperm(n_samples, device=device)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                X_b = X[batch_idx]
                y_b = y[batch_idx]
                f = torch.mv(X_b, w)
                pu_weights = torch.where(y_b == 1, pos_weight, neg_weight)
                grad_coeffs = _compute_linex_grad(f, y_b, a_linex, pu_weights)
                gradient = torch.mv(X_b.T, grad_coeffs) / len(batch_idx)
                w = w - alpha * gradient

    return w.cpu().numpy()


def OPU_BLINEX(data_train, pi, alpha, gamma, reg, a_blinex, b, lambda_param,
               pos_weight_factor=2.3, unlabeled_weight_factor=1,
               n_epochs=1, batch_size=512, dataset_name='', seed=42):
    """
    Online PU Learning with BLINEX Loss (paper version)
    """
    device = get_device()
    cols = data_train.shape[1]
    X = torch.tensor(np.asarray(data_train.iloc[:, 3:cols], dtype=np.float32), device=device)
    y = torch.tensor(np.asarray(data_train.iloc[:, 0:1].values.flatten(), dtype=np.float32), device=device)
    n_samples = len(X)

    torch.manual_seed(seed)
    w = torch.randn(X.shape[1], dtype=torch.float32, device=device) * 0.01

    if a_blinex < 0:
        asymmetry_compensation = 1.0 + 0.42 * abs(a_blinex)
    else:
        asymmetry_compensation = 1.0

    use_balanced_sampling = ('YelpZip' in dataset_name) or ('dianping' in dataset_name)

    if use_balanced_sampling:
        pos_indices = torch.where(y == 1)[0]
        unlabeled_indices = torch.where(y == -1)[0]
        n_pos = len(pos_indices)
        n_unlabeled = len(unlabeled_indices)
        half_batch = batch_size // 2

        pos_weight = torch.tensor(pos_weight_factor * asymmetry_compensation * pi / gamma,
                                  dtype=torch.float32, device=device)
        neg_weight = torch.tensor(1.0 / (1 - gamma), dtype=torch.float32, device=device)
        n_batches = max(n_pos, n_unlabeled) // half_batch
        if n_batches == 0:
            n_batches = 1

        for _ in range(n_epochs):
            for _ in range(n_batches):
                pos_sample = pos_indices[torch.randint(0, n_pos, (half_batch,), device=device)]
                if n_unlabeled >= half_batch:
                    unlabeled_sample = unlabeled_indices[torch.randperm(n_unlabeled, device=device)[:half_batch]]
                else:
                    unlabeled_sample = unlabeled_indices[torch.randint(0, n_unlabeled, (half_batch,), device=device)]
                batch_idx = torch.cat([pos_sample, unlabeled_sample])
                batch_idx = batch_idx[torch.randperm(len(batch_idx), device=device)]

                X_b = X[batch_idx]
                y_b = y[batch_idx]
                f = torch.mv(X_b, w)

                pu_weights = torch.where(y_b == 1, pos_weight, neg_weight)
                grad_coeffs = _compute_blinex_grad(f, y_b, a_blinex, b, lambda_param, pu_weights)
                gradient = torch.mv(X_b.T, grad_coeffs) / len(batch_idx)
                gradient = gradient + (reg / len(batch_idx)) * w
                w = w - alpha * gradient
    else:
        pos_weight = torch.tensor(pos_weight_factor * asymmetry_compensation * (2.0 * pi) / gamma,
                                  dtype=torch.float32, device=device)
        neg_weight = torch.tensor(unlabeled_weight_factor / (1 - gamma), dtype=torch.float32, device=device)
        for _ in range(n_epochs):
            indices = torch.randperm(n_samples, device=device)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                X_b = X[batch_idx]
                y_b = y[batch_idx]
                f = torch.mv(X_b, w)
                pu_weights = torch.where(y_b == 1, pos_weight, neg_weight)
                grad_coeffs = _compute_blinex_grad(f, y_b, a_blinex, b, lambda_param, pu_weights)
                gradient = torch.mv(X_b.T, grad_coeffs) / len(batch_idx)
                gradient = gradient + (reg / len(batch_idx)) * w
                w = w - alpha * gradient

    return w.cpu().numpy()