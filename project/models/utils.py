# -*- coding: utf-8 -*-


import torch


def get_device():
    """Auto-detect CUDA availability"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
