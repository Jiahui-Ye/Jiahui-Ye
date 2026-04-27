# -*- coding: utf-8 -*-
"""
Data Loader for PU Learning with Soft-MSA (Missing in a Subset Area)

Implements two data loading strategies:
1. get_data: Random missing (SCAR - Selected Completely At Random)
2. get_data_biased: Soft-MSA missing (Non-random, harder scenario)

Author: Algorithm Engineering Team
Date: 2025-12-01
Mathematical Reference: theory_reference.md
"""

import pandas as pd
import numpy as np
import os

def get_data_pic(file, r, beta=3.0, verbose=False, data_dir='../客户data/data', balance_sampling=False, no_label_flip=False, balance_ratio=1.0):
    """
    Prediction-Induced Camouflage (PIC) 缺失机制
    基于预测的伪装缺失 - 专为稀疏数据设计
    
    PIC使用简单模型的预测置信度，利用特征的判别能力来决定标记概率。
    
    物理意义：
    - 模型置信度高（明显的假评论）-> 人工审核容易发现 -> 高概率被标记
    - 模型置信度低（难识别的假评论）-> 人工审核也难发现 -> 留在Unlabeled
    
    Parameters:
    -----------
    file : str
        Dataset name
    r : float
        Target labeling ratio
    beta : float
        Bias strength (default=3.0)
    verbose : bool
        Print detailed statistics
    data_dir : str
        Data directory path
    balance_sampling : bool
        Whether to apply class-balanced sampling
    no_label_flip : bool
        是否禁用自动标签翻转
    balance_ratio : float
        平衡比例
    
    Returns:
    --------
    data_train, data_test, pi, gamma
    """
    from sklearn.linear_model import LogisticRegression
    
    # 数据加载
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file) if data_dir else file
        d = pd.read_csv(file_path, header=None)
        d = d.drop_duplicates()
        
        labels = d[0].copy()
        unique_labels = np.unique(labels)
        
        if len(unique_labels) == 2:
            label_map = {min(unique_labels): -1, max(unique_labels): 1}
            labels = labels.map(label_map)
        elif len(unique_labels) == 3 and 0 in unique_labels:
            label_map = {-1: -1, 0: -1, 1: 1}
            labels = labels.map(label_map)
        
        feature_cols = d.columns[1:]
        d[feature_cols] = d[feature_cols].apply(
            lambda x: 2 * (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10) - 1
        )
        
        d[0] = labels
        d.rename(columns={0: 'label'}, inplace=True)
        X = d[feature_cols].values
        y_true = labels.values
        
    else:
        dataset_name = file
        npy_patterns = [f'{dataset_name}_X_array.npy', f'{dataset_name}.npy']
        csv_patterns = [f'{dataset_name}_label.csv', f'{dataset_name}.csv']
        
        X = None
        for pattern in npy_patterns:
            npy_path = os.path.join(data_dir, pattern)
            if os.path.exists(npy_path):
                X = np.load(npy_path)
                if verbose:
                    print(f"  加载特征: {pattern}")
                break
        
        if X is None:
            raise FileNotFoundError(f"找不到特征文件: {npy_patterns}")
        
        y_true = None
        for pattern in csv_patterns:
            csv_path = os.path.join(data_dir, pattern)
            if os.path.exists(csv_path):
                df_label = pd.read_csv(csv_path, header=0)
                if df_label.shape[1] >= 2:
                    y_true = df_label.iloc[:, -1].values
                elif df_label.shape[1] == 1:
                    y_true = df_label.iloc[:, 0].values
                y_true = y_true[~pd.isna(y_true)]
                if verbose:
                    print(f"  加载标签: {pattern}, 有效标签数: {len(y_true)}")
                break
        
        if y_true is None:
            raise FileNotFoundError(f"找不到标签文件: {csv_patterns}")
        
        if len(X) != len(y_true):
            raise ValueError(f"特征数({len(X)}) != 标签数({len(y_true)})")
        
        unique_labels, counts = np.unique(y_true, return_counts=True)
        label_freq = dict(zip(unique_labels, counts))
        
        minority_label = min(label_freq, key=label_freq.get)
        majority_label = max(label_freq, key=label_freq.get)
        minority_ratio = label_freq[minority_label] / len(y_true)
        
        if verbose:
            print(f"\n原始标签分布:")
            for lbl, cnt in label_freq.items():
                print(f"   Label {lbl}: {cnt} ({cnt/len(y_true)*100:.1f}%)")
        
        # 标签翻转逻辑
        if no_label_flip:
            y_true = np.where(y_true == max(unique_labels), 1, -1)
            if verbose:
                print(f"   禁用翻转: 保持原始映射")
        elif minority_ratio < 0.15:
            y_true = np.where(y_true == minority_label, 1, -1)
            if verbose:
                print(f"   标签翻转: {minority_label}->+1 (极端不平衡)")
        else:
            y_true = np.where(y_true == max(unique_labels), 1, -1)
            if verbose:
                print(f"   保持原始映射")
        
        X = X.astype(np.float32)
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        X = (X - X_min) / X_range
    
    # PIC Mechanism
    if verbose:
        print(f"\n应用 PIC (Prediction-Induced Camouflage) 缺失机制")
        print(f"   Beta = {beta:.2f}")
    
    # 训练简单逻辑回归模型获取预测置信度
    lr_model = LogisticRegression(
        max_iter=100, 
        solver='lbfgs', 
        C=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    lr_model.fit(X, y_true)
    
    if verbose:
        train_acc = lr_model.score(X, y_true)
        print(f"   初始模型训练准确率: {train_acc*100:.2f}%")
    
    # 获取正样本的预测置信度
    pos_mask = (y_true == 1)
    X_pos = X[pos_mask]
    n_pos = len(X_pos)
    
    probs = lr_model.predict_proba(X_pos)[:, 1]
    
    if verbose:
        print(f"   正样本预测置信度: min={probs.min():.3f}, max={probs.max():.3f}, mean={probs.mean():.3f}")
    
    # 将置信度转换为标记概率
    label_probs = np.power(probs, beta)
    label_probs = label_probs / (label_probs.max() + 1e-9)
    
    if verbose:
        print(f"   标记概率: min={label_probs.min():.3f}, max={label_probs.max():.3f}")
    
    # 按概率排序
    target_labeled = int(r * n_pos)
    sorted_indices = np.argsort(label_probs)[::-1]
    
    if verbose:
        print(f"   目标标记数: {target_labeled}/{n_pos} ({r*100:.1f}%)")
    
    # 构造训练/测试集
    d = pd.DataFrame(X)
    d.insert(0, 'label', y_true)
    d = d.sort_values(by='label', ascending=False)
    
    num_p = (y_true == 1).sum()
    n = len(d)
    pi = num_p / n
    
    d_p = d[d.label == 1]
    d_n = d[d.label == -1]
    
    # 双向平衡采样
    if balance_sampling:
        n_p, n_n = len(d_p), len(d_n)
        if n_n > n_p:
            target_n = int(n_p * balance_ratio)
            target_n = min(target_n, n_n)
            d_n_balanced = d_n.sample(n=target_n, replace=False, random_state=42)
            if verbose:
                print(f"\n平衡采样 (PIC - 欠采样负类):")
                print(f"   原始分布: {n_p} (+) vs {n_n} (-)")
                print(f"   欠采样后: {n_p} (+) vs {target_n} (-)")
            d_n = d_n_balanced
            d = pd.concat([d_p, d_n], axis=0).sort_values(by='label', ascending=False)
            pi = len(d_p) / len(d)
        elif n_p > n_n:
            target_p = int(n_n * balance_ratio)
            target_p = min(target_p, n_p)
            d_p_balanced = d_p.sample(n=target_p, replace=False, random_state=42)
            if verbose:
                print(f"\n平衡采样 (PIC - 欠采样正类):")
                print(f"   原始分布: {n_p} (+) vs {n_n} (-)")
                print(f"   欠采样后: {target_p} (+) vs {n_n} (-)")
            d_p = d_p_balanced
            d = pd.concat([d_p, d_n], axis=0).sort_values(by='label', ascending=False)
            pi = len(d_p) / len(d)
    
    d1 = d_p.sample(n=None, frac=0.8, replace=False, random_state=42)
    d2 = d_n.sample(n=None, frac=0.8, replace=False, random_state=42)
    
    data_train = pd.concat([d1, d2], axis=0)
    data_test = d[~d.index.isin(data_train.index)]
    
    data_train.insert(1, 'ones', 1)
    data_test.insert(1, 'ones', 1)
    
    # 应用PIC标注策略
    data_train.insert(0, 'pu', -1)
    
    train_pos_mask = (data_train['label'] == 1).values
    train_pos_indices = np.where(train_pos_mask)[0]
    n_train_pos = len(train_pos_indices)
    
    X_train_pos = data_train.iloc[train_pos_indices, 3:].values
    probs_train = lr_model.predict_proba(X_train_pos)[:, 1]
    label_probs_train = np.power(probs_train, beta)
    label_probs_train = label_probs_train / (label_probs_train.max() + 1e-9)
    
    sorted_train_indices = np.argsort(label_probs_train)[::-1]
    num_keep_labeled = int(r * n_train_pos)
    
    selected_train_indices = train_pos_indices[sorted_train_indices[:num_keep_labeled]]
    data_train.iloc[selected_train_indices, 0] = 1
    
    gamma = num_keep_labeled / len(data_train)
    
    if verbose:
        print(f"\n数据集统计 (PIC)")
        print(f"   总样本: {n}")
        print(f"   正样本: {num_p} (pi={pi:.3f})")
        print(f"   训练集: {len(data_train)} (Labeled={num_keep_labeled}, gamma={gamma:.3f})")
        print(f"   测试集: {len(data_test)}")
    
    return data_train, data_test, pi, gamma
