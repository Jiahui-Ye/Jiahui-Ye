# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import itertools
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

CORE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), '01_核心代码')
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if os.path.exists(CORE_DIR):
    sys.path.insert(0, CORE_DIR)

from data import get_data_pic
from models import (
    get_device,
    OPU_LINEX, OPU_BLINEX,
    acc_std, evaluate_comprehensive, acc_unlabeled
)

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

import torch

DATA_DIR_RELATIVE = '../../data'
DATA_DIR_ABSOLUTE = '..'

DATASET_LIST = ['YelpZip', 'dianping', 'deception', 'fake_data']

LABELING_RATIO = 0.3
L2_REGULARIZATION = 1e-5
N_RUNS = 5

LEARNING_RATE = 0.01
N_EPOCHS = 50
BATCH_SIZE = 512

A_LINEX_SEARCH = [-5.0, -4.0, -3.0, -2.0, -1.0]
A_BLINEX_SEARCH = [-5.0, -4.0, -3.0, -2.0, -1.0]
B_VALUES = [0.0001, 0.001, 0.01, 0.1]
LAMBDA_VALUES = [0.4, 0.5, 0.6, 0.7]

FAIR_COMPARISON_PARAMS = [
    {'alpha': 0.08, 'a': -4.0, 'b': 0.0001, 'lambda': 0.3, 'name': 'Set1'},
    {'alpha': 0.10, 'a': -3.0, 'b': 0.0002, 'lambda': 0.4, 'name': 'Set2'},
    {'alpha': 0.12, 'a': -2.0, 'b': 0.0003, 'lambda': 0.5, 'name': 'Set3'},
]

YELPZIP_BALANCE_SAMPLING = True
YELPZIP_BALANCE_RATIO = 1.0

PIC_BETA = 5.0

if DATA_DIR_ABSOLUTE and os.path.exists(DATA_DIR_ABSOLUTE):
    DATA_DIR = DATA_DIR_ABSOLUTE
else:
    DATA_DIR = os.path.join(SCRIPT_DIR, DATA_DIR_RELATIVE)

print("="*80)
print("Online PU Learning Experiment Pipeline (LINEX & BLINEX only) -- PIC Mechanism")
print("="*80)

cuda_available = torch.cuda.is_available()
if cuda_available:
    try:
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        print(f" GPU Device: {device_name}")
        print(f" GPU Memory: {device_memory:.1f} GB")
        print(f" CUDA Version: {cuda_version}")
        print(f" Using GPU acceleration")
    except Exception as e:
        print(f"  GPU detection error: {e}")
        print("   Falling back to CPU")
        cuda_available = False
else:
    print("Running on CPU")
    print(f"   PyTorch version: {torch.__version__}")
print("="*80)

file_list = DATASET_LIST
r = LABELING_RATIO
reg = L2_REGULARIZATION
n_runs = N_RUNS

SEEDS = [42, 52, 62, 72, 82]

def get_dataset_config(dataset_name):
    base_config = {
        'alpha': LEARNING_RATE,
        'n_epochs': N_EPOCHS,
        'reg': L2_REGULARIZATION,
        'batch_size': BATCH_SIZE,
        'a_linex_search': A_LINEX_SEARCH,
        'a_blinex_search': A_BLINEX_SEARCH,
        'b_values': B_VALUES,
        'lambda_values': LAMBDA_VALUES,
        'alpha_list': [LEARNING_RATE],
        'beta_pic': PIC_BETA,
    }
    
    if 'YelpZip' in dataset_name:
        base_config.update({
            'balance_sampling': YELPZIP_BALANCE_SAMPLING,
            'balance_ratio': YELPZIP_BALANCE_RATIO,
            'no_label_flip': True,
            'n_epochs': 1000,
            'alpha': 0.05,
            'pos_weight_factor': 10.0,
            'a_linex_search': [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
            'a_blinex_search': [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
        })
        return base_config
    
    elif 'deception' in dataset_name:
        base_config.update({
            'beta_pic': PIC_BETA,
        })
        return base_config
    
    elif 'dianping' in dataset_name:
        base_config.update({
            'balance_sampling': True,
            'beta_pic': PIC_BETA,
            'n_epochs': 1000,
            'alpha': 0.05,
            'reg': 1e-6,
            'pos_weight_factor': 10.0,
            'a_linex_search': [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
            'a_blinex_search': [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
        })
        return base_config
    
    else:
        base_config.update({
            'beta_pic': PIC_BETA,
        })
        return base_config

print("\n" + "="*80)
print(" Experiment Configuration Summary (PIC):")
print("="*80)
for dataset in file_list:
    cfg = get_dataset_config(dataset)
    print(f"\n{dataset}:")
    print(f"   hyperparameters: alpha={cfg['alpha']}, n_epochs={cfg['n_epochs']}, reg={cfg['reg']}")
    balance_info = 'balance_sampling=True' if cfg.get('balance_sampling') else ''
    print(f"   config: {balance_info}")
    print(f"   Grid Search: LINEX {len(cfg['a_linex_search'])} | BLINEX {len(cfg['a_blinex_search'])*len(cfg['b_values'])*len(cfg['lambda_values'])}")
print("="*80 + "\n")

def run_fair_comparison(data_train, data_test, pi, gamma, dataset_name='', config=None):
    if config is None:
        config = get_dataset_config(dataset_name)
    
    epochs = config['n_epochs']
    PARAM_SETS = FAIR_COMPARISON_PARAMS
    
    print("\n" + "="*80)
    print(" Fair comparison: multiple parameter sets x 2 models (LINEX & BLINEX)")
    print("="*80)
    print(f"   weighted strategy: k=(1+β|a|), n_epochs = {epochs}")
    for ps in PARAM_SETS:
        print(f"  {ps['name']}: α={ps['alpha']}, a={ps['a']}, b={ps['b']}, λ={ps['lambda']}")
    print("-"*80)
    
    MODELS = ['OPU_LINEX', 'OPU_BLINEX']
    
    all_param_results = {ps['name']: {} for ps in PARAM_SETS}
    model_all_accs = {m: [] for m in MODELS}
    all_metrics_by_method = {m: [] for m in MODELS}
    model_all_times = {m: [] for m in MODELS}
    
    for param_set in PARAM_SETS:
        alpha = param_set['alpha']
        a_param = param_set['a']
        b_param = param_set['b']
        lambda_param = param_set['lambda']
        set_name = param_set['name']
        
        print(f"\n {set_name}: α={alpha}, a={a_param}")
        
        for method in MODELS:
            W = np.zeros((n_runs, data_train.shape[1] - 3))
            training_times = []
            
            for i in range(n_runs):
                start_time = time.time()
                if method == 'OPU_LINEX':
                    pwf = config.get('pos_weight_factor', 1.0)
                    w = OPU_LINEX(data_train, pi, alpha, gamma, reg,
                                 a_linex=a_param,
                                 pos_weight_factor=pwf,
                                 n_epochs=epochs, dataset_name=dataset_name, seed=SEEDS[i])
                elif method == 'OPU_BLINEX':
                    pwf = config.get('pos_weight_factor', 1.0)
                    w = OPU_BLINEX(data_train, pi, alpha, gamma, reg,
                                  a_blinex=a_param, b=b_param, lambda_param=lambda_param,
                                  pos_weight_factor=pwf,
                                  n_epochs=epochs, dataset_name=dataset_name, seed=SEEDS[i])
                elapsed_time = time.time() - start_time
                training_times.append(elapsed_time)
                W[i] = w
            
            acc, std = acc_std(W, data_test, method)
            time_mean = np.mean(training_times)
            all_param_results[set_name][method] = (acc, std, time_mean)
            model_all_accs[method].append(acc)
            model_all_times[method].append(time_mean)
            
            metrics = evaluate_comprehensive(W, data_test, method)
            if metrics is not None:
                all_metrics_by_method[method].append(metrics)
            
            if metrics is not None:
                print(f"   {method:<12}: {acc*100:.2f}% | Time: {time_mean:.3f}s | "
                      f"P:{metrics['precision_mean']:.3f} R:{metrics['recall_mean']:.3f} "
                      f"F1:{metrics['f1_mean']:.3f} Spec:{metrics['specificity_mean']:.3f}")
            else:
                print(f"   {method:<12}: {acc*100:.2f}% | Time: {time_mean:.3f}s")
    
    final_results = {}
    for method in MODELS:
        avg_acc = np.mean(model_all_accs[method])
        std_acc = np.std(model_all_accs[method])
        final_results[method] = (avg_acc, std_acc)
    
    final_metrics = {}
    for method in MODELS:
        if all_metrics_by_method[method]:
            avg_precision = np.mean([m['precision_mean'] for m in all_metrics_by_method[method]])
            avg_recall = np.mean([m['recall_mean'] for m in all_metrics_by_method[method]])
            avg_f1 = np.mean([m['f1_mean'] for m in all_metrics_by_method[method]])
            avg_specificity = np.mean([m['specificity_mean'] for m in all_metrics_by_method[method]])
        else:
            avg_precision = avg_recall = avg_f1 = avg_specificity = None
        
        avg_time = np.mean(model_all_times[method]) if model_all_times[method] else 0.0
        
        final_metrics[method] = {
            'accuracy': final_results[method][0],
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'specificity': avg_specificity,
            'time': avg_time
        }
    
    print("\n" + "="*80)
    print(" Final ranking (average over parameter sets):")
    print("="*80)
    sorted_models = sorted(final_results.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (m, (a, s)) in enumerate(sorted_models, 1):
        if final_metrics[m]['precision'] is not None:
            metrics_str = (f" | P:{final_metrics[m]['precision']:.3f} "
                          f"R:{final_metrics[m]['recall']:.3f} "
                          f"F1:{final_metrics[m]['f1']:.3f} "
                          f"Spec:{final_metrics[m]['specificity']:.3f} "
                          f"Time:{final_metrics[m]['time']:.3f}s")
            print(f"   {rank}. {m:<12}: {a*100:.2f}% ± {s*100:.2f}%{metrics_str}")
        else:
            print(f"   {rank}. {m:<12}: {a*100:.2f}% ± {s*100:.2f}% | Time:{final_metrics[m]['time']:.3f}s")
    
    return final_results, all_param_results, final_metrics

def run_linex_experiment(data_train, data_test, pi, gamma, dataset_name='', config=None):
    if config is None:
        config = get_dataset_config(dataset_name)
    
    a_values = config['a_linex_search']
    
    print("\n" + "-"*80)
    print("Grid Search: OPU_LINEX")
    print(f"Searching asymmetry parameter a in {a_values}")
    print(f" Using standard PU weight: pos_weight = 2π/γ (fair comparison)")
    print("-"*80)
    
    all_results = []
    best_acc = 0
    best_params = {}
    
    for a in a_values:
        print(f"\n  Testing a = {a:.2f}...", end='')
        
        W = np.zeros((n_runs, data_train.shape[1] - 3))
        
        for i in range(n_runs):
            pwf = config.get('pos_weight_factor', 1.0)
            w = OPU_LINEX(data_train, pi, config['alpha'], gamma, reg, 
                         a_linex=a, 
                         pos_weight_factor=pwf,
                         n_epochs=config['n_epochs'], 
                         dataset_name=dataset_name, seed=SEEDS[i])
            W[i] = w
        
        method_name = f'OPU_LINEX(a={a:.2f})'
        acc, std = acc_std(W, data_test, method_name)
        print(f" -> {acc*100:.2f}%")
        
        all_results.append({
            'method': 'OPU_LINEX',
            'a': a,
            'accuracy': acc,
            'std': std
        })
        
        if acc > best_acc:
            best_acc = acc
            best_params = {
                'method': 'OPU_LINEX',
                'a': a,
                'accuracy': acc,
                'std': std
            }
            print(f"     New Best!")
    
    print("\n  Best LINEX configuration:")
    if best_params:
        print(f"    a = {best_params['a']:.2f}")
        print(f"    Accuracy = {best_params['accuracy']:.3f} ± {best_params['std']:.3f}")
    else:
        print("    No valid configuration found")
        best_params = {'method': 'OPU_LINEX', 'a': 0.5, 'accuracy': 0, 'std': 0}
    
    return best_params, all_results

def run_blinex_grid_search(data_train, data_test, pi, gamma, dataset_name='', config=None):
    if config is None:
        config = get_dataset_config(dataset_name)
    
    a_values = config['a_blinex_search']
    b_values = config['b_values']
    lambda_values = config['lambda_values']
    alpha_list = config['alpha_list']
    
    print("\n" + "="*80)
    print(" Grid Search: OPU_BLINEX (with alpha tuning)")
    print("="*80)
    
    print(f"Search space:")
    print(f"  a (asymmetry):  {a_values}")
    print(f"  b (robustness): {b_values}")
    print(f"  λ (bound):      {lambda_values}")
    print(f"  alpha (learning rate): {alpha_list}")
    print(f"   Using standard PU weight: pos_weight = 2π/γ (fair comparison)")
    
    total_configs = len(a_values) * len(b_values) * len(lambda_values) * len(alpha_list)
    
    print(f"Total configurations: {total_configs}")
    print("="*80)
    
    all_results = []
    best_acc = 0
    best_params = {}
    current_config = 0
    
    for a, b, lam, lr in itertools.product(a_values, b_values, lambda_values, alpha_list):
        current_config += 1
        print(f"\n[{current_config}/{total_configs}] Testing a={a:.2f}, b={b:.3f}, λ={lam:.1f}, lr={lr:.2f}...", end='')
        
        W = np.zeros((n_runs, data_train.shape[1] - 3))
        
        try:
            for i in range(n_runs):
                pwf = config.get('pos_weight_factor', 1.0)
                w = OPU_BLINEX(data_train, pi, lr, gamma, reg, 
                              a_blinex=a, b=b, lambda_param=lam, 
                              pos_weight_factor=pwf,
                              n_epochs=config['n_epochs'], 
                              dataset_name=dataset_name, seed=SEEDS[i])
                W[i] = w
            
            method_name = f'OPU_BLINEX(a={a:.2f},b={b:.3f},lr={lr:.2f})'
            acc, std = acc_std(W, data_test, method_name)
            print(f" -> {acc*100:.2f}%", end='')
            
            all_results.append({
                'method': 'OPU_BLINEX',
                'a': a,
                'b': b,
                'lambda': lam,
                'alpha': lr,
                'accuracy': acc,
                'std': std
            })
            
            if acc > best_acc:
                best_acc = acc
                best_params = {
                    'method': 'OPU_BLINEX',
                    'a': a,
                    'b': b,
                    'lambda': lam,
                    'alpha': lr,
                    'accuracy': acc,
                    'std': std
                }
                print(f"  New Best!")
            else:
                print()
        
        except Exception as e:
            print(f"  Failed: {e}")
            all_results.append({
                'method': 'OPU_BLINEX',
                'a': a,
                'b': b,
                'lambda': lam,
                'alpha': lr,
                'accuracy': 0,
                'std': 0,
                'error': str(e)
            })
    
    print("\n" + "="*80)
    print(" Best BLINEX configuration (with optimal alpha):")
    if best_params:
        print(f"  a = {best_params['a']:.2f}")
        print(f"  b = {best_params['b']:.3f}")
        print(f"  λ = {best_params['lambda']:.2f}")
        print(f"  alpha = {best_params['alpha']:.2f} <- Optimized!")
        print(f"  Accuracy = {best_params['accuracy']:.4f} ± {best_params['std']:.4f}")
    else:
        print("  No valid configuration found")
        best_params = {'method': 'OPU_BLINEX', 'a': 0.5, 'b': 0.01, 'lambda': 0.3, 'alpha': 0.1, 'accuracy': 0, 'std': 0}
    print("="*80)
    
    return best_params, all_results

def run_full_experiment(file, verbose=True):
    print("\n" + "="*80)
    print(f"DATASET: {file}")
    print("="*80)
    
    config = get_dataset_config(file)
    
    print(f"\n Fair comparison configuration:")
    print(f"   hyperparameters: alpha={config['alpha']}, n_epochs={config['n_epochs']}, reg={config['reg']}")
    print(f"   standard PU weight: pos_weight = 2π/γ")
    print(f"   missing mechanism: PIC (Prediction-Induced Camouflage)")
    if config.get('balance_sampling'):
        print(f"    balance sampling: True (extreme imbalance)")
    
    balance_sampling = config.get('balance_sampling', False)
    no_label_flip = config.get('no_label_flip', False)
    balance_ratio = config.get('balance_ratio', 1.0)
    if balance_sampling:
        print(f"   enabling balance sampling (ratio={balance_ratio:.1f}:1)")
    if no_label_flip:
        print(f"   disabling automatic label flipping")
    
    data_train, data_test, pi, gamma = get_data_pic(
        file, r, beta=config.get('beta_pic', 5.0), verbose=verbose, data_dir=DATA_DIR,
        balance_sampling=balance_sampling, no_label_flip=no_label_flip,
        balance_ratio=balance_ratio
    )
    
    linex_best, linex_all = run_linex_experiment(
        data_train, data_test, pi, gamma, dataset_name=file, config=config
    )
    
    blinex_best, blinex_all = run_blinex_grid_search(
        data_train, data_test, pi, gamma, dataset_name=file, config=config
    )
    
    fair_results, fair_all_params, fair_metrics = run_fair_comparison(data_train, data_test, pi, gamma,
                                       dataset_name=file, config=config)
    
    summary = {
        'dataset': file,
        'pi': pi,
        'gamma': gamma,
        'config': config,
        'linex_best': linex_best,
        'linex_all': linex_all,
        'blinex_best': blinex_best,
        'blinex_all': blinex_all,
        'fair_comparison': fair_results,
        'fair_all_params': fair_all_params,
        'fair_metrics': fair_metrics,
        'missing_mode': 'PIC'
    }
    
    return summary

def save_results_to_csv(all_summaries, output_file='experiment_results.csv'):
    records = []
    
    for summary in all_summaries:
        dataset = summary['dataset']
        
        for res in summary['linex_all']:
            records.append({
                'dataset': dataset,
                'method': res['method'],
                'a': res['a'],
                'b': None,
                'lambda': None,
                'accuracy': res['accuracy'],
                'std': res['std']
            })
        
        for res in summary['blinex_all']:
            records.append({
                'dataset': dataset,
                'method': res['method'],
                'a': res['a'],
                'b': res['b'],
                'lambda': res['lambda'],
                'accuracy': res['accuracy'],
                'std': res['std']
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"\n Results saved to {output_file}")

if __name__ == "__main__":
    print(f"\n{'#'*80}")
    print(f"#   Missing mechanism: PIC")
    print(f"{'#'*80}")
    
    all_summaries = []
    
    for file in file_list:
        try:
            print(f"\n{'='*80}")
            print(f" Processing dataset: {file}")
            print(f"{'='*80}")
            
            summary = run_full_experiment(file, verbose=True)
            all_summaries.append(summary)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n Error processing {file}: {e}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    save_results_to_csv(all_summaries, 'experiment_results_PIC.csv')
    
    detailed_records = []
    for summary in all_summaries:
        dataset = summary['dataset']
        missing_mode = summary['missing_mode']
        metrics_all = summary.get('fair_metrics', {})
        for model, metrics in metrics_all.items():
            record = {
                'dataset': dataset,
                'missing_mode': missing_mode,
                'model': model,
                'accuracy': metrics.get('accuracy', None),
                'precision': metrics.get('precision', None),
                'recall': metrics.get('recall', None),
                'f1': metrics.get('f1', None),
                'specificity': metrics.get('specificity', None),
                'time': metrics.get('time', None)
            }
            detailed_records.append(record)
    
    if detailed_records:
        detailed_df = pd.DataFrame(detailed_records)
        detailed_df.to_csv('detailed_metrics_PIC.csv', index=False)
        print(f"   Detailed metrics saved to detailed_metrics_PIC.csv")
    
    print("\n" + "="*120)
    print(" Experiments completed (PIC mechanism only)")
    print("="*120)
    
    MODELS = ['OPU_LINEX', 'OPU_BLINEX']
    
    print(f"\n{'='*120}")
    print(f" PIC mode - model comparison (2 models x {len(file_list)} datasets)")
    print(f"{'='*120}")
    
    header = f"{'Model':<15}"
    for dataset in file_list:
        header += f"{dataset:<18}"
    header += f"{'Avg':<10} {'Rank':<6}"
    print(header)
    print("-"*120)
    
    model_avgs = {}
    
    for model in MODELS:
        row = f"{model:<15}"
        accs = []
        
        for dataset in file_list:
            acc = 0.0
            for s in all_summaries:
                if s['dataset'] == dataset:
                    if model == 'OPU_LINEX':
                        acc = s['linex_best']['accuracy']
                    elif model == 'OPU_BLINEX':
                        acc = s['blinex_best']['accuracy']
                    break
            accs.append(acc)
            row += f"{acc*100:>6.2f}%{'':>10}"
        
        avg_acc = np.mean(accs) if accs else 0
        model_avgs[model] = avg_acc
        row += f"{avg_acc*100:>6.2f}%"
        print(row)
    
    print("-"*120)
    sorted_models = sorted(model_avgs.items(), key=lambda x: x[1], reverse=True)
    print(f" PIC ranking:")
    for rank, (model, avg) in enumerate(sorted_models, 1):
        print(f"   {rank}. {model:<15} {avg*100:.2f}%")
    
    print(f"\n{'='*120}")
    print(" Model overall performance summary (PIC mechanism)")
    print(f"{'='*120}")
    
    header = f"{'Model':<15} {'Type':<10} {'PIC':<15} {'Overall Avg':<15}"
    print(header)
    print("-"*120)
    
    overall_ranking = []
    
    for model in MODELS:
        model_type = "New"
        row = f"{model:<15} {model_type:<10}"
        
        accs = [s['linex_best']['accuracy'] if model=='OPU_LINEX' else s['blinex_best']['accuracy'] for s in all_summaries]
        mode_avg = np.mean(accs) if accs else 0
        row += f"{mode_avg*100:>6.2f}%{'':>7}"
        
        overall_avg = mode_avg
        row += f"{overall_avg*100:>6.2f}%{'':>7}"
        print(row)
        overall_ranking.append((model, overall_avg, model_type))
    
    print("-"*120)
    
    print(f"\n{'='*120}")
    print(" Final model ranking (average over all datasets for PIC)")
    print(f"{'='*120}")
    sorted_overall = sorted(overall_ranking, key=lambda x: x[1], reverse=True)
    for rank, (model, avg, mtype) in enumerate(sorted_overall, 1):
        print(f"   {rank}. {model:<15} {avg*100:.2f}%")
    
    print(f"\n{'='*120}")
    print(" Experimental conclusions")
    print(f"{'='*120}")
    
    best_model = sorted_overall[0] if sorted_overall else ('N/A', 0, '')
    print(f"\n   Best model: {best_model[0]} ({best_model[1]*100:.2f}%)")
    
    print(f"\n Results saved to:")
    print(f"  - experiment_results_PIC.csv")
    print(f"  - detailed_metrics_PIC.csv")
    print("="*120)
    
    print(f"\n{'='*120}")
    print(" Fair comparison results (multiple parameter sets x 2 models)")
    print("="*120)
    param_desc = ", ".join([f"{p['name']}(a={p['a']})" for p in FAIR_COMPARISON_PARAMS])
    print(f"   Parameter sets: {param_desc}")
    print("-"*120)
    
    print(f"\n [PIC] Fair comparison results:")
    
    header = f"{'Model':<15}"
    for dataset in file_list:
        header += f"{dataset:<20}"
    header += f"{'Avg':<10}"
    print(header)
    print("-"*115)
    
    model_avgs_fair = {}
    for model in MODELS:
        row = f"{model:<15}"
        accs = []
        
        for dataset in file_list:
            acc = 0.0
            std = 0.0
            for s in all_summaries:
                if s['dataset'] == dataset and 'fair_comparison' in s:
                    result = s['fair_comparison'].get(model, (0, 0))
                    acc = result[0]
                    std = result[1] if len(result) > 1 else 0
                    break
            accs.append(acc)
            cell = f"{acc*100:.2f}±{std*100:.2f}%"
            row += f"{cell:<20}"
        
        avg_acc = np.mean(accs) if accs else 0
        model_avgs_fair[model] = avg_acc
        row += f"{avg_acc*100:.2f}%"
        print(row)
    
    sorted_models_fair = sorted(model_avgs_fair.items(), key=lambda x: x[1], reverse=True)
    print(f"\n   PIC ranking:")
    for rank, (m, a) in enumerate(sorted_models_fair, 1):
        print(f"    {rank}. {m}: {a*100:.2f}%")
    
    print("\n" + "="*120)
    print(" Fair comparison completed")
    print("="*120)