#!/usr/bin/env python3
"""
Clean Analysis Script for Neural Network Atlas Experiments

This script correctly calculates retention rates by comparing class-specific
accuracies before and after ablation.

CSV Structure:
- First column: row index (ignored)
- Second column (labeled "0"): ablation_class
  * -1 = baseline (no ablation)
  * 0-9 = class that was ablated
- Columns 2-11 (labeled "1"-"10"): predictions for classes 0-9

Data layout:
- Rows 0-9: baseline confusion matrix (ablation=-1)
- Rows 10-19: class 0 ablated (ablation=0)
- Rows 20-29: class 1 ablated (ablation=1)
- etc.

Each 10-row block forms a complete 10x10 confusion matrix where:
- Row within block = true class (0-9)
- Column = predicted class (0-9)
"""

import csv
import json
from io import StringIO
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import entropy


def parse_confusion_matrix_csv(csv_file):
    """
    Parse confusion matrix CSV file.
    
    Returns:
        dict: {ablation_class: confusion_matrix_array}
              where confusion_matrix_array is shape (10, 10)
    """
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Group rows by ablation class
    ablation_matrices = {}
    
    for row in rows[1:]:  # Skip header row
        if not row or len(row) < 12:
            continue
        
        # Column structure: [row_index, ablation_class, pred_0, ..., pred_9, extra]
        ablation_class = int(row[1])
        predictions = np.array([int(x) for x in row[2:12]])  # Take exactly 10 predictions
        
        if ablation_class not in ablation_matrices:
            ablation_matrices[ablation_class] = []
        
        ablation_matrices[ablation_class].append(predictions)
    
    # Convert lists to numpy arrays
    for abl_class in ablation_matrices:
        ablation_matrices[abl_class] = np.array(ablation_matrices[abl_class])
    
    return ablation_matrices


def calculate_per_class_accuracy(confusion_matrix):
    """
    Calculate per-class accuracy from confusion matrix.
    
    Args:
        confusion_matrix: (10, 10) numpy array
    
    Returns:
        dict: {class_id: accuracy} for each class
    """
    per_class_acc = {}
    
    for true_class in range(10):
        total_samples = confusion_matrix[true_class].sum()
        if total_samples > 0:
            correct = confusion_matrix[true_class, true_class]
            per_class_acc[true_class] = correct / total_samples
        else:
            per_class_acc[true_class] = 0.0
    
    return per_class_acc


def calculate_overall_accuracy(confusion_matrix):
    """Calculate overall accuracy from confusion matrix."""
    total_correct = np.trace(confusion_matrix)
    total_samples = confusion_matrix.sum()
    return total_correct / total_samples if total_samples > 0 else 0.0


def calculate_retention_rates(baseline_cm, ablation_matrices):
    """
    Calculate retention rates: how much accuracy is retained after ablation.
    
    CORRECT METHOD:
    retention_rate[class_i] = accuracy_after_ablation[class_i] / baseline_accuracy[class_i]
    
    This compares the SAME class before and after ablation.
    
    Args:
        baseline_cm: (10, 10) baseline confusion matrix
        ablation_matrices: dict of {ablation_class: confusion_matrix}
    
    Returns:
        dict: {ablation_class: retention_rate}
    """
    # Get baseline per-class accuracies
    baseline_per_class = calculate_per_class_accuracy(baseline_cm)
    
    retention_rates = {}
    
    for abl_class, ablated_cm in ablation_matrices.items():
        if abl_class == -1:  # Skip baseline
            continue
        
        # Calculate accuracy for the ablated class after ablation
        ablated_per_class = calculate_per_class_accuracy(ablated_cm)
        
        # Retention rate: how much of the baseline accuracy is retained
        if abl_class in baseline_per_class and baseline_per_class[abl_class] > 0:
            retention = ablated_per_class[abl_class] / baseline_per_class[abl_class]
            retention_rates[abl_class] = retention
    
    return retention_rates


def calculate_modularity_score(confusion_matrix):
    """
    Calculate modularity score using KL divergence.
    
    For each class, calculate KL divergence between:
    - Actual distribution of predictions for that class
    - Uniform distribution (1/10 for each class)
    
    Higher KL divergence = more specialized/modular
    """
    uniform_dist = np.ones(10) / 10
    kl_divergences = []
    
    for true_class in range(10):
        class_predictions = confusion_matrix[true_class]
        total = class_predictions.sum()
        
        if total > 0:
            pred_dist = class_predictions / total
            # Add small epsilon to avoid log(0)
            pred_dist = pred_dist + 1e-10
            kl_div = entropy(pred_dist, uniform_dist)
            kl_divergences.append(kl_div)
    
    return np.mean(kl_divergences) if kl_divergences else 0.0


def analyze_trial(trial_dir):
    """
    Analyze a single trial.
    
    Returns:
        dict with metrics for this trial
    """
    # Extract experiment base name
    exp_full_name = trial_dir.parent.parent.name
    
    # Handle "fashion_mnist_X_hash" vs "mnist_X_hash" patterns
    if exp_full_name.startswith('fashion_mnist_'):
        # fashion_mnist_a_hash -> fashion_mnist_a
        exp_base_name = '_'.join(exp_full_name.split('_')[:3])
    else:
        # mnist_a_hash -> mnist_a
        exp_base_name = '_'.join(exp_full_name.split('_')[:2])
    
    cm_file = trial_dir / f"{exp_base_name}_{trial_dir.name}_ablation_cm.csv"
    msk_file = trial_dir / f"{exp_base_name}_{trial_dir.name}_ablation_msk.csv"
    
    if not cm_file.exists():
        return None
    
    # Parse confusion matrices
    ablation_matrices = parse_confusion_matrix_csv(cm_file)
    
    if -1 not in ablation_matrices:
        print(f"WARNING: No baseline found in {cm_file}")
        return None
    
    baseline_cm = ablation_matrices[-1]
    
    # Calculate metrics
    baseline_acc = calculate_overall_accuracy(baseline_cm)
    modularity = calculate_modularity_score(baseline_cm)
    retention_rates = calculate_retention_rates(baseline_cm, ablation_matrices)
    
    # Calculate reduction percentage from mask file
    reduction_pcts = []
    if msk_file.exists():
        with open(msk_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Get total neurons from baseline row (ablation_class = -1)
            total_neurons = None
            for row in rows[1:]:
                if len(row) >= 2 and int(row[1]) == -1:
                    total_neurons = sum(int(x) for x in row[2:12])
                    break
            
            # Calculate reduction for each ablation
            if total_neurons and total_neurons > 0:
                for row in rows[1:]:
                    if len(row) >= 2:
                        ablation_class = int(row[1])
                        if ablation_class >= 0:
                            masked_neurons = sum(int(x) for x in row[2:12])
                            reduction_pct = (masked_neurons / total_neurons) * 100
                            reduction_pcts.append(reduction_pct)
    
    # Average retention across all ablated classes
    avg_retention = np.mean(list(retention_rates.values())) if retention_rates else 0.0
    std_retention = np.std(list(retention_rates.values())) if retention_rates else 0.0
    
    avg_reduction = np.mean(reduction_pcts) if reduction_pcts else 0.0
    std_reduction = np.std(reduction_pcts) if reduction_pcts else 0.0
    
    return {
        'baseline_accuracy': baseline_acc,
        'modularity_score': modularity,
        'retention_rate_mean': avg_retention,
        'retention_rate_std': std_retention,
        'reduction_pct_mean': avg_reduction,
        'reduction_pct_std': std_reduction,
        'retention_rates_per_class': retention_rates
    }


def analyze_experiment(exp_dir):
    """
    Analyze all trials in an experiment.
    
    Returns:
        list of trial results
    """
    trials_dir = exp_dir / "trials"
    
    if not trials_dir.exists():
        return []
    
    results = []
    
    for trial_dir in sorted(trials_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        
        trial_result = analyze_trial(trial_dir)
        
        if trial_result:
            trial_result['experiment'] = exp_dir.name
            trial_result['trial'] = trial_dir.name
            results.append(trial_result)
    
    return results


def load_all_trials_from_ax_snapshot(json_file):
    """
    Load all trial metrics from ax_client_snapshot.json.
    
    Args:
        json_file: Path to ax_client_snapshot.json
        
    Returns:
        DataFrame with columns: trial_id, trial_phase, val_acc, modularity_score, train_acc, parameters...
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    experiment = data['experiment']
    trials = experiment['trials']
    
    all_trials = []
    for trial_id, trial_data in trials.items():
        trial_num = int(trial_id)
        row = {
            'trial_id': trial_num,
            'trial_phase': 'random' if trial_num < 25 else 'bayesian',
            'status': trial_data['status']['name']
        }
        
        # Extract parameters from generator_run
        if 'generator_run' in trial_data and trial_data['generator_run']:
            gen_run = trial_data['generator_run']
            if 'arms' in gen_run and gen_run['arms']:
                arm = gen_run['arms'][0]
                if 'parameters' in arm:
                    row.update(arm['parameters'])
        
        all_trials.append(row)
    
    # Extract metrics from data_by_trial
    if 'data_by_trial' in experiment:
        for trial_id_str, trial_data_dict in experiment['data_by_trial'].items():
            trial_id = int(trial_id_str)
            
            # Find the corresponding row
            for row in all_trials:
                if row['trial_id'] == trial_id:
                    # Extract metrics from the nested structure
                    if 'value' in trial_data_dict and trial_data_dict['value']:
                        # Get the last data point (most recent)
                        data_point = trial_data_dict['value'][-1]
                        if len(data_point) > 1 and 'df' in data_point[1]:
                            df_dict = data_point[1]['df']
                            if 'value' in df_dict:
                                # Parse the JSON string containing metrics
                                metrics_df = pd.read_json(StringIO(df_dict['value']))
                                # Extract mean values for each metric
                                for _, metric_row in metrics_df.iterrows():
                                    row[metric_row['metric_name']] = metric_row['mean']
                    break
    
    return pd.DataFrame(all_trials)


def load_pareto_trials(json_file):
    """
    Load Pareto optimal trial results from pareto_optimal_results.json.
    
    Args:
        json_file: Path to pareto_optimal_results.json
        
    Returns:
        DataFrame with trial_id, parameters, and metrics for Pareto solutions
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pareto_trials = []
    for trial_id, (params, metrics) in data.items():
        row = {'trial_id': int(trial_id)}
        row.update(params)
        
        # Extract metrics (first element of metrics list)
        if metrics and len(metrics) > 0:
            row.update(metrics[0])
        
        pareto_trials.append(row)
    
    return pd.DataFrame(pareto_trials)


def generate_5_tables(output_base, report_dir):
    """
    Generate 5 comprehensive tables (both LaTeX and CSV):
    1. All 50 trials MNIST (variants A-D)
    2. All 50 trials Fashion-MNIST (variants A-D)
    3. All 4 variants A-D (regardless of dataset)
    4. Best Pareto MNIST (variants A-D)
    5. Best Pareto Fashion-MNIST (variants A-D)
    
    Args:
        output_base: Path to output directory
        report_dir: Path to save tables
    """
    
    # Collect all trials data from ax_client_snapshot.json files
    all_trials_data = []
    pareto_data = []
    
    exp_dirs = [d for d in output_base.iterdir() 
                if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('logs')]
    
    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name
        
        # Extract dataset and variant
        if exp_name.startswith('fashion_mnist_'):
            dataset = "fashion_mnist"
            variant = exp_name.split('_')[2]
        elif exp_name.startswith('mnist_'):
            dataset = "mnist"
            variant = exp_name.split('_')[1]
        else:
            continue
        
        # Load all trials from ax_client_snapshot.json
        ax_file = exp_dir / "config" / "ax_client_snapshot.json"
        if ax_file.exists() and ax_file.stat().st_size > 0:
            trials_df = load_all_trials_from_ax_snapshot(ax_file)
            trials_df['dataset'] = dataset
            trials_df['variant'] = variant.upper()
            all_trials_data.append(trials_df)
        
        # Load Pareto optimal trials
        pareto_file = exp_dir / "pareto_optimal_results.json"
        if pareto_file.exists():
            pareto_df = load_pareto_trials(pareto_file)
            pareto_df['dataset'] = dataset
            pareto_df['variant'] = variant.upper()
            pareto_data.append(pareto_df)
    
    # Combine all data
    if not all_trials_data:
        print("No trial data found!")
        return
    
    all_trials = pd.concat(all_trials_data, ignore_index=True)
    all_pareto = pd.concat(pareto_data, ignore_index=True) if pareto_data else pd.DataFrame()
    
    # Open LaTeX file
    latex_file = report_dir / "tables.tex"
    
    with open(latex_file, 'w') as f:
        # TABLE 1: All 50 trials MNIST (A-D)
        f.write("% Table 1: All 50 Trials - MNIST (Variants A-D)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{MNIST: All Trials Summary Across Variants}\n")
        f.write("\\label{tab:mnist_all_trials}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Variant & Val Acc. & Modularity & \\# Trials & Status \\\\\n")
        f.write("\\hline\n")
        
        mnist_trials = all_trials[all_trials['dataset'] == 'mnist']
        table1_data = []
        
        for variant in sorted(mnist_trials['variant'].unique()):
            var_data = mnist_trials[mnist_trials['variant'] == variant]
            if 'val_acc' in var_data.columns and 'modularity_score' in var_data.columns:
                val_acc_mean = var_data['val_acc'].mean()
                val_acc_std = var_data['val_acc'].std()
                mod_mean = var_data['modularity_score'].mean()
                mod_std = var_data['modularity_score'].std()
                n_trials = len(var_data)
                
                f.write(f"{variant} & ")
                f.write(f"{val_acc_mean:.4f} $\\pm$ {val_acc_std:.4f} & ")
                f.write(f"{mod_mean:.2f} $\\pm$ {mod_std:.2f} & ")
                f.write(f"{n_trials} & Complete \\\\\n")
                
                table1_data.append({
                    'Variant': variant,
                    'Val_Acc_Mean': val_acc_mean,
                    'Val_Acc_Std': val_acc_std,
                    'Modularity_Mean': mod_mean,
                    'Modularity_Std': mod_std,
                    'N_Trials': n_trials
                })
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Save Table 1 CSV
        pd.DataFrame(table1_data).to_csv(report_dir / "table1_mnist_all_trials.csv", index=False)
        
        # TABLE 2: All 50 trials Fashion-MNIST (A-D)
        f.write("% Table 2: All 50 Trials - Fashion-MNIST (Variants A-D)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Fashion-MNIST: All Trials Summary Across Variants}\n")
        f.write("\\label{tab:fashion_mnist_all_trials}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Variant & Val Acc. & Modularity & \\# Trials & Status \\\\\n")
        f.write("\\hline\n")
        
        fmnist_trials = all_trials[all_trials['dataset'] == 'fashion_mnist']
        table2_data = []
        
        for variant in sorted(fmnist_trials['variant'].unique()):
            var_data = fmnist_trials[fmnist_trials['variant'] == variant]
            if 'val_acc' in var_data.columns and 'modularity_score' in var_data.columns:
                val_acc_mean = var_data['val_acc'].mean()
                val_acc_std = var_data['val_acc'].std()
                mod_mean = var_data['modularity_score'].mean()
                mod_std = var_data['modularity_score'].std()
                n_trials = len(var_data)
                
                f.write(f"{variant} & ")
                f.write(f"{val_acc_mean:.4f} $\\pm$ {val_acc_std:.4f} & ")
                f.write(f"{mod_mean:.2f} $\\pm$ {mod_std:.2f} & ")
                f.write(f"{n_trials} & Complete \\\\\n")
                
                table2_data.append({
                    'Variant': variant,
                    'Val_Acc_Mean': val_acc_mean,
                    'Val_Acc_Std': val_acc_std,
                    'Modularity_Mean': mod_mean,
                    'Modularity_Std': mod_std,
                    'N_Trials': n_trials
                })
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Save Table 2 CSV
        pd.DataFrame(table2_data).to_csv(report_dir / "table2_fashion_mnist_all_trials.csv", index=False)
        
        # TABLE 3: All 4 variants A-D (regardless of dataset)
        f.write("% Table 3: Variants A-D Comparison (Dataset-Agnostic)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Experimental Variants A-D (Averaged Across Datasets)}\n")
        f.write("\\label{tab:variants_comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Variant & Val Acc. & Modularity & \\# Trials & Datasets \\\\\n")
        f.write("\\hline\n")
        
        table3_data = []
        
        for variant in sorted(all_trials['variant'].unique()):
            var_data = all_trials[all_trials['variant'] == variant]
            if 'val_acc' in var_data.columns and 'modularity_score' in var_data.columns:
                val_acc_mean = var_data['val_acc'].mean()
                val_acc_std = var_data['val_acc'].std()
                mod_mean = var_data['modularity_score'].mean()
                mod_std = var_data['modularity_score'].std()
                n_trials = len(var_data)
                n_datasets = var_data['dataset'].nunique()
                
                f.write(f"{variant} & ")
                f.write(f"{val_acc_mean:.4f} $\\pm$ {val_acc_std:.4f} & ")
                f.write(f"{mod_mean:.2f} $\\pm$ {mod_std:.2f} & ")
                f.write(f"{n_trials} & {n_datasets} \\\\\n")
                
                table3_data.append({
                    'Variant': variant,
                    'Val_Acc_Mean': val_acc_mean,
                    'Val_Acc_Std': val_acc_std,
                    'Modularity_Mean': mod_mean,
                    'Modularity_Std': mod_std,
                    'N_Trials': n_trials,
                    'N_Datasets': n_datasets
                })
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Save Table 3 CSV
        pd.DataFrame(table3_data).to_csv(report_dir / "table3_variants_comparison.csv", index=False)
        
        # TABLE 4: Best Pareto MNIST (A-D)
        f.write("% Table 4: Best Pareto Solutions - MNIST (Variants A-D)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{MNIST: Pareto Optimal Solutions}\n")
        f.write("\\label{tab:pareto_mnist}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Variant & \\# Solutions & Best Val Acc. & Avg Val Acc. & Avg Modularity \\\\\n")
        f.write("\\hline\n")
        
        mnist_pareto = all_pareto[all_pareto['dataset'] == 'mnist']
        table4_data = []
        
        for variant in sorted(mnist_pareto['variant'].unique()):
            var_data = mnist_pareto[mnist_pareto['variant'] == variant]
            if 'val_acc' in var_data.columns:
                n_solutions = len(var_data)
                best_val_acc = var_data['val_acc'].max()
                avg_val_acc = var_data['val_acc'].mean()
                avg_mod = var_data['modularity_score'].mean() if 'modularity_score' in var_data.columns else 0.0
                
                f.write(f"{variant} & {n_solutions} & ")
                f.write(f"{best_val_acc:.4f} & {avg_val_acc:.4f} & {avg_mod:.2f} \\\\\n")
                
                table4_data.append({
                    'Variant': variant,
                    'N_Solutions': n_solutions,
                    'Best_Val_Acc': best_val_acc,
                    'Avg_Val_Acc': avg_val_acc,
                    'Avg_Modularity': avg_mod
                })
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Save Table 4 CSV
        pd.DataFrame(table4_data).to_csv(report_dir / "table4_pareto_mnist.csv", index=False)
        
        # TABLE 5: Best Pareto Fashion-MNIST (A-D)
        f.write("% Table 5: Best Pareto Solutions - Fashion-MNIST (Variants A-D)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Fashion-MNIST: Pareto Optimal Solutions}\n")
        f.write("\\label{tab:pareto_fashion_mnist}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Variant & \\# Solutions & Best Val Acc. & Avg Val Acc. & Avg Modularity \\\\\n")
        f.write("\\hline\n")
        
        fmnist_pareto = all_pareto[all_pareto['dataset'] == 'fashion_mnist']
        table5_data = []
        
        for variant in sorted(fmnist_pareto['variant'].unique()):
            var_data = fmnist_pareto[fmnist_pareto['variant'] == variant]
            if 'val_acc' in var_data.columns:
                n_solutions = len(var_data)
                best_val_acc = var_data['val_acc'].max()
                avg_val_acc = var_data['val_acc'].mean()
                avg_mod = var_data['modularity_score'].mean() if 'modularity_score' in var_data.columns else 0.0
                
                f.write(f"{variant} & {n_solutions} & ")
                f.write(f"{best_val_acc:.4f} & {avg_val_acc:.4f} & {avg_mod:.2f} \\\\\n")
                
                table5_data.append({
                    'Variant': variant,
                    'N_Solutions': n_solutions,
                    'Best_Val_Acc': best_val_acc,
                    'Avg_Val_Acc': avg_val_acc,
                    'Avg_Modularity': avg_mod
                })
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Save Table 5 CSV
        pd.DataFrame(table5_data).to_csv(report_dir / "table5_pareto_fashion_mnist.csv", index=False)
    
    print(f"\nGenerated 5 tables:")
    print(f"  LaTeX: {latex_file}")
    print(f"  CSV: table1_mnist_all_trials.csv")
    print(f"  CSV: table2_fashion_mnist_all_trials.csv")
    print(f"  CSV: table3_variants_comparison.csv")
    print(f"  CSV: table4_pareto_mnist.csv")
    print(f"  CSV: table5_pareto_fashion_mnist.csv")


def generate_latex_tables(summary_df, report_dir):
    """
    Generate LaTeX tables from summary data.
    
    Args:
        summary_df: DataFrame with summary statistics
        report_dir: Path to save LaTeX files
    """
    latex_file = report_dir / "tables.tex"
    
    with open(latex_file, 'w') as f:
        # Write main results table
        f.write("% Main Results Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Analysis Results by Dataset and Variant}\n")
        f.write("\\label{tab:ablation_results}\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\hline\n")
        f.write("Dataset & Variant & Trials & Baseline Acc. & Modularity & Reduction (\\%) & Retention \\\\\n")
        f.write("\\hline\n")
        
        for _, row in summary_df.iterrows():
            dataset = row['Dataset'].replace('_', '\\_')
            f.write(f"{dataset} & {row['Variant']} & {row['Trials']} & ")
            f.write(f"{row['Baseline_Acc_Mean']:.3f} $\\pm$ {row['Baseline_Acc_Std']:.3f} & ")
            f.write(f"{row['Modularity_Mean']:.3f} $\\pm$ {row['Modularity_Std']:.3f} & ")
            f.write(f"{row['Reduction_Mean']:.1f} $\\pm$ {row['Reduction_Std']:.1f} & ")
            f.write(f"{row['Retention_Mean']:.3f} $\\pm$ {row['Retention_Std']:.3f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Write separate tables for each dataset
        for dataset in summary_df['Dataset'].unique():
            dataset_data = summary_df[summary_df['Dataset'] == dataset]
            dataset_name = dataset.replace('_', '\\_')
            
            f.write(f"% {dataset} Results\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{{dataset_name} Ablation Results}}\n")
            f.write(f"\\label{{tab:{dataset.lower()}_results}}\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\hline\n")
            f.write("Variant & Baseline Acc. & Modularity & Reduction (\\%) & Retention \\\\\n")
            f.write("\\hline\n")
            
            for _, row in dataset_data.iterrows():
                f.write(f"{row['Variant']} & ")
                f.write(f"{row['Baseline_Acc_Mean']:.3f} $\\pm$ {row['Baseline_Acc_Std']:.3f} & ")
                f.write(f"{row['Modularity_Mean']:.3f} $\\pm$ {row['Modularity_Std']:.3f} & ")
                f.write(f"{row['Reduction_Mean']:.1f} $\\pm$ {row['Reduction_Std']:.1f} & ")
                f.write(f"{row['Retention_Mean']:.3f} $\\pm$ {row['Retention_Std']:.3f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
    
    print(f"Saved LaTeX tables to: {latex_file}")


def main():
    """Run analysis on all experiments."""
    output_base = Path("/home/cesare/Projects/StrangeNet/output")
    report_dir = Path("/home/cesare/Projects/StrangeNet/report")
    report_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("CLEAN ANALYSIS - RETENTION RATE VERIFICATION")
    print("=" * 80)
    print()
    
    all_results = []
    
    # Find all experiment directories
    exp_dirs = [d for d in output_base.iterdir() 
                if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('logs')]
    
    for exp_dir in sorted(exp_dirs):
        print(f"Processing: {exp_dir.name}")
        
        trial_results = analyze_experiment(exp_dir)
        all_results.extend(trial_results)
        
        if trial_results:
            # Extract dataset and variant
            exp_name = exp_dir.name
            if exp_name.startswith('fashion_mnist_'):
                dataset = "FASHION_MNIST"
                variant = exp_name.split('_')[2]  # Get the letter (a, b, c, d)
            elif exp_name.startswith('mnist_'):
                dataset = "MNIST"
                variant = exp_name.split('_')[1]  # Get the letter (a, b, c, d)
            else:
                dataset = "UNKNOWN"
                variant = "?"
            
            # Print summary for this experiment
            retention_means = [r['retention_rate_mean'] for r in trial_results]
            modularity_scores = [r['modularity_score'] for r in trial_results]
            baseline_accs = [r['baseline_accuracy'] for r in trial_results]
            reduction_means = [r['reduction_pct_mean'] for r in trial_results]
            
            print(f"  Dataset: {dataset}, Variant: {variant.upper()}")
            print(f"  Trials: {len(trial_results)}")
            print(f"  Baseline Accuracy: {np.mean(baseline_accs):.4f} ± {np.std(baseline_accs):.4f}")
            print(f"  Modularity Score: {np.mean(modularity_scores):.4f} ± {np.std(modularity_scores):.4f}")
            print(f"  Ablation Reduction: {np.mean(reduction_means):.2f}% ± {np.std(reduction_means):.2f}%")
            print(f"  Retention Rate: {np.mean(retention_means):.4f} ± {np.std(retention_means):.4f}")
            print()
    
    # Save detailed results
    if all_results:
        # Flatten retention_rates_per_class for CSV
        rows = []
        for result in all_results:
            base_row = {
                'experiment': result['experiment'],
                'trial': result['trial'],
                'baseline_accuracy': result['baseline_accuracy'],
                'modularity_score': result['modularity_score'],
                'reduction_pct_mean': result['reduction_pct_mean'],
                'reduction_pct_std': result['reduction_pct_std'],
                'retention_rate_mean': result['retention_rate_mean'],
                'retention_rate_std': result['retention_rate_std']
            }
            
            # Add per-class retention rates
            for class_id, retention in result['retention_rates_per_class'].items():
                base_row[f'retention_class_{class_id}'] = retention
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        output_file = report_dir / "analysis_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")
        
        # Print summary by dataset and variant
        print("\n" + "=" * 80)
        print("SUMMARY BY CONFIGURATION")
        print("=" * 80)
        
        summary_data = []
        
        # Group by dataset and variant
        for exp_name in sorted(df['experiment'].unique()):
            exp_data = df[df['experiment'] == exp_name]
            
            if exp_name.startswith('fashion_mnist_'):
                dataset = "FASHION_MNIST"
                variant = exp_name.split('_')[2].upper()
            elif exp_name.startswith('mnist_'):
                dataset = "MNIST"
                variant = exp_name.split('_')[1].upper()
            else:
                dataset = "UNKNOWN"
                variant = "?"
            
            summary_row = {
                'Dataset': dataset,
                'Variant': variant,
                'Trials': len(exp_data),
                'Baseline_Acc_Mean': exp_data['baseline_accuracy'].mean(),
                'Baseline_Acc_Std': exp_data['baseline_accuracy'].std(),
                'Modularity_Mean': exp_data['modularity_score'].mean(),
                'Modularity_Std': exp_data['modularity_score'].std(),
                'Reduction_Mean': exp_data['reduction_pct_mean'].mean(),
                'Reduction_Std': exp_data['reduction_pct_mean'].std(),
                'Retention_Mean': exp_data['retention_rate_mean'].mean(),
                'Retention_Std': exp_data['retention_rate_mean'].std()
            }
            summary_data.append(summary_row)
            
            print(f"\n{dataset} - Variant {variant}:")
            print(f"  Trials: {len(exp_data)}")
            print(f"  Baseline Accuracy: {exp_data['baseline_accuracy'].mean():.4f} ± {exp_data['baseline_accuracy'].std():.4f}")
            print(f"  Modularity Score: {exp_data['modularity_score'].mean():.4f} ± {exp_data['modularity_score'].std():.4f}")
            print(f"  Ablation Reduction: {exp_data['reduction_pct_mean'].mean():.2f}% ± {exp_data['reduction_pct_mean'].std():.2f}%")
            print(f"  Retention Rate: {exp_data['retention_rate_mean'].mean():.4f} ± {exp_data['retention_rate_mean'].std():.4f}")
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = report_dir / "summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary to: {summary_file}")
        
        # Generate LaTeX table (old ablation tables)
        generate_latex_tables(summary_df, report_dir)
        
        # Generate the 5 comprehensive NAS tables
        print("\n" + "=" * 80)
        print("GENERATING NAS TABLES")
        print("=" * 80)
        generate_5_tables(output_base, report_dir)


if __name__ == "__main__":
    main()
