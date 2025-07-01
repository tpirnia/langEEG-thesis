import os
from typing import Dict, List, Tuple
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from itertools import combinations

# Configuration constants
PROJECT_PATH = '/project_data/volume0/tpirnia/thesis_repository'
DEFAULT_CLASSIFIERS = ['Ridge Regression', 'Linear SVM', 'Decision Tree']
COL_RENAMES = {
    'subject': 'Subject',
    'classifier': 'Classifier', 
    'window': 'Trial Length',
    'train_accuracy': 'Train Accuracy', 
    't_500': 'Test Accuracy'
}
CLASSIFIER_NAMES = {
    'ridge': 'Ridge Regression',
    'svm': 'SVM',
    'decisiontree': 'Decision Tree',
    'svm_linear': 'Linear SVM',
    'log': 'Logistic Regression'
}
def keep_common_columns(dfs):
    """Keep only common columns in each dataframe, maintaining the order."""
    common_columns = set.intersection(*[set(df.columns) for df in dfs])
    original_order = [col for col in dfs[0].columns if col in common_columns]
    dfs = [df[original_order] for df in dfs]
    df = pd.concat(dfs, ignore_index=True)
    return df

def load_and_prepare_data(results_files: List[str], exclude_subjects: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and prepare the classification results data from multiple files.
    
    Args:
        results_files: List of paths to the CSV files containing results
        exclude_subjects: List of subject IDs to exclude
    
    Returns:
        Tuple of (prepared DataFrame, classifier name mappings)
    """
    dfs = []
    for results_file in results_files:
        df = pd.read_csv(results_file)
        dfs.append(df)
    
    # Concatenate all dataframes
    # df = pd.concat(dfs, ignore_index=True)
    df = keep_common_columns(dfs)
    
    # Continue with the existing preparation steps
    df = (df.rename(columns=COL_RENAMES)
          .query('Subject not in @exclude_subjects')
          .reset_index(drop=True))
    
    df['Classifier'] = (df['Classifier']
                       .replace(CLASSIFIER_NAMES)
                       .astype(pd.CategoricalDtype(categories=list(CLASSIFIER_NAMES.values()),
                                                 ordered=True)))
    
    df['Subject'] = pd.Categorical(df['Subject'],
                                 categories=df['Subject'].unique(),
                                 ordered=True)
    
    return df, CLASSIFIER_NAMES

def prepare_performance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data for plotting performance metrics.
    """
    id_cols = ['Subject', 'Classifier', 'Trial Length', 'step', 'w_train']
    
    return (df.melt(id_vars=id_cols,
                    var_name='Performance',
                    value_name='Accuracy')
            .replace({'Performance': {'Train Accuracy': 'Train',
                                    'Test Accuracy': 'Test'}}))

def create_accuracy_plot(df: pd.DataFrame, 
                        output_path: str, 
                        figure_name: str, 
                        classifiers: List[str] = None) -> None:
    """
    Create and save the accuracy bar plot.
    """
    plt.rcParams.update(plt.rcParamsDefault)
    classifiers = classifiers or DEFAULT_CLASSIFIERS
    
    plot_df = df[df['Classifier'].isin(classifiers)].copy()
    plot_df['Classifier'] = pd.Categorical(plot_df['Classifier'],
                                         categories=classifiers,
                                         ordered=True)
    
    plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-poster'])
    fig, ax = plt.subplots(figsize=(8.7, 5.27))
    
    sns.barplot(y='Accuracy',
                x='Classifier',
                hue='Performance',
                data=plot_df,
                ax=ax, 
                err_kws={'linewidth': 3},
                errorbar=('ci', 95))
    
    locs, _ = plt.yticks()
    ax.xaxis.labelpad = 17
    ax.set(ylabel='Accuracy (%)', 
           ylim=[0,1],
           yticks=locs, 
           yticklabels=[int(loc*100) for loc in locs])
    plt.gca().legend().set_title('')
    
    plt.savefig(os.path.join(output_path, figure_name),
                dpi=450, 
                bbox_inches="tight")
    # plt.show()

def create_window_plot(df: pd.DataFrame, 
                        output_path: str, 
                        figure_name: str, 
                        classifiers: List[str] = None) -> None:
    """
    Create and save the accuracy bar plot comparing two window sizes
    """
    # Train and test Accuracy per window length (full trial and 350ms)
    plt.rcParams.update(plt.rcParamsDefault)
    classifiers = ['Ridge Regression']
    plot_df = df[df['Classifier'].isin(classifiers)].copy()
    plot_df['Classifier'] = pd.Categorical(plot_df['Classifier'],
                                         categories=classifiers,
                                         ordered=True)

    plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-poster'])
    _figdims = (8.7, 5.27)
    _fig, _ax = plt.subplots(figsize=_figdims)
    _g = sns.barplot(y='Accuracy',
                     x='Trial Length',
                     hue='Performance',
                     data=plot_df,
                     ax=_ax, 
                     err_kws={'linewidth': 3},
                     palette={'Train': '#003B6F', 'Test': '#FF8C00'},
                     errorbar=('ci', 95))
    locs= np.linspace(0, 1, 6)#plt.yticks()
    xticks = ['$350$','$1000$']
    xlocs=_ax.get_xticks()
    _ax.set(ylabel='Accuracy (%)', 
            yticks=locs, 
            xlabel='Epoch length (ms)',
            ylim=[0,1],
            yticklabels=list(map(int,locs*100)),
            xticks=xlocs,
            xticklabels=xticks)
    plt.gca().legend().set_title('')
    _ax.xaxis.labelpad = 17
    plt.savefig(os.path.join(output_path, figure_name),
                dpi=450, 
                bbox_inches="tight")
    # plt.show()

def perform_paired_ttests(main_df: pd.DataFrame, classifiers: List[str] = None) -> None:
    """
    Perform and print results of paired t-tests between classifiers.
    """
    classifiers = classifiers or DEFAULT_CLASSIFIERS
    masks = {clf: (main_df['Classifier'] == clf) for clf in classifiers}
    get_acc = lambda clf: main_df.loc[masks[clf], 'Test Accuracy'].values
    
    pairs = list(combinations(classifiers, 2))
    results = {f"{c1} v. {c2}": ttest_rel(get_acc(c1), get_acc(c2)) 
              if len(get_acc(c1)) == len(get_acc(c2)) else (np.nan, np.nan)
              for c1, c2 in pairs}

    for pair, (t_stat, p_val) in results.items():
        print(f"{pair}: [{t_stat:.2f}, {p_val:.10f}]")
def perform_paired_window_ttests(main_df: pd.DataFrame, classifier: str = "Ridge Regression") -> None:
    """
    Perform and print results of paired t-tests between Trial Length groups for a specified classifier.
    
    Args:
        main_df: DataFrame containing the results
        classifier: Classifier to analyze (defaults to "Ridge Regression")
    """
    # Filter for the specified classifier
    df_subset = main_df[main_df['Classifier'] == classifier]
    
    # Get unique Trial Lengths
    trial_lengths = df_subset['Trial Length'].unique()
    
    # Create masks for each Trial Length
    masks = {length: (df_subset['Trial Length'] == length) for length in trial_lengths}
    get_acc = lambda length: df_subset.loc[masks[length], 'Test Accuracy'].values
    
    # Get all pairs of Trial Lengths
    pairs = list(combinations(trial_lengths, 2))
    
    # Perform t-tests
    results = {f"{t1} v. {t2}": ttest_rel(get_acc(t1), get_acc(t2)) 
              if len(get_acc(t1)) == len(get_acc(t2)) else (np.nan, np.nan)
              for t1, t2 in pairs}

    # Print results
    print(f"\nT-test results for {classifier}:")
    for pair, (t_stat, p_val) in results.items():
        print(f"{pair}: [{t_stat:.2f}, {p_val:.10f}]")  
        
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process EEG classification results')
    parser.add_argument('-results_files', type=str, nargs='+',
                      required=True,
                      help='Names of results files (at least one required)')
    parser.add_argument('-figure_name', type=str,
                      default='phase_classification_accuracy_w1000-w350',
                      help='Name for output figure (without extension)')
    parser.add_argument('-ttest', action='store_true',
                      help='Run paired t-tests between two time windows')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Configuration
    results_path = os.path.join(PROJECT_PATH, 'results')
    results_files = [os.path.join(results_path, fname) for fname in args.results_files]
    exclude_subjects = ['EP24']
    
    # Process data and generate outputs
    main_df, _ = load_and_prepare_data(results_files, exclude_subjects)
    performance_df = prepare_performance_data(main_df)
    
    # Only run t-tests if flag is provided
    if args.ttest:
        perform_paired_window_ttests(main_df)
        
    create_window_plot(performance_df, 
                        results_path,
                        f"{args.figure_name}.png")

if __name__ == "__main__":
    main()