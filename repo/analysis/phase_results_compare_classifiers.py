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
DEFAULT_CLASSIFIERS = ['Ridge Regression', 'Linear SVM', 'SVM','Decision Tree']
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

def load_and_prepare_data(results_file: str, exclude_subjects: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and prepare the classification results data.
    
    Args:
        results_file: Path to the CSV file containing results
        exclude_subjects: List of subject IDs to exclude
    
    Returns:
        Tuple of (prepared DataFrame, classifier name mappings)
    """
    df = (pd.read_csv(results_file)
          .rename(columns=COL_RENAMES)
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
    
    n_locs = len(classifiers)*2
    plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-poster'])
    fig, ax = plt.subplots(figsize=(8.7, 5.27))
    
    sns.barplot(y='Accuracy',
                x='Classifier',
                hue='Performance',
                data=plot_df,
                ax=ax, 
                err_kws={'linewidth': 2},#errwidth=2,
                palette={'Train': '#003B6F', 'Test': '#FF8C00'},
                errorbar=('ci', 95))
    
    locs = np.linspace(0, 1, n_locs) #was (0,1,6)
    ax.xaxis.labelpad = 17
    ax.set(ylabel='Accuracy (%)', 
           ylim=[0,1],
           yticks=locs, 
           yticklabels=list(map(int,locs*100)))#[int(loc*100) for loc in locs])
    plt.gca().legend().set_title('')
    
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
    print(main_df)
    for pair, (t_stat, p_val) in results.items():
        print(f"{pair}: [{t_stat:.2f}, {p_val:.10f}]")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Process EEG classification results')
    parser.add_argument('-results_file', type=str, 
                      default='phase_accuracy_w1000.csv',
                      help='Name of results file')
    parser.add_argument('-figure_name', type=str,
                      default='phase_classification_accuracy_w1000-barplot',
                      help='Name for output figure (without extension)')
    parser.add_argument('-ttest', action='store_true',
                      help='Run paired t-tests between classifiers')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Configuration
    results_path = os.path.join(PROJECT_PATH, 'results')
    results_file = os.path.join(results_path, args.results_file)
    exclude_subjects = ['EP24']
    
    # Process data and generate outputs
    main_df, _ = load_and_prepare_data(results_file, exclude_subjects)
    performance_df = prepare_performance_data(main_df)
    
    # Only run t-tests if flag is provided
    if args.ttest:
        perform_paired_ttests(main_df)
        
    create_accuracy_plot(performance_df, 
                        results_path,
                        f"{args.figure_name}.png")

if __name__ == "__main__":
    main()