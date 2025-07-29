# %%
import pandas as pd
import numpy as np
import argparse
from os import path,makedirs
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from matplotlib import colors
import matplotlib as mpl


import seaborn as sns


acc_dict = {
    "context_accuracy_US-US_w1000.csv": {
        "Test Classification": "Within Context",
        "Train Set": "Unrelated vs Semantic",
        "Test Set": "Unrelated vs Semantic",
        "Label": r"$US \rightarrow US$",
        "figure_title": "Unrelated vs Semantically-related",
    },
    "context_accuracy_UP-UP_w1000.csv": {
        "Test Classification": "Within Context",
        "Train Set": "Unrelated vs Phonological",
        "Test Set": "Unrelated vs Phonological",
        "Label": r"$UP \rightarrow UP$",
        "figure_title": "Unrelated vs Phonologically-related",
    },
    "context_accuracy_US-UP_w1000.csv": {
        "Test Classification": "Cross Context",
        "Train Set": "Unrelated vs Semantic",
        "Test Set": "Unrelated vs Phonological",
        "Label": r"$US \rightarrow UP$",
    },
    "context_accuracy_UP-US_w1000.csv": {
        "Test Classification": "Cross Context",
        "Train Set": "Unrelated vs Phonological",
        "Test Set": "Unrelated vs Semantic",
        "Label": r"$UP \rightarrow US$",
    },
    "context_accuracy_US-CI_w1000.csv": {
        "Test Classification": "Cross Conflict",
        "Train Set": "Unrelated vs Semantic",
        "Test Set": "Congruent vs Incongruent",
        "Label": r"$US \rightarrow CI$",
    },
    "context_accuracy_UP-CI_w1000.csv": {
        "Test Classification": "Cross Conflict",
        "Train Set": "Unrelated vs Phonological",
        "Test Set": "Congruent vs Incongruent",
        "Label": r"$UP \rightarrow CI$",
    },
}

clf_dict = {'ridge':'Ridge','log':'Logistic','svm':'SVM','svm_linear':'SVM Linear','decisiontree':'Decision Tree'}

bar_color_scheme = {'US':[{'face':'#765F1C','hatch': '#7A6003'},{'face':'#FED231','hatch': '#F5D053'}],
              'UP':[{'face':'#2F624C','hatch': '#31644E'},{'face':'#5CB98E','hatch': '#5AB68B'}]}

def update_experiment_labels(df,exp_dict):
    for k,v in exp_dict.items():
        df[k] = v
        df[k] = df[k].astype('category')
    return df


def clean_up_labels(df, clf_dict=None):
    if 't_0' in df.columns: df = df.drop('t_0', axis=1)
    # rename columns
    col_renames={'subject': 'Subject',
                 'classifier':'Classifier', 
                 'window': 'Trial Length',
                 'train_accuracy':'Train Accuracy', 
                 't_500': 'Test Accuracy',
                 'step': 'Step Size',
                 'w_train': 'Training Window'
                }
    df.rename(columns=col_renames,
               inplace=True)
    if clf_dict is None:
        clf_dict = {'ridge':'Ridge','log':'Logistic','svm':'SVM','svm_linear':'SVM Linear','decisiontree':'Decision Tree'}

    for k in clf_dict.keys():
            df['Classifier'] = df['Classifier'].str.replace(k,clf_dict[k])

    _cattype = pd.api.types.CategoricalDtype(categories=list(df['Classifier'].unique()), ordered=True)
    df['Classifier'] = df['Classifier'].astype(_cattype)
    
    #  if does not contain '_', capitalize first letter of each column name.
    df.columns = [col if '_' in col else col.title() for col in df.columns]
    df.drop_duplicates(inplace=True)
    return df

def make_accuracy_table(df):
    col_names = [col for col in df.columns if col not in ['Train Accuracy', 'Test Accuracy']]
    # melt the dataframe
    df_melted = df.melt(id_vars=col_names,
                         var_name='Performance',
                         value_name='Accuracy')
    _inds_tr, _inds_te = (df_melted['Performance'] == 'Train Accuracy'),(df_melted['Performance'] == 'Test Accuracy')
    df_melted.loc[_inds_tr,'Performance'] = 'Validation'
    df_melted.loc[_inds_te,'Performance'] = 'Test'

    return df_melted

def make_context_figure(df, color_scheme,_ax):
    _g = sns.barplot(y='Accuracy',
                     x='Classifier',
                     hue='Performance',
                     data=df,
                     ax=_ax, 
                     errwidth=3,
                     errorbar=('ci', 95))
    
    cols = color_scheme
    
    # sort patches by x-position
    patches_sorted = sorted(_ax.patches, key=lambda p: p.get_x())
    
    for i, bar in enumerate(patches_sorted):
        hatch = '//'
        bar.set_hatch(hatch)
        bar._hatch_color = colors.to_rgba(cols[i]['hatch'], alpha=1) 
        bar.set_facecolor(colors.to_rgba(cols[i]['face'], alpha=1))
    
    locs, _ = plt.yticks()
    _ax.xaxis.labelpad = 17
    _ax.set(ylabel='Accuracy (%)', 
            yticks=locs, 
            yticklabels=list(map(int,locs*100)),
            xlabel='Classifier')
    plt.gca().legend().set_title('')
    return _ax

def main(fname_root, color_scheme, project_path, exclude_subjects=['EP24'],return_dfs=False):
    # set paths & names
    fig_name = path.join(project_path, 'results','figures', f'Context-Figure1_{fname_root}_by_classifier.png')
    makedirs(path.dirname(fig_name), exist_ok=True)

    train_label = fname_root.split('-')[0][-2:]
    exp_title = acc_dict[fname_root+'.csv']['figure_title']
    fig_title = f'Performance by classifier of {exp_title} stimuli'
    # test_label = fname_root.split('-')[1][:2]

    # load data
    main_df = pd.read_csv(path.join(project_path, 'results','accuracy', fname_root+'.csv'))
    main_df = main_df[~main_df['subject'].isin(exclude_subjects)]
    main_df = clean_up_labels(main_df)
    main_df.head()
    df = make_accuracy_table(main_df)
    # df.head()
    plot_figure(df, train_label, fig_name, fig_title)
    if return_dfs:
        return main_df, df


def plot_figure(df, train_label, fig_name, fig_title):
    # PLOT
    cols = bar_color_scheme[train_label]* len(df['Classifier'].unique())
    params = {'lines.markersize':6,
        'axes.labelsize': 20,
        'axes.titlesize':20,
        'axes.labelpad':8,
        'xtick.labelsize' :16,
        'ytick.labelsize' :16,
        'xtick.major.pad':7, 
        'ytick.major.pad':6, 
        'legend.labelspacing':.5,
        'legend.handletextpad':.175,
        'legend.title_fontsize':20,
        'legend.fontsize':20,
        'figure.figsize':(8,10),
        'figure.subplot.hspace': .95,
        'figure.subplot.wspace': .95,
        'image.origin': 'lower',
        'figure.autolayout': False,
        'scatter.marker':'o',
        'savefig.dpi': 450,
        'savefig.bbox': 'tight',
        'savefig.transparent': True,
        }
    with mpl.rc_context(params):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.style.use('seaborn-v0_8-poster')

        _figdims = (8.7, 5.27)
        _fig, _ax = plt.subplots(figsize=_figdims)
        plt.ylim(0,1)
        _ax = make_context_figure(df, cols,_ax)
        plt.gca().set_title(fig_title, pad=15)
        plt.savefig(fig_name, dpi=450, bbox_inches='tight', transparent=True)
        plt.close()

def parse_arguments():
    """Parse command line arguments."""
    description = ('Results are saved to png in project_path/results/figures.\n'
    'To run with default arguments, run: python context_plot_within_context_by_classifier.py\n'
    'otherwise:\n'
    'python repo/context_plot_within_context_by_classifier.py -f accuracy/context_accuracy_w1000 -project_path path/to/repository')

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                   description=description)
    parser.add_argument('-f','--fname_root', type=str, default='context_accuracy_US-US_w1000')
    parser.add_argument('--project_path', type=str, default='/project_data/volume0/tpirnia/data/Thesis_Repository/')
    parser.add_argument('--exclude', type=list, default=['EP24'])
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    main(args.fname_root, bar_color_scheme, args.project_path, args.exclude)
