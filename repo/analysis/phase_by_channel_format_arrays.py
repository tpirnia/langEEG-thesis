import sys
import os
import argparse
import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Reformat null results")
    parser.add_argument('-s', '--subject', required=True, 
                       help='Subject ID (e.g., EP06)')
    parser.add_argument('-w', '--window', type=int, default=1000, help='Window size in ms (default=1000)')
    parser.add_argument(
        '--project_path', default='/project_data/volume0/tpirnia/thesis_repository'
    )
    parser.add_argument('--is_perm', action='store_true', help="whether to format null results or observed ones."
    )
    parser.add_argument('--t_start', type=int,default=1)
    return parser.parse_args()

def reformat_obs(args,bad_ch):
   # Setup paths
    project_path = args.project_path
    subject = args.subject
    w = args.window
    out_path = os.path.join(
        project_path, f"results/phase_accuracy_by_channel_w{w}/obs/"
    )
    os.makedirs(out_path, exist_ok=True)
    
    # Load and preprocess data
    df_fname = os.path.join(
        project_path,
        f"results/phase_accuracy_by_channel_w{w}/{subject}_ridge_accuracy.csv",
    )
    df_ = pd.read_csv(df_fname)

    df_.drop_duplicates(inplace=True)
    df_.reset_index(inplace=True,drop=True)
    df_filt = df_[(df_['w_train']>= args.t_start)]
    df_filt.drop_duplicates(subset=['fold','channel','w_train'],inplace=True, ignore_index=True)

    df_filt.drop(columns=['window','step','classifier','w_train','t_0'],inplace=True)

    df_filt.head()

    df_filt.sort_values(by=['channel','fold'],inplace=True, ignore_index=True)

    folds = df_filt.fold.unique()
    n_folds = len(folds)

    n_channels = 128 + 1 # for index 0
    chs = np.arange(n_channels)
    chs_inds = chs[np.isin(chs, bad_ch,invert=True)]

    arr_out = np.full(shape=(n_channels,n_folds, 1), fill_value=np.nan)
    arr_tr_out = np.full(shape=(n_channels,n_folds, 1), fill_value=np.nan)

    
    for ch_i, ch in enumerate(chs_inds): # actual_ch_id = chs_inds[ch_i] (aka ch)
        arr_out[ch,:,:] = df_filt.loc[(df_filt['channel']==ch_i),'t_500':].values
        arr_tr_out[ch,:,:] = (df_filt.loc[(df_filt['channel']==ch_i),'train_accuracy'].values)[:,np.newaxis]

    # Save results
    np.save(os.path.join(out_path, f"{subject}_by_channel_test.npy"), arr_out)
    np.save(os.path.join(out_path, f"{subject}_by_channel_train.npy"), arr_tr_out)

def reformat_null(args,bad_ch):
    project_path = args.project_path
    subject = args.subject
    w = args.window
    
    out_path = os.path.join(
        project_path, f"results/phase_accuracy_by_channel_w{w}/null/"
    )
    os.makedirs(out_path, exist_ok=True)

    
    # Load and preprocess data
    df_fname = os.path.join(
        project_path,
        f"results/permutations/phase_accuracy_by_channel_w{w}/{subject}_ridge_accuracy.csv",
    )
    df_ = pd.read_csv(df_fname)
    df_.drop_duplicates(inplace=True)
    df_.reset_index(inplace=True, drop=True)
    df_.rename(columns={"perm": "permutation"}, inplace=True)
    # Filter data
    df_filt = df_[(df_["w_train"] >= args.t_start)].copy()

    df_filt.drop_duplicates(
        subset=["permutation", "fold", "channel", "w_train"],
        inplace=True,
        ignore_index=True,
    )

    # Drop unnecessary columns
    df_filt.drop(columns=['window',
                          'step',
                          'classifier',
                          'w_train',
                          't_0'],inplace=True)
    df_filt.sort_values(
        by=["permutation", "channel", "fold"], inplace=True, ignore_index=True
    )
    del df_

    # Setup arrays
    folds = df_filt.fold.unique()
    n_folds = len(folds)
    n_channels = 128 + 1
    chs = np.arange(n_channels)
    chs_inds = chs[np.isin(chs, bad_ch,invert=True)]
    perms = df_filt['permutation'].unique()
    n_perms = len(perms)
    
    arr_out = np.full(shape=(n_perms, n_channels, n_folds, 1), fill_value=np.nan)
    arr_tr_out = np.full(shape=(n_perms, n_channels, n_folds, 1), fill_value=np.nan)

    for p_i, p in enumerate(perms): 
        for f in folds:
            subset = df_filt[(df_filt['permutation']==p) & (df_filt['fold']==f)]
            if len(subset) == len(chs_inds):
                arr_out[p_i, chs_inds,f,:] = subset.loc[:,'t_500':].values
                arr_tr_out[p_i, chs_inds,f,:] = subset['train_accuracy'].values[:,np.newaxis]
            else:
                print(f"Warning: Expected {len(chs_inds)} channels, got {len(subset)} for perm {p}, fold {f}")
    # Save results
    np.save(os.path.join(out_path, f"{subject}_by_channel_test.npy"), arr_out)
    np.save(os.path.join(out_path, f"{subject}_by_channel_train.npy"), arr_tr_out)

    print(f"{subject}\t{n_perms}")

def main():
    args = parse_arguments()
    
    # Import and get bad channels
    sys.path.append(os.path.join(args.project_path, 'repo'))
    from data_handling.preprocessing import get_bad_channels
    
    bad_ch = get_bad_channels(args.project_path, args.subject)
    
    if args.is_perm:
        reformat_null(args, bad_ch)
    else:
        reformat_obs(args, bad_ch)
    
if __name__ == "__main__":
    main()
