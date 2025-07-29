import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# Constants
WINDOW, STEP = 50, 10
# RESULTS_PATH = '/project_data/volume0/tpirnia/thesis_repository/results'
SUBJECT_LIST = "EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split()
exp_types = {
    'US-US': 'context',
    'UP-UP': 'context',
    'phase': 'phase',
    'phase_loo': 'phase'
}

exp_suffs = {
    'US-US': 'US-US',
    'UP-UP': 'UP-UP',
    'phase': 'subject',
    'phase_loo': 'loo'
}
def process_and_save_data(input_fname, trained_type, subject=None):
    df = pd.read_csv(input_fname)
    df = df[df['w_train'] >= 50]
    df = (df.groupby(["w_train", "subject"], as_index=False)
         .mean(numeric_only=True)
         .drop_duplicates()
         .sort_values(["subject", "w_train"], ignore_index=True))
    subjects_to_process = [subject] if subject else SUBJECT_LIST
    for subj in subjects_to_process:
        if subj not in SUBJECT_LIST:
            raise ValueError(f"Subject {subj} not in subject list")
        time_data = df[df['subject'] == subj].loc[:, 't_500':'t_1450']
        print(f"{subj}: {time_data.shape}")
        np.save(f"{out_dir}{subj}_trained-{trained_type}.npy", time_data)

if __name__ == "__main__":
    # python repo/context_within_results_df_to_mat.py -experiment US-US -s EP03
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment', choices=['US-US', 'UP-UP','phase','phase_loo'], help='Type of training')
    parser.add_argument('-s', '--subject', help='Subject ID')
    parser.add_argument('--results_path', default='/project_data/volume0/tpirnia/thesis_repository/results')
    args = parser.parse_args()

    EXP = args.experiment
    RESULTS_PATH = args.results_path
    exp_type = exp_types[EXP]
    exp_suffix = exp_suffs[EXP]


    # Create output directory
    out_dir = f"{RESULTS_PATH}/TG_{exp_type}/"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    # File paths
    exp_fname = f"{RESULTS_PATH}/accuracy/{exp_type}_accuracy_{exp_suffix}_w{WINDOW}.csv"

    if args.subject:
        process_and_save_data(exp_fname, exp_suffix, args.subject)
    else:
        process_and_save_data(exp_fname, exp_suffix)

    print("Done")