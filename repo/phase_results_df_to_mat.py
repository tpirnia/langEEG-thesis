import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# Constants
WINDOW, STEP = 50, 10
RESULTS_PATH = '/project_data/volume0/tpirnia/thesis_repository/results'
SUBJECT_LIST = "EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-trained', choices=['subject', 'loo'], help='Type of training')
    parser.add_argument('-s', '--subject', help='Subject ID')
    args = parser.parse_args()

    # Create output directory
    out_dir = f"{RESULTS_PATH}/TG/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # File paths
    within_fname = f"{RESULTS_PATH}/phase_accuracy_sliding_w{WINDOW}.csv"
    across_fname = f"{RESULTS_PATH}/phase_accuracy_loo_w{WINDOW}.csv"

    if args.trained:
        # Process specific training type and optional subject
        fname = within_fname if args.trained == 'subject' else across_fname
        process_and_save_data(fname, args.trained, args.subject)
    else:
        # Process both within and across participant models for all subjects
        process_and_save_data(within_fname, "subject")
        process_and_save_data(across_fname, "loo")