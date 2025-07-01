# formatting of classification output into format for statistical analysis and/or plotting.
import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse




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

        
def clean_dataframe(df):
    # remove index column if included
    # drop duplicate rows.
    df.drop_duplicates(inplace=True)
    return df