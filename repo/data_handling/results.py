# import packages and set constants
import pandas as pd
import numpy as np
import argparse
from os import path, makedirs

# Constants:
PROJECT_PATH = "/Users/tpirnia/Library/CloudStorage/Dropbox/Repository/LanguageProduction/COPY_FROM_SERVER"

SUBJECT_LIST = "EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split()

classifier_dict = {
    "ridge": "Ridge",
    "log": "Logistic",
    "svm": "SVM",
    "svm_linear": "SVM Linear",
    "decisiontree": "Decision Tree",
}

column_dict = {
    "subject": "Subject",
    "classifier": "Classifier",
    "window": "Trial Length",
    "w_train": "Training Window",
    "step": "Step Size",
    "train_accuracy": "Train Accuracy",
}

exp_dict = {
    "phase": {
        "fname": "phase_accuracy_sliding_w50",
        "tg_path": "TG_phase",
        "suffix": "subject",
        "cmap": [{"face": "#F0953B", "hatch": "#F0953B"}],
        "Label": "Within Participant",
        "classifier": "Ridge",
    },
    "phase_loo": {
        "fname": "phase_accuracy_loo_w50",
        "tg_path": "TG_phase",
        "suffix": "loo",
        "cmap": [{"face": "#F0953B", "hatch": "#F0953B"}],
        "Label": "Cross-Participant",
        "classifier": "Ridge",
    },
    "US-US_w50": {
        "fname": "context_accuracy_US-US_w50",
        "tg_path": "TG_context",
        "figure_title": "Unrelated vs Semantically-related",
        "suffix": "US-US",
        "cmap": [
            {"face": "#765F1C", "hatch": "#7A6003"},
            {"face": "#FED231", "hatch": "#F5D053"},
        ],
        "Test Classification": "Within Context",
        "Train Set": "Unrelated vs Semantic",
        "Test Set": "Unrelated vs Semantic",
        "Label": r"$US \rightarrow US$",
        "within_condition": 2,
        "across_condition": 2,
        "classifier": "SVM",
    },
    "UP-UP_w50": {
        "fname": "context_accuracy_UP-UP_w50",
        "tg_path": "TG_context",
        "figure_title": "Unrelated vs Phonologically-related",
        "suffix": "UP-UP",
        "cmap": [
            {"face": "#2F624C", "hatch": "#31644E"},
            {"face": "#5CB98E", "hatch": "#5AB68B"},
        ],
        "Test Classification": "Within Context",
        "Train Set": "Unrelated vs Phonological",
        "Test Set": "Unrelated vs Phonological",
        "Label": r"$UP \rightarrow UP$",
        "within_condition": 1,
        "across_condition": 1,
        "classifier": "SVM",
    },
    "US-US": {
        "fname": "context_accuracy_US-US_w1000",
        "tg_path": "TG_context",
        "figure_title": "Unrelated vs Semantically-related",
        "suffix": "US-US",
        "cmap": [
            {"face": "#765F1C", "hatch": "#7A6003"},
            {"face": "#FED231", "hatch": "#F5D053"},
        ],
        "Test Classification": "Within Context",
        "Train Set": "Unrelated vs Semantic",
        "Test Set": "Unrelated vs Semantic",
        "Label": r"$US \rightarrow US$",
        "within_condition": 2,
        "across_condition": 2,
        "classifier": "SVM",
    },
    "UP-UP": {
        "fname": "context_accuracy_UP-UP_w1000",
        "tg_path": "TG_context",
        "figure_title": "Unrelated vs Phonologically-related",
        "suffix": "UP-UP",
        "cmap": [
            {"face": "#2F624C", "hatch": "#31644E"},
            {"face": "#5CB98E", "hatch": "#5AB68B"},
        ],
        "Test Classification": "Within Context",
        "Train Set": "Unrelated vs Phonological",
        "Test Set": "Unrelated vs Phonological",
        "Label": r"$UP \rightarrow UP$",
        "within_condition": 1,
        "across_condition": 1,
        "classifier": "SVM",
    },
    "US-UP": {
        "fname": "context_accuracy_US-UP_w1000",
        "suffix": "US-UP",
        "cmap": [
            {"face": "#765F1C", "hatch": "#7A6003"},
            {"face": "#5CB98E", "hatch": "#F5D053"},
        ],
        "Test Classification": "Cross Context",
        "Train Set": "Unrelated vs Semantic",
        "Test Set": "Unrelated vs Phonological",
        "Label": r"$US \rightarrow UP$",
        "within_condition": 2,
        "across_condition": 1,
        "classifier": "SVM",
    },
    "UP-US": {
        "fname": "context_accuracy_UP-US_w1000",
        "suffix": "UP-US",
        "cmap": [
            {"face": "#2F624C", "hatch": "#31644E"},
            {"face": "#FED231", "hatch": "#5AB68B"},
        ],
        "Test Classification": "Cross Context",
        "Train Set": "Unrelated vs Phonological",
        "Test Set": "Unrelated vs Semantic",
        "Label": r"$UP \rightarrow US$",
        "within_condition": 1,
        "across_condition": 2,
        "classifier": "SVM",
    },
    "US-CI": {
        "fname": "context_accuracy_US-CI_w1000",
        "suffix": "US-CI",
        "cmap": [
            {"face": "#765F1C", "hatch": "#7A6003"},
            {"face": "#F0953B", "hatch": "#F5D053"},
        ],
        "Test Classification": "Cross Conflict",
        "Train Set": "Unrelated vs Semantic",
        "Test Set": "Congruent vs Incongruent",
        "Label": r"$US \rightarrow CI$",
        "within_condition": 2,
        "across_condition": 3,
        "classifier": "SVM",
    },
    "UP_CI": {
        "fname": "context_accuracy_UP-CI_w1000",
        "suffix": "UP-CI",
        "cmap": [
            {"face": "#2F624C", "hatch": "#31644E"},
            {"face": "#F0953B", "hatch": "#5AB68B"},
        ],
        "Test Classification": "Cross Conflict",
        "Train Set": "Unrelated vs Phonological",
        "Test Set": "Congruent vs Incongruent",
        "Label": r"$UP \rightarrow CI$",
        "within_condition": 1,
        "across_condition": 3,
        "classifier": "SVM",
    },
}

# functions for all projects

def clean_dataframe(df, column_dict=column_dict):
    """
    This function cleans up the results csv files by doing the following:
    1. drop the following: "Unnamed: 0" column if included, empty and duplicate rows.
    2. rename columns to proper format
    3. add permutation column if not included (for observed data)
    4. Set numerical columns to numeric
    5. Sort dataframe by [Subject, Classifier, Permutation, Training Window]
    """
    # drop index column if included
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    # drop rows where all values are NaN
    df.dropna(how="all", inplace=True)

    # drop duplicate rows.
    df.drop_duplicates(inplace=True)

    if "Subject" not in df.columns:  # checking if subject column has been capitalized.
        df = relabel_columns(df, column_dict)

    if ("perm" not in df.columns) and (
        "Permutation" not in df.columns
    ):  # insert a column 'Permutation' between 'step' and 'w_train'. Set values to np.nan
        idx_to_insert = df.columns.get_loc("Training Window")
        df.insert(idx_to_insert, "Permutation", np.nan)

    df = df.apply(pd.to_numeric, errors="ignore")
    df.sort_values(
        ["Subject", "Classifier", "Permutation", "Training Window"],
        inplace=True,
        ignore_index=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def drop_pretrial(df):
    # drop rows and columns before onset.
    # for window = 500, rows where 'w_train'==1 should be kept. Only keep column t_500 -> rename to 'Test Accuracy'
    window_size = df.window.unique()
    if 1000 in window_size:
        df = df.loc[df["w_train"] == 1, :]
    elif window_size == 350:
        df = df.loc[df["w_train"] == 5, :]
    df = pd.concat([df.loc[:, :"train_accuracy"], df.loc[:, "t_500":]], axis=1).copy()
    # df = df.drop("t_0", axis=1, inplace=True)
    df.rename(columns={"t_500": "Test Accuracy"}, inplace=True)
    return df


def relabel_columns(df, column_dict):
    for k, v in column_dict.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
        else:
            print(f"{k} not in df.columns")

    #  if does not contain '_', capitalize first letter of each column name.
    df.columns = [col if "_" in col else col.title() for col in df.columns]

    if "perm" in df.columns:
        df.rename(columns={"perm": "Permutation"}, inplace=True)
    return df


def relabel_values(df, column, label_dict):
    for k in df[column].unique():
        df[column] = df[column].str.replace(k, label_dict[k])
    return df


# functions for 50ms sliding window
class sliding_window:
    def __init__(self, exp_name, project_path, window=50, step=10):
        self.window = window
        self.step = step
        self.project_path = project_path
        # map items in exp_dict to self:
        for key, value in exp_dict[exp_name].items():
            setattr(self, key, value)
        _ = self.__get_paths(project_path)
        print(
            f"Initialized {self.__class__.__name__} with window {self.window}ms and step {self.step}ms.\n\tAccuracy data: {self.fname + '.csv'}\n\tTG named: $(subject)_trained-{self.suffix}.npy\n\tNumpy files saved to: {self.out_path},\n\tFigures saved to: {self.fig_path}"
        )

    def __get_paths(self, project_path):
        self.out_path = path.join(project_path, "results", f"{self.tg_path}/")
        makedirs(self.out_path, exist_ok=True)
        self.fig_path = path.join(project_path, "results", "figures", f"{self.fname}")
        makedirs(self.fig_path, exist_ok=True)
        return self.out_path, self.fig_path

    def get_TG_from_df(self, df, subject):
        if "Training Window" not in df.columns:
            df = self.process(df)
        df = df[df["Training Window"] >= 50]
        df.sort_values(["Subject", "Training Window"], ignore_index=True, inplace=True)
        tg_data = df[df["Subject"] == subject].loc[:, "t_500":"t_1450"]
        print(f"{subject}: {tg_data.shape}")
        return tg_data

    def save_all_to_npy(self, df):
        # make sure df is sorted by w_train and only contains columns t_500 to t_1450
        SUBJECTS = df.Subject.unique()
        for subject in SUBJECTS:
            tg_data = self.get_TG_from_df(df, subject)
            self.save_subject_tg(tg_data, subject)

    def save_subject_tg(self, tg_data, subject):
        npy_fname = path.join(self.out_path, f"{subject}_trained-{self.suffix}.npy")
        np.save(npy_fname, tg_data)

    def drop_pretrial(self, df):
        df = df[df["w_train"] >= 50]
        df = pd.concat(
            [df.loc[:, :"train_accuracy"], df.loc[:, "t_500":]], axis=1
        ).copy()
        # df.drop("t_0", axis=1, inplace=True)
        return df

    def load_TG(self, subject, project_path):
        npy_fname = path.join(self.out_path, f"{subject}_trained-{ self.suffix}.npy")
        return np.load(npy_fname)

    def load_acc_df(self):
        accuracy_fname = path.join(
            self.project_path, "results", "accuracy", self.fname + ".csv"
        )
        df = pd.read_csv(accuracy_fname)
        print(
            f'+ Results loaded from {accuracy_fname.split("/")[-1]}: {df.shape[0]} rows and {len(df.columns)} columns. \n'
        )
        return df

    def process(self, df):
        df = self.drop_pretrial(df)
        df = clean_dataframe(df)
        # df = relabel_columns(df,column_dict)
        df = relabel_values(df, "Classifier", classifier_dict)
        df.sort_values(["Subject", "Training Window"], ignore_index=True, inplace=True)
        return df


# functions for 1000ms single window
class full_trial:
    def __init__(self, exp_name, project_path):
        self.exp = exp_dict[exp_name]
        self.project_path = project_path
        self.window = 1000
        self.step = 500
        # w_train to keep = 1
        # t_{start} to keep = 't_500'
        # columns to rename: w_train to Training Window, t_500 to Test Accuracy


# make class called Context_Experiment
class Context_Experiment:
    def __init__(self, df, exp_name, project_path):
        self.df = df
        for key, value in exp_dict[exp_name].items():
            setattr(self, key, value)
        _ = self.__get_paths(project_path)

    def load_acc_df(self, exp_name):
        fname = self.exp_dict[exp_name]["fname"]
        df = pd.read_csv(fname)
        df = self.set_context_experiment_labels(df)
        df = self.clean_up_labels(df)
        return df

    def set_context_experiment_labels(self, df):
        """by csv file name"""
        for k, v in self.exp_dict.items():
            df[k] = v
            df[k] = df[k].astype("category")
        return df

    def clean_up_labels(self, df, clf_dict=classifier_dict):
        # make column type numeric if possisble
        df = df.apply(pd.to_numeric, errors="ignore")
        # drop columns starting at t_0 and ending before t_500. if window = 500, t_500 is the last column and should be renamed to Test Accuracy.
        df = pd.concat([df.iloc[:, :"t_0"], df.loc[:, "t_500":]], axis=1)
        # or get index of all columns starting from t_0 and ending before t_500. if window = 500, t_500 is the last column and should be renamed to Test Accuracy.
        df = df.iloc[:, df.columns.get_loc("t_0") : df.columns.get_loc("t_500")]

        if "t_0" in df.columns:
            df = df.drop("t_0", axis=1)
        # rename columns
        if "1000" in df["window"].unique():
            if "t_0" in df.columns:
                df = df.drop("t_0", axis=1)
            if "t_500" in df.columns:
                df.rename(columns={"t_500": "Test Accuracy"}, inplace=True)

        df.rename(columns=column_dict, inplace=True)
        if 1000 in df.window.unique():
            df.rename(columns={"t_500": "Test Accuracy"}, inplace=True)

        for k in clf_dict.keys():
            df["Classifier"] = df["Classifier"].str.replace(k, clf_dict[k])

        _cattype = pd.api.types.CategoricalDtype(
            categories=list(df["Classifier"].unique()), ordered=True
        )
        df["Classifier"] = df["Classifier"].astype(_cattype)

        #  if does not contain '_', capitalize first letter of each column name.
        df.columns = [col if "_" in col else col.title() for col in df.columns]
        df.drop_duplicates(inplace=True)
        return df

    def make_accuracy_table(self, df):
        col_names = [
            col for col in df.columns if col not in ["Train Accuracy", "Test Accuracy"]
        ]
        # melt the dataframe
        df_melted = df.melt(
            id_vars=col_names, var_name="Performance", value_name="Accuracy"
        )
        _inds_tr, _inds_te = (df_melted["Performance"] == "Train Accuracy"), (
            df_melted["Performance"] == "Test Accuracy"
        )
        df_melted.loc[_inds_tr, "Performance"] = "Validation"
        df_melted.loc[_inds_te, "Performance"] = "Test"

        return df_melted


def main(experiment_name, project_path):
    if experiment_name in ['phase','phase_loo','US-US_w50','UP-UP_w50']:
        experiment = sliding_window(
            experiment_name, window=50, step=10, project_path=project_path
        )
        experiment_df = experiment.load_acc_df()
        experiment_df = experiment.process(experiment_df)
        experiment.save_all_to_npy(experiment_df)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment', choices=['phase','phase_loo','US-US_w50','UP-UP_w50','US-US', 'UP-UP','US-UP','UP-US','US-CI','UP-CI'], help='Type of experiment')
    parser.add_argument('-project_path', default='/project_data/volume0/tpirnia/thesis_repository')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get parameters for main()
    args = parse_arguments()
    main(args.experiment, args.project_path)