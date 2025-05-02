import ast
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.pyplot as plt

# ===========================
# Example Usage:
# python resid_analysis/run_analysis.py --run_id tubg3tim --group_by "Attention-Deficit/Hyperactivity Disorder" --start_frame 300 --end_frame 400 --data test
# ===========================

# CLI Setup
parser = argparse.ArgumentParser(description="Display information on subject-specific residuals. \
                                 Color on whether subject is healthy or unhealthy, i.e. diagnosed with ASD etc.")
parser.add_argument('--run_id',
                    type=str,
                    required=True,
                    help='Run ID (e.g., cvy8kv4o), should contain a csv file in results_each/<run_id>/residuals.csv')
parser.add_argument('--start_frame',
                    type=int,
                    default=0,
                    help='Start frame for the sequence (default: 0)')
parser.add_argument('--end_frame',
                    type=int,
                    default=-1,
                    help='End frame for the sequence (default: -1, which means the last frame)')
parser.add_argument('--specify_emotion',
                    type=str,
                    default=None,
                    help='Specify a single emotion to plot (e.g., "Anger"). If not specified, all emotions will be plotted.')
parser.add_argument('--group_by',
                    type=str,
                    default="Sex",
                    choices=['Sex', 'SITE', 'n_diagnoses', 'Attention-Deficit/Hyperactivity Disorder', 'Autism Spectrum Disorder', 'Specific Learning Disorder'],
                    help='Group by specified attribute')
parser.add_argument('--data',
                    type=str,
                    required=True,
                    choices=['train', 'val', 'test'],
                    help='Data type to load (train, val, test)')
args_cli = parser.parse_args()

RUN_ID = args_cli.run_id
START_FRAME = args_cli.start_frame
END_FRAME = args_cli.end_frame
SPECIFY_EMOTION = args_cli.specify_emotion
GROUP_BY = args_cli.group_by
DATA_SPLIT = args_cli.data

EMOTION_LABELS = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
if SPECIFY_EMOTION:
    EMOTION_IDX = EMOTION_LABELS.index(SPECIFY_EMOTION)
    
PROJECT_ROOT = Path("/home/patrickstyll/Bachelorstudiengang-Software_and_Information_Engineering/SNU_Connectome_Lab/SwiFT-IO")
DATA_CSV = PROJECT_ROOT / "results_each" / RUN_ID / f"residuals_{DATA_SPLIT}.csv"
METADATA_CSV = PROJECT_ROOT / "split_fixed_1_test.w.Dx.csv"

# load data
data = pd.read_csv(DATA_CSV)
subjects = data['subject'].unique()
print(f"Found {len(subjects)} unique subjects in the data.")

# load metadata
metadata = pd.read_csv(METADATA_CSV)
metadata = metadata[metadata['SUBJECT_ID'].isin(subjects)]
print(f"Found {len(metadata)} unique subjects in the metadata.")

# filter data to only include subjects in metadata
subjects = [subject for subject in subjects if subject in metadata['SUBJECT_ID'].values]
data = data[data['subject'].isin(metadata['SUBJECT_ID'])]
print(f"Filtered data to {len(data)} subjects based on metadata.")

# make sure that we compare <certain_diagnosis> vs "No Diagnosis"
if GROUP_BY not in ['Sex', 'SITE', 'n_diagnoses']:
    metadata = metadata[(metadata[GROUP_BY] == 1) | ((metadata[GROUP_BY] == 0) & (metadata['No Diagnosis'] == 1))]
    subjects = metadata['SUBJECT_ID'].unique()
    data = data[data['subject'].isin(subjects)]
    print(f"Filtered metadata to {len(metadata)} subjects based on {GROUP_BY}.")

# add unique attributes
UNIQUE_ATTRS = metadata[GROUP_BY].unique()
palette = sns.color_palette("hls", len(UNIQUE_ATTRS))
COLORS = ListedColormap(palette)
COLOR_MAP = {attr: COLORS(i) for i, attr in enumerate(UNIQUE_ATTRS)}

# retrieve actual end frame
if END_FRAME == -1:
    END_FRAME = data['end_frame'].max() + 1
    print(f"WARNING: Setting end frame to {END_FRAME} based on data.")

def retrieve_list_from_string(string: str) -> list:
    parsed_lists = [ast.literal_eval(s) for s in string]
    return np.concatenate(parsed_lists)

# knit together all subjects
knitted_data = pd.DataFrame()
for subject in tqdm(subjects, desc="Knitting subject sequences", unit="subject"):
    knitted_subject = {}
    subject_data = data[data['subject'] == subject]
    subject_data = subject_data.sort_values(by='start_frame')
    skip_subject = False
    if SPECIFY_EMOTION:
        resids = retrieve_list_from_string(subject_data[f"residual_{EMOTION_IDX}_{SPECIFY_EMOTION}"].values)[START_FRAME:END_FRAME]
        preds = retrieve_list_from_string(subject_data[f"prediction_{EMOTION_IDX}_{SPECIFY_EMOTION}"].values)[START_FRAME:END_FRAME]
        if len(resids) == END_FRAME - START_FRAME:
            knitted_subject[f"residual_{EMOTION_IDX}_{SPECIFY_EMOTION}"] = resids
            knitted_subject[f"prediction_{EMOTION_IDX}_{SPECIFY_EMOTION}"] = preds
        else:
            skip_subject = True
    else:
        for i, e in enumerate(EMOTION_LABELS):
            resids = retrieve_list_from_string(subject_data[f"residual_{i}_{e}"].values)[START_FRAME:END_FRAME]
            preds = retrieve_list_from_string(subject_data[f"prediction_{i}_{e}"].values)[START_FRAME:END_FRAME]
            if len(resids) == END_FRAME - START_FRAME:
                knitted_subject[f"residual_{i}_{e}"] = resids
                knitted_subject[f"prediction_{i}_{e}"] = preds
            else:
                skip_subject = True
    if not skip_subject:
        knitted_subject["subject"] = subject
        knitted_subject["attr"] = metadata[metadata['SUBJECT_ID'] == subject][GROUP_BY].values[0]
        knitted_data = pd.concat([knitted_data, pd.DataFrame([knitted_subject])], ignore_index=True)
    else:
        print(f"WARNING: Skipping subject {subject} due to inconsistent data length.")

# plot residuals of all subjects against time
time = torch.arange(START_FRAME, END_FRAME, 1)

def plot_resids_against_time(knitted_data: pd.DataFrame,
                             emotion_idx: int,
                             emotion_label: str) -> None:
    plt.figure(figsize=(12, 6))
    for _, data in knitted_data.iterrows():
        subj = data["subject"]
        residuals = data[f"residual_{emotion_idx}_{emotion_label}"]
        plt.plot(time, residuals, label=f"Subject {subj}", alpha=0.5, color=COLOR_MAP[data["attr"]])
    plt.xlabel("Time")
    plt.ylabel("Residual Value")
    plt.title(f"Residuals of {emotion_label}")
    handles = [plt.Line2D([0], [0], color=COLOR_MAP[attr], lw=2, label=attr) for attr in UNIQUE_ATTRS]
    plt.legend(handles=handles, title=f"{GROUP_BY} Attributes", loc="upper right",)
    plt.grid(True)
    plt.show()

if SPECIFY_EMOTION:
    plot_resids_against_time(knitted_data, EMOTION_IDX, SPECIFY_EMOTION)
else:
    for i, e in enumerate(EMOTION_LABELS):
        plot_resids_against_time(knitted_data, i, e)
        
# plot mean residuals via density plot

def plot_mean_residuals(knitted_data: pd.DataFrame,
                        emotion_idx: int,
                        emotion_label: str) -> None:
    plt.figure(figsize=(12, 6))
    for attr in UNIQUE_ATTRS:
        attr_data = knitted_data[knitted_data["attr"] == attr]
        mean_residuals = attr_data[f"residual_{emotion_idx}_{emotion_label}"].mean()
        plt.hist(mean_residuals, bins=50, density=True, alpha=0.5, label=f"Attr: {attr}", color=COLOR_MAP[attr])
    plt.xlabel("Mean Residual Value")
    plt.ylabel("Density")
    plt.title(f"Mean Residuals of {emotion_label} by Attribute")
    plt.legend(title=f"{GROUP_BY} Attributes", loc="upper right")
    plt.grid(True)
    plt.show()

if SPECIFY_EMOTION:
    plot_mean_residuals(knitted_data, EMOTION_IDX, SPECIFY_EMOTION)
else:
    for i, e in enumerate(EMOTION_LABELS):
        plot_mean_residuals(knitted_data, i, e)

print("Done!")