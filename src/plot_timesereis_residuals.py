import ast
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# ===========================
# Example Usage:
# python resid_analysis/run_analysis.py --run_id tubg3tim
# ===========================

# CLI Setup
parser = argparse.ArgumentParser(description="Extract residuals from model predictions and save with frame info.")
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
args_cli = parser.parse_args()

RUN_ID = args_cli.run_id
START_FRAME = args_cli.start_frame
END_FRAME = args_cli.end_frame
SPECIFY_EMOTION = args_cli.specify_emotion

EMOTION_LABELS = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
if SPECIFY_EMOTION:
    EMOTION_IDX = EMOTION_LABELS.index(SPECIFY_EMOTION)

PROJECT_ROOT = Path("/home/patrickstyll/Bachelorstudiengang-Software_and_Information_Engineering/SNU_Connectome_Lab/SwiFT-IO")
DATA_CSV = PROJECT_ROOT / "results_each" / RUN_ID / "residuals.csv"

# load data
data = pd.read_csv(DATA_CSV)
subjects = data['subject'].unique()
print(f"Found {len(subjects)} unique subjects in the data.")

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
    if SPECIFY_EMOTION:
        knitted_subject[f"residual_{EMOTION_IDX}_{SPECIFY_EMOTION}"] = retrieve_list_from_string(subject_data[f"residual_{EMOTION_IDX}_{SPECIFY_EMOTION}"].values)[START_FRAME:END_FRAME]
        knitted_subject[f"prediction_{EMOTION_IDX}_{SPECIFY_EMOTION}"] = retrieve_list_from_string(subject_data[f"prediction_{EMOTION_IDX}_{SPECIFY_EMOTION}"].values)[START_FRAME:END_FRAME]
    else:
        for i, e in enumerate(EMOTION_LABELS):
            knitted_subject[f"residual_{i}_{e}"] = retrieve_list_from_string(subject_data[f"residual_{i}_{e}"].values)[START_FRAME:END_FRAME]
            knitted_subject[f"prediction_{i}_{e}"] = retrieve_list_from_string(subject_data[f"prediction_{i}_{e}"].values)[START_FRAME:END_FRAME]
    knitted_subject["subject"] = subject
    knitted_data = pd.concat([knitted_data, pd.DataFrame([knitted_subject])], ignore_index=True)

# plot residuals of all subjects against time
time = torch.arange(START_FRAME, END_FRAME, 1)

def plot_resids_against_time(knitted_data: pd.DataFrame,
                             emotion_idx: int,
                             emotion_label: str) -> None:
    plt.figure(figsize=(12, 6))
    for _, data in knitted_data.iterrows():
        subj = data["subject"]
        residuals = data[f"residual_{emotion_idx}_{emotion_label}"]
        plt.plot(time, residuals, label=f"Subject {subj}", alpha=0.5, color='black')
    plt.xlabel("Time")
    plt.ylabel("Residual Value")
    plt.title(f"Residuals of {emotion_label}")
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
    mean_residuals = knitted_data[f"residual_{emotion_idx}_{emotion_label}"].mean()
    plt.figure(figsize=(12, 6))
    plt.hist(mean_residuals, bins=30, density=True, alpha=0.5, color='black')
    plt.xlabel("Mean Residual Value")
    plt.ylabel("Density")
    plt.title(f"Mean Residuals of {emotion_label}")
    plt.grid(True)
    plt.show()

if SPECIFY_EMOTION:
    plot_mean_residuals(knitted_data, EMOTION_IDX, SPECIFY_EMOTION)
else:
    for i, e in enumerate(EMOTION_LABELS):
        plot_mean_residuals(knitted_data, i, e)

print("Done!")