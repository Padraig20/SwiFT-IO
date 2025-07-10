import torch
import os
import glob
import pandas as pd

# Paths (adjust if needed)
data_dir = "/global/cfs/cdirs/m4750/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120/img"
csv_path = "/pscratch/sd/k/kimbo/SwiFT-IO-test/SwiFT-IO/data/splits/split_seed777_Sex_Age.csv"
output_dir = "/pscratch/sd/k/kimbo/SwiFT-IO-test/SwiFT-IO/data/isc_timecourses"

print("Loading subject split CSV...")
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error: Could not read CSV file at {csv_path}: {e}")
    exit(1)

# Verify expected columns
if 'SUBJECT_ID' not in df.columns or 'split' not in df.columns:
    print("Error: CSV file does not contain 'SUBJECT_ID' or 'split' columns.")
    exit(1)

total_subjects = len(df)
print(f"Found {total_subjects} entries in CSV file.")

# Filter subjects with exactly 750 frames
eligible_subjects = []
ineligible_subjects = []
print("Checking for subjects with 750 frames...")
for subj in df['SUBJECT_ID']:
    subj_str = str(subj)
    subj_dir = os.path.join(data_dir, subj_str)
    if not os.path.isdir(subj_dir):
        ineligible_subjects.append(subj)
        print(f"Warning: Directory not found for subject {subj_str}, skipping.")
        continue
    frame_files = glob.glob(os.path.join(subj_dir, "frame_*.pt"))
    if len(frame_files) == 750:
        # Verify first and last frame exist
        first_frame = os.path.join(subj_dir, "frame_0.pt")
        last_frame = os.path.join(subj_dir, "frame_749.pt")
        if os.path.exists(first_frame) and os.path.exists(last_frame):
            eligible_subjects.append(subj)
        else:
            ineligible_subjects.append(subj)
            print(f"Warning: Subject {subj_str} missing frame_0.pt or frame_749.pt, skipping.")
    else:
        ineligible_subjects.append(subj)
        print(f"Skipping subject {subj_str}: expected 750 frames, found {len(frame_files)}.")

print(f"Total eligible subjects with 750 frames: {len(eligible_subjects)}")
if ineligible_subjects:
    print(f"Skipped {len(ineligible_subjects)} subject(s) due to incomplete data.")

# Split subjects into train/val/test groups
train_subjects = [subj for subj in eligible_subjects if df.loc[df['SUBJECT_ID'] == subj, 'split'].iloc[0] == 'train']
val_subjects   = [subj for subj in eligible_subjects if df.loc[df['SUBJECT_ID'] == subj, 'split'].iloc[0] == 'val']
test_subjects  = [subj for subj in eligible_subjects if df.loc[df['SUBJECT_ID'] == subj, 'split'].iloc[0] == 'test']

print(f"Subjects in train split: {len(train_subjects)}")
print(f"Subjects in val split:   {len(val_subjects)}")
print(f"Subjects in test split:  {len(test_subjects)}")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Compute ISC timecourse for each split
for split_name, subject_list in [("train", train_subjects), ("val", val_subjects), ("test", test_subjects)]:
    if len(subject_list) == 0:
        print(f"No subjects in {split_name} split to process. Skipping.")
        continue
    if len(subject_list) < 2:
        print(f"Skipping {split_name} split (only {len(subject_list)} subject(s)).")
        continue

    print(f"\nProcessing {split_name} split ({len(subject_list)} subjects)...")
    # Determine feature dimension from first subject's first frame
    first_subj = str(subject_list[0])
    sample_path = os.path.join(data_dir, first_subj, "frame_0.pt")
    try:
        sample_tensor = torch.load(sample_path)
    except Exception as e:
        print(f"Error loading sample frame for subject {first_subj}: {e}")
        continue
    if sample_tensor.ndim > 1:
        sample_tensor = sample_tensor.view(-1)
    D = sample_tensor.numel()
    T = 750  # number of timepoints/frames
    # Initialize sum of normalized vectors and count of contributions
    S = torch.zeros((T, D), dtype=torch.float32)
    count_vec = torch.zeros(T, dtype=torch.int32)
    processed_count = 0

    for idx, subj in enumerate(subject_list):
        subj_str = str(subj)
        subj_dir = os.path.join(data_dir, subj_str)
        print(f"  Loading subject {subj_str} ({idx+1}/{len(subject_list)})...")
        subject_data = torch.empty((T, D), dtype=torch.float32)
        error_flag = False
        # Load all frames for this subject
        for i in range(T):
            frame_path = os.path.join(subj_dir, f"frame_{i}.pt")
            try:
                x = torch.load(frame_path)
            except Exception as e:
                print(f"    Error loading {frame_path}: {e}")
                error_flag = True
                break
            # flatten the 4D volume into a vector of length D
            x = x.view(-1).to(torch.float32)  
            subject_data[i] = x

        if error_flag:
            print(f"    Skipping subject {subj_str} due to read errors.")
            continue
        # Normalize each timepoint vector for this subject
        means = subject_data.mean(dim=1, keepdim=True)      # mean across features per timepoint
        centered = subject_data - means                     # subtract mean
        norms = centered.norm(dim=1, keepdim=True)          # L2 norm per timepoint
        zero_rows = (norms == 0).squeeze()                  # identify any timepoints with zero norm
        if zero_rows.any():
            centered[zero_rows] = 0
            norms[zero_rows] = 1.0
            print(f"    Warning: Subject {subj_str} has zero-variance at {zero_rows.sum().item()} timepoint(s).")
        normalized = centered / norms                       # (T, D) each row is unit norm now
        # Accumulate sums and counts
        S += normalized
        valid_mask = torch.ones(T, dtype=torch.int32)
        valid_mask[zero_rows] = 0
        count_vec += valid_mask
        processed_count += 1

    if processed_count == 0:
        print(f"No valid subjects processed for {split_name} split. Skipping.")
        continue

    # Compute average Pearson correlation at each timepoint
    sum_sq = (S**2).sum(dim=1)  # sum of squares of S for each timepoint
    avg_corr = torch.zeros(T, dtype=torch.float32)
    mask = count_vec > 1
    if mask.any():
        avg_corr[mask] = (sum_sq[mask] - count_vec[mask].to(torch.float32)) / \
                         (count_vec[mask].to(torch.float32) * (count_vec[mask] - 1).to(torch.float32))
    # Save ISC timecourse for this split
    output_path = os.path.join(output_dir, f"{split_name}.pt")
    torch.save(avg_corr, output_path)
    print(f"  Saved ISC timecourse for {split_name} split to {output_path}")
    if processed_count != len(subject_list):
        print(f"  Note: Processed {processed_count}/{len(subject_list)} subjects for {split_name} split.")
print("Done.")
