import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
# python /pscratch/sd/k/kimbo/SwiFT-IO/src/derive_residuals_whole_2.py --run_id tubg3tim --split val --save_name residuals_test
# python /pscratch/sd/k/kimbo/SwiFT-IO/src/derive_residuals_whole_2.py --run_id tubg3tim --split val --save_name residuals_test --resume

# python /pscratch/sd/k/kimbo/SwiFT-IO/src/derive_residuals_whole_2.py --run_id tubg3tim --split val --save_name residuals_valid
# python /pscratch/sd/k/kimbo/SwiFT-IO/src/derive_residuals_whole_2.py --run_id tubg3tim --split val --save_name residuals_valid --resume

# python /pscratch/sd/k/kimbo/SwiFT-IO/src/derive_residuals_whole_2.py --run_id tubg3tim --split train --save_name residuals_train
# python /pscratch/sd/k/kimbo/SwiFT-IO/src/derive_residuals_whole_2.py --run_id tubg3tim --split train --save_name residuals_train --resume


# ======== 감정 레이블 ========
emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']


# ======== 모델 및 데이터 로딩 함수 ========
def load_model_and_data(run_id, input_type, seq_length, input_offset, dummy_split_path):
    project_root = Path("/pscratch/sd/k/kimbo/SwiFT-IO")
    sys.path.append(str(project_root / "src"))

    from module.pl_classifier import LitClassifier
    from module.utils.data_module import fMRIDataModule

    ckpt_dir = project_root / f"output/moviefmri/{run_id}"
    ckpt_path = list(ckpt_dir.glob("checkpt*"))[0]
    ckpt = torch.load(ckpt_path, map_location='cpu')

    args = ckpt['hyper_parameters']
    args.update({
        'input_type': input_type,
        'seq_length': seq_length,
        'image_path': "/global/cfs/cdirs/m4750/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120",
        'default_root_dir': str(project_root / "output/moviefmri"),
        'shuffle_time_sequence': False,
        'time_as_channel': False,
        'eval_batch_size': 1,
        'input_offset': input_offset,
        'bad_subj_path': None,
        'limit_training_samples': 0,
        'img_size': [96, 96, 96, seq_length],
        'eval_num_workers': 1, 
        'dataset_split_seed': 777,
        'stratified_params': ['Sex', 'Age']
    })

    data_module = fMRIDataModule(**args)
    data_module.split_file_path = str(dummy_split_path)
    data_module.prepare_data()
    data_module.setup(stage='fit')

    model = LitClassifier(data_module=data_module, **args)
    model.load_state_dict(ckpt['state_dict'], strict = False)
    model.eval().cpu()

    return model, data_module, project_root


# ======== 예측 및 오차 저장 함수 (subject 단위, resume 포함) ========
def save_predictions_and_residuals(model, loader, save_dir, seq_length, input_offset, max_segments,
                                   save_name="residuals_whole", resume=False, save_subject_wise_npy=True):
    rows = []
    resume_subjects = set()
    resume_csv_path = save_dir / f"{save_name}.csv"

    # ✅ 이미 저장된 subject 목록 불러오기
    if resume and resume_csv_path.exists():
        prev_df = pd.read_csv(resume_csv_path)
        resume_subjects = set(prev_df["subject"].unique())
        print(f"🔁 Resuming from existing file. Skipping {len(resume_subjects)} subjects.")
    else:
        print("🚀 Starting fresh.")

    segment_counter = {}
    current_subject = None
    subject_rows = []
    subject_data = {key: [] for key in ["split", "subject", "start_frame", "end_frame"]}
    for i, label in enumerate(emotion_labels):
        subject_data[f"residual_{i}_{label}"] = []
        subject_data[f"prediction_{i}_{label}"] = []

    for data in tqdm(loader):
        try:
            subj_name = data['subject_name'][0]

            if subj_name != current_subject:
                # 저장 루틴 (이전 subject 저장)
                if current_subject is not None and subject_rows:
                    df = pd.DataFrame(subject_rows)
                    df.to_csv(save_dir / f"{save_name}.csv", mode='a', index=False, header=not resume_csv_path.exists())
                    if save_subject_wise_npy:
                        np.save(save_dir / f"{save_name}_{current_subject}.npy", subject_data)
                    print(f"📁 Saved subject: {current_subject}")
                    subject_rows = []
                    subject_data = {key: [] for key in ["split", "subject", "start_frame", "end_frame"]}
                    for i, label in enumerate(emotion_labels):
                        subject_data[f"residual_{i}_{label}"] = []
                        subject_data[f"prediction_{i}_{label}"] = []

                if resume and subj_name in resume_subjects:
                    continue

                current_subject = subj_name

            seg_idx = segment_counter.get(subj_name, 0)
            if seg_idx >= max_segments:
                continue

            input_ts = data['fmri_sequence'].float().cpu()
            target = data['target'].float().cpu()

            with torch.no_grad():
                pred = model(input_ts)

            residual = torch.abs(pred - target.squeeze(0))

            start = seg_idx * seq_length
            input_start = start + input_offset
            end = input_start + seq_length - 1
            segment_counter[subj_name] = seg_idx + 1

            residual_np = residual.numpy()
            pred_np = pred.numpy()

            row = {"split": "whole", "subject": subj_name, "start_frame": input_start, "end_frame": end}
            subject_data["split"].append("whole")
            subject_data["subject"].append(subj_name)
            subject_data["start_frame"].append(input_start)
            subject_data["end_frame"].append(end)

            for i, emotion in enumerate(emotion_labels):
                row[f"residual_{i}_{emotion}"] = residual_np[:, i].tolist()
                row[f"prediction_{i}_{emotion}"] = pred_np[:, i].tolist()
                subject_data[f"residual_{i}_{emotion}"].append(residual_np[:, i].tolist())
                subject_data[f"prediction_{i}_{emotion}"].append(pred_np[:, i].tolist())

            subject_rows.append(row)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    # 마지막 subject 저장
    if current_subject is not None and subject_rows:
        df = pd.DataFrame(subject_rows)
        df.to_csv(save_dir / f"{save_name}.csv", mode='a', index=False, header=not resume_csv_path.exists())
        if save_subject_wise_npy:
            np.save(save_dir / f"{save_name}_{current_subject}.npy", subject_data)
        print(f"📁 Saved subject: {current_subject} (final)")

    print(f"✅ All residuals saved to {save_dir}")


# ======== 메인 실행부 ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--input_offset', type=int, default=3)
    parser.add_argument('--input_type', type=str, default="movieDM")
    parser.add_argument('--resume', action='store_true', help='이전에 저장된 subject는 건너뜀')

    args = parser.parse_args()

    dummy_split_path = '/pscratch/sd/k/kimbo/SwiFT-IO/tmp/7_checkpoint/kimbo_aica/split_fixed_1.txt'
    model, data_module, project_root = load_model_and_data(
        args.run_id, args.input_type, args.seq_length, args.input_offset, dummy_split_path
    )

    # data_module.setup(stage='fit')

    # split 선택
    if args.split == 'train':
        selected_dataset = data_module.train_dataset
    elif args.split == 'val':
        selected_dataset = data_module.val_dataset
    elif args.split == 'test':
        selected_dataset = data_module.test_dataset
    else:
        raise ValueError(f"Invalid split: {args.split}")

    data_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=1, shuffle=False, num_workers=1)

    total_frames = 750
    max_segments = (total_frames - args.input_offset) // args.seq_length

    save_dir = project_root / f"analysis/residuals/{args.run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_name = args.save_name or f"residuals_{args.split}"

    save_predictions_and_residuals(
        model=model,
        loader=data_loader,
        save_dir=save_dir,
        seq_length=args.seq_length,
        input_offset=args.input_offset,
        max_segments=max_segments,
        save_name=save_name,
        resume=args.resume
    )
