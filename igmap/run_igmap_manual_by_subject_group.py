import argparse
import time
from pathlib import Path
import torch
import nibabel as nib
from multiprocessing import Pool
import sys
import os
from torch.utils.data import DataLoader, Subset

emotion_labels = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

# Global variable for shared model
model = None
data_module = None  # <- 빠졌던 이 부분 추가

def init_model_and_data(ckpt_path_str, args_model_dict, project_root):
    global model, data_module

    sys.path.append(str(project_root / "src"))
    from module.pl_classifier import LitClassifier
    from module.utils.data_module import fMRIDataModule

    ckpt = torch.load(ckpt_path_str, map_location="cpu")
    args_model_dict["num_workers"] = 0
    args_model_dict["eval_num_workers"] = 0
    args_model_dict['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
    args_model_dict['default_root_dir'] = str(project_root / "output/moviefmri")
    args_model_dict['shuffle_time_sequence'] = False
    args_model_dict['time_as_channel'] = False
    args_model_dict['eval_batch_size'] = 1
    # Keep original model's input_type, input_offset, sequence_length from checkpoint
    # args_model_dict already has these from checkpoint
    args_model_dict['bad_subj_path'] = None
    args_model_dict['limit_training_samples'] = 0
    # img_size will match checkpoint's sequence_length
    args_model_dict['downstream_task'] = 'emotions'
    args_model_dict['decoder'] = 'series_decoder'

    # Add missing data module parameters
    if 'dataset_split_seed' not in args_model_dict:
        args_model_dict['dataset_split_seed'] = args_model_dict.get('seed', 777)
    if 'stratified_params' not in args_model_dict:
        args_model_dict['stratified_params'] = None
    if 'train_split' not in args_model_dict:
        args_model_dict['train_split'] = 0.7
    if 'val_split' not in args_model_dict:
        args_model_dict['val_split'] = 0.15

    print("🚀 Initializing model & data...", flush=True)
    data_module = fMRIDataModule(**args_model_dict)
    data_module.prepare_data()
    data_module.setup(stage='test')

    model = LitClassifier(data_module=data_module, **args_model_dict)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.cpu()
    print("✅ Model & Data initialized.")

def compute_ig_on_prediction_average(input_ts, baseline, i, n_steps=10):
    global model
    alphas = torch.linspace(0, 1.0, steps=n_steps).view(-1, 1, 1, 1, 1, 1).to(input_ts.device)
    delta = input_ts - baseline
    scaled_inputs = baseline + alphas * delta
    grads = []
    for s_input in scaled_inputs:
        s_input = s_input.unsqueeze(0).requires_grad_(True)
        output = model(s_input)
        print(f"📈 Forward IG step {i}/{n_steps}, emotion: {emotion_labels[i]}")
        scalar = output[:, i].mean()
        grad = torch.autograd.grad(outputs=scalar, inputs=s_input)[0]
        grads.append(grad)
    avg_grads = torch.stack(grads).mean(dim=0)
    integrated_grads = delta * avg_grads
    return integrated_grads.detach()

def process_subject(args_tuple):
    subject, args, affine_path, project_root = args_tuple
    global model, data_module

    print(f"\n🚀 Start subject: {subject} | PID: {os.getpid()}", flush=True)
    overall_start = time.time()

    affine = nib.load(str(affine_path)).affine

    testset = model.data_module.test_dataset
    subj_indices = [
        idx for idx, s in enumerate(testset)
        if (s["subject_name"] if isinstance(s["subject_name"], str) else s["subject_name"][0]) == subject
    ]
    if not subj_indices:
        print(f"❌ No matching data for subject: {subject}", flush=True)
        return
    print(f"✅ Found {len(subj_indices)} sequences for {subject}", flush=True)

    test_loader = DataLoader(Subset(testset, subj_indices), batch_size=1, shuffle=False, num_workers=0)
    instance_count = 0  # 인스턴스 번호 카운터 (옵션 사용 시)

    for data in test_loader:
        subj = data['subject_name'] if isinstance(data['subject_name'], str) else data['subject_name'][0]
        TR_index = int(data['TR'])
        input_ts = data['fmri_sequence'].float().cpu()
        baseline = torch.zeros_like(input_ts)

        for i, emotion_label in enumerate(emotion_labels):
            out_dir = project_root / f"analysis/4_IGmap/results_each/{args.run_id}/nii_segments" / subject / f"target{i}_{emotion_label}"
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.instance_level:
                # Instance-level 모드: 결과 tensor의 마지막 차원(시간 축)을 개별 파일로 저장
                result = compute_ig_on_prediction_average(input_ts, baseline, i=i, n_steps=10)
                # result shape: [B, C, X, Y, Z, T], B=1, C=1
                result_tensor = result[0, 0, :, :, :, :]  # shape: [X, Y, Z, T]
                num_instances = result_tensor.shape[-1]
                for t in range(num_instances):
                    ig_map = result_tensor[..., t].cpu().numpy()
                    out_path = out_dir / f"{subject}_{emotion_label}_inst{instance_count:03d}_t{t}.nii.gz"
                    nib.save(nib.Nifti1Image(ig_map, affine), out_path)
                    print(f"[IG OK] {subject} - {emotion_label} instance {instance_count:03d} timepoint {t}")
                instance_count += 1
            else:
                # 기본 모드: 양수와 음수를 분리하여 각 영역 평균
                result = compute_ig_on_prediction_average(input_ts, baseline, i=i, n_steps=20)
                # result shape: [B, C, X, Y, Z, T] → select result[0,0] gives shape: [X, Y, Z, T]
                result_tensor = result[0, 0, :, :, :, :]
                # 마스크 없이 각 voxel의 마지막 차원(인스턴스)을 대상으로,
                # 양수 값과 음수 값을 분리하여 계산합니다.
                pos_mask = (result_tensor > 0).float()
                neg_mask = (result_tensor < 0).float()
                pos_sum = (result_tensor * pos_mask).sum(dim=-1)
                pos_count = pos_mask.sum(dim=-1) + 1e-8  # 0 division 방지
                avgpred_positive = pos_sum / pos_count

                neg_sum = (result_tensor * neg_mask).sum(dim=-1)
                neg_count = neg_mask.sum(dim=-1) + 1e-8
                avgpred_negative = neg_sum / neg_count

                # 파일 이름 구성
                out_path_pos = out_dir / f"{subject}_{emotion_label}_AVGpred_positive.nii.gz"
                out_path_neg = out_dir / f"{subject}_{emotion_label}_AVGpred_negative.nii.gz"
                nib.save(nib.Nifti1Image(avgpred_positive.cpu().numpy(), affine), out_path_pos)
                nib.save(nib.Nifti1Image(avgpred_negative.cpu().numpy(), affine), out_path_neg)
                print(f"[IG OK] {subject} - {emotion_label} (positive and negative separated)")
        # 모든 데이터 인스턴스를 처리하도록 break 제거.
    print(f"✅ Total time for {subject}: {time.time() - overall_start:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--subject_file', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of parallel processes (default: 1 for stability)")
    parser.add_argument('--instance_level', action='store_true',
                        help="If set, do not average over instances; save each instance's IG map separately.")
    parser.add_argument('--project_root', type=str, default='/scratch/connectome/kimbo/SwiFT-IO-4-v9/SwiFT-IO',
                        help="Project root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    # Create analysis directory if not exists
    analysis_dir = project_root / f"analysis/4_IGmap/results_each/{args.run_id}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Read subject list
    subject_file_path = analysis_dir / args.subject_file if not Path(args.subject_file).is_absolute() else Path(args.subject_file)
    with open(subject_file_path, 'r') as f:
        subject_list = [line.strip() for line in f if line.strip()]

    print(f"📋 Processing {len(subject_list)} subjects: {subject_list}")

    # Load checkpoint
    ckpt_path = list((project_root / f"output/moviefmri/{args.run_id}").glob("checkpt*"))[0]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_model_dict = ckpt['hyper_parameters']

    print(f"✅ Loaded checkpoint: {ckpt_path.name}")
    print(f"   Model config: seq_len={args_model_dict.get('sequence_length')}, offset={args_model_dict.get('input_offset')}")

    # Use reference affine (standard MNI152 2mm)
    affine_path = project_root / "analysis/4_IGmap/reference_affine_MNI152_2mm.nii.gz"
    if not affine_path.exists():
        raise FileNotFoundError(f"❌ Affine reference file not found: {affine_path}\nPlease run the affine creation script first.")

    with Pool(processes=args.n_jobs, initializer=init_model_and_data,
              initargs=(str(ckpt_path), args_model_dict, project_root)) as pool:
        pool.map(process_subject, [(subj, args, affine_path, project_root) for subj in subject_list])

    print("✅ All processing complete!")
