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

def init_model_and_data(ckpt_path_str, args_model_dict):
    global model, data_module

    sys.path.append(str(Path("/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/src")))
    from module.pl_classifier import LitClassifier
    from module.utils.data_module import fMRIDataModule

    ckpt = torch.load(ckpt_path_str, map_location="cpu")
    args_model_dict["num_workers"] = 0
    args_model_dict["eval_num_workers"] = 0
    args_model_dict['image_path'] = "/scratch/HBN/3.3.1.movieDM_MNI_to_TRs_smooth_znorm_241120"
    args_model_dict['default_root_dir'] = str(Path("/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/output/moviefmri"))
    args_model_dict['shuffle_time_sequence'] = False
    args_model_dict['time_as_channel'] = False
    args_model_dict['eval_batch_size'] = 1
    args_model_dict['input_type'] = 'movieDM'
    args_model_dict['input_offset'] = 0
    args_model_dict['seq_length'] = 50
    args_model_dict['bad_subj_path'] = None
    args_model_dict['limit_training_samples'] = 0
    args_model_dict['img_size'] = [96, 96, 96, 50]
    args_model_dict['downstream_task'] = 'emotions'
    args_model_dict['decoder'] = 'series_decoder'

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
    subject, args, affine_path = args_tuple
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
            out_dir = Path(f"/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/results_each/{args.run_id}/nii_segments") / subject / f"target{i}_{emotion_label}"
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
    parser.add_argument('--input_offset', type=int, default=0)
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--input_type', type=str, choices=['movieDM', 'movieTP'], default='movieDM')
    parser.add_argument('--n_jobs', type=int, default=4)
    # instance-level 모드를 위한 인자 추가
    parser.add_argument('--instance_level', action='store_true',
                        help="If set, do not average over instances; save each instance's IG map separately.")
    args = parser.parse_args()

    save_dir = Path(f"/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/results_each/{args.run_id}/")
    with open(save_dir / args.subject_file, 'r') as f:
        subject_list = [line.strip() for line in f if line.strip()]

    if len(subject_list) != 1:
        raise ValueError("❌ 단일 subject 실행 모드에서는 subject_file에 한 명만 포함되어야 합니다.")
    subject = subject_list[0]

    project_root = Path("/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO")
    ckpt_path = list((project_root / f"output/moviefmri/{args.run_id}").glob("checkpt*"))[0]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_model_dict = ckpt['hyper_parameters']

    affine_path = Path("/scratch/connectome/kimbo/SwiFT-IO-2/SwiFT-IO/analysis/4_IGmap/sub-NDARVN715MJ9_task-movieDM_space-MNI152NLin2009cAsym_desc-preproc_bold_smoothed.nii.gz")

    with Pool(processes=args.n_jobs, initializer=init_model_and_data, initargs=(str(ckpt_path), args_model_dict)) as pool:
        pool.map(process_subject, [(subj, args, affine_path) for subj in subject_list])

    print("✅ All processing complete!")
