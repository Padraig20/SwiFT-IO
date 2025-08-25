import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle

from torchmetrics import PearsonCorrCoef # Accuracy,
from torchmetrics.regression import R2Score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, average_precision_score, precision_recall_curve # kimbo change
from sklearn.preprocessing import label_binarize
import monai.transforms as monai_t

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from .models.load_model import load_model
from .utils.metrics import Metrics
from .utils.lr_scheduler import CosineAnnealingWarmUpRestarts

from einops import rearrange

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wandb 
import copy
import pdb

import torch
# --- 필요한 모든 torchmetrics 임포트 ---
from torchmetrics.classification import (
    AUROC, 
    Precision, 
    Recall, 
    F1Score, 
    MatthewsCorrCoef, 
    AveragePrecision
)
from torchmetrics.regression import PearsonCorrCoef, R2Score


class LitClassifier(pl.LightningModule):
    
    def __init__(self,data_module, **kwargs):
        super().__init__()
        self.data_module = data_module  # Pickle 불가능한 객체는 직접 저장

        # ✅ 1. Pickle 불가능한 객체 필터링 함수
        def is_pickleable(v):
            try:
                copy.deepcopy(v)  # Pickle 가능 여부 테스트
                return True
            except Exception:
                return False

        # ✅ 2. `data_module`과 None 값 제거
        hparams = {k: v for k, v in kwargs.items() if k != "data_module" and v is not None}

        # ✅ 3. wandb 같은 Pickle 불가능한 객체 제거
        hparams = {k: v for k, v in hparams.items() if not isinstance(v, wandb.sdk.wandb_run.Run)}

        # ✅ 4. `id` 필드 문자열 변환 (Pickle 가능하게 처리)
        if "id" in hparams:
            hparams["id"] = str(hparams["id"])  # Pickle 가능하도록 변환

        # ✅ 5. DDP 환경에서만 Pickle 검증 수행
        if torch.cuda.device_count() > 1:
            remove_keys = []  # Pickle 불가능한 값들을 저장할 리스트
            for k, v in hparams.items():
                if not is_pickleable(v):
                    print(f"❌ Pickle 불가능한 값 (DDP에서 오류 가능): {k} -> {type(v)} 제거됨")
                    remove_keys.append(k)  # 삭제할 키 저장

            # ✅ 한꺼번에 삭제 (딕셔너리 변경 중 반복 방지)
            for k in remove_keys:
                del hparams[k]
        
        if not hasattr(self.hparams, "use_youden_threshold"):
            self.hparams.use_youden_threshold = True
        if not hasattr(self.hparams, "threshold_scope"):
            self.hparams.threshold_scope = "per_target"   # or "global"
        if not hasattr(self.hparams, "positive_class_index"):
            self.hparams.positive_class_index = 1


        # Safe defaults:
        if not hasattr(self.hparams, "use_youden_threshold"):
            self.hparams.use_youden_threshold = True
        if not hasattr(self.hparams, "threshold_scope"):
            self.hparams.threshold_scope = "per_target"   # or "global"
        if not hasattr(self.hparams, "positive_class_index"):
            self.hparams.positive_class_index = 1

        # Storage for learned thresholds (set during validation, used in test)
        if not hasattr(self, "eval_thresholds"):
            self.eval_thresholds = {"global": 0.5, "per_target": {}}

        # ✅ 6. 안전한 값만 `save_hyperparameters()`에 전달
        self.save_hyperparameters(hparams)

        # you should define target_values at the Dataset classes
        if data_module and hasattr(data_module, "train_dataset"):
            target_values = data_module.train_dataset.target_values

            if self.hparams.label_scaling_method == 'standardization':
                scaler = StandardScaler()
                normalized_target_values = scaler.fit_transform(target_values)
                print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
            elif self.hparams.label_scaling_method == 'minmax': 
                scaler = MinMaxScaler()
                normalized_target_values = scaler.fit_transform(target_values)
                print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
            self.scaler = scaler
        else:
            print("⚠️ No train_dataset provided — skipping target normalization")
            self.scaler = None  # fallback: not used


        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)
        
        self.output_head = load_model(self.hparams.decoder, self.hparams)

        self.metric = Metrics()

        self.valid_only = kwargs.get("valid_only", False) # kimbo change

    def forward(self, x):
        x = self.model(x)
        return self.output_head(x)
    
    def augment(self, img):
        """
        Applies data augmentation to a 6D image tensor. The augmentations include random affine transformations, Gaussian noise, and Gaussian smoothing. 
        Augmentation can be controlled to target intensity or affine transformations only. Ensures consistent augmentation across time steps.
        """
        B, C, H, W, D, T = img.shape

        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=0.5, # we are using 0.5 rather than 1.0 in SwiFT v2 research.
            # 0.175 rad = 10 degrees
            rotate_range=(0.175, 0.175, 0.175),
            scale_range = (0.1, 0.1, 0.1),
            mode = "bilinear",
            padding_mode = "border",
            device = device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) 

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            # set augmentation seed to be the same for all time steps
            for t in range(T):
                if self.hparams.augment_only_affine:
                    rand_affine.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                else:
                    comp.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        img = rearrange(img, 'b t c h w d -> b c h w d t')

        return img


    def _compute_logits(self, batch, augment_during_training=None, mode=None):
        """
        Processes a batch of data to compute logits for either classification or regression tasks. 
        Applies optional augmentation during training and handles label scaling for regression tasks.
        """
        fmri, subj, target_value, tr, sex = batch.values()

        if augment_during_training:
            if mode == 'train': # kimbo change
                fmri = self.augment(fmri) # torch.Size([2, 1, 96, 96, 96, 30])

        feature = self.model(fmri) # torch.Size([2, 768, 120])
        # TODO: shape 확인. 
        # Classification task
        if self.hparams.downstream_task_type == 'classification':
            logits = self.output_head(feature).squeeze() # (b,num_classes)  /  (b,t,num_targets,num_classes) # torch.Size([2, 30, 7, 2]) (output_head(feature): torch.Size([2, 30, 7, 2]))
            target = target_value.float().squeeze()      # (b,num_classes)  /  (b,t,num_targets,num_classes) # torch.Size([2, 30, 7])
            if self.hparams.decoder == 'series_decoder': 
                logits = rearrange(logits, 'b t ta c -> b (t ta) c') # torch.Size([2, 210, 2])
                target = rearrange(target, 'b t ta -> b (t ta)') # torch.Size([2, 210])
        # Regression task
        elif self.hparams.downstream_task_type == 'regression':
            logits = self.output_head(feature) # 
            if logits.ndim == 2:  # e.g. [20, 7]
                logits = logits.unsqueeze(0)  # → [1, 20, 7]
            unnormalized_target = target_value.float() # (b,1)

            if self.hparams.decoder == 'series_decoder': # (batch, T, E) -> (batch, T*E)
                logits = logits.view(logits.size(0), -1)
                unnormalized_target = unnormalized_target.view(unnormalized_target.size(0), -1)
            
            if self.hparams.label_scaling_method == 'standardization': # default
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])

        return subj, logits, target
    
    def _calculate_loss(self, batch, mode):
        """
        Calculates the loss and performance metrics for classification or regression tasks. 
        Logs the results for monitoring during training or evaluation.
        """
        subj, logits, target = self._compute_logits(batch, augment_during_training = self.hparams.augment_during_training)

        if self.hparams.downstream_task_type == 'classification':
            if self.hparams.decoder == 'series_decoder': # [b, (t ta), c] -> [(b t ta), c]
                logits = rearrange(logits, 'b tta c -> (b tta) c') # torch.Size([420, 2])
                target = target.flatten() # (b,tta) -> (b*tta) # torch.Size([420])
            # ---- Unified binary/multiclass handling ----
            num_classes = logits.size(-1)
            if num_classes == 2 and self.hparams.num_classes == 2: 
                # Binary classification with BCEWithLogitsLoss
                # Convert 2-logits to a single logit (log-odds): logit_pos - logit_neg
                binary_logit = logits[:, 1] - logits[:, 0]  # shape: [N]
                target_f = target.float()

                # Optional positive class weight for imbalance
                pos_weight = getattr(self.hparams.pos_weight, "pos_weight", None)
                if pos_weight is not None and not torch.is_tensor(pos_weight):
                    pos_weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)

                if pos_weight is not None:
                    loss = F.binary_cross_entropy_with_logits(binary_logit, target_f, pos_weight=pos_weight)
                else:
                    loss = F.binary_cross_entropy_with_logits(binary_logit, target_f)

                # Accuracy: sigmoid(logit)>0.5 <=> logit>0
                pred = (binary_logit > 0).long()
                acc = (pred == target.long()).float().mean()
            else: 
                # CrossEntropy for binary (C=2) or multiclass (C>=3)
                loss = F.cross_entropy(logits, target.long()) # target is float
                acc = self.metric.get_accuracy(logits, target.float().squeeze())
            result_dict = {
                f"{mode}_loss": loss,
                f"{mode}_acc": acc,
            }
        elif self.hparams.downstream_task_type == 'regression':
            loss = F.mse_loss(logits.squeeze(), target.squeeze())
            l1 = F.l1_loss(logits.squeeze(), target.squeeze())
            result_dict = {
                f"{mode}_loss": loss,
                f"{mode}_mse": loss,
                f"{mode}_l1_loss": l1
            }
        self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)
        return loss

    def _best_threshold_youden(self, pos_scores: torch.Tensor, y_true: torch.Tensor):
        """
        pos_scores: (N,) 양성 클래스 점수(확률) [0,1]
        y_true:     (N,) {0,1}
        return: best_threshold(float), best_J(float), tpr_at_best(float), fpr_at_best(float)
        """
        with torch.no_grad():
            # 내림차순 정렬(점수 높을수록 양성)
            scores = pos_scores.detach().flatten().cpu().to(torch.float64)
            y = y_true.detach().flatten().cpu().to(torch.int64)
            scores, idx = torch.sort(scores, descending=True)
            y = y[idx]

            P = int((y == 1).sum().item())
            N = int((y == 0).sum().item())
            if P == 0 or N == 0:
                # 한 클래스만 있으면 ROC/Youden 정의 불가 → 기본값 반환
                return 0.5, 0.0, 0.0, 0.0

            # 누적 TP/FP
            tp_cum = torch.cumsum((y == 1).to(torch.int64), dim=0)  # 길이 N
            fp_cum = torch.cumsum((y == 0).to(torch.int64), dim=0)

            # TPR/FPR
            tpr = tp_cum.to(torch.float64) / P
            fpr = fp_cum.to(torch.float64) / N

            # 점수가 같은 연속 구간 중 '첫 발생 지점'만 사용 (unique_consecutive 대체)
            # mask[k] == True 이면 scores[k] 가 새로운 값의 첫 위치
            mask = torch.ones_like(scores, dtype=torch.bool)
            if scores.numel() > 1:
                mask[1:] = scores[1:] != scores[:-1]

            scores_u = scores[mask]
            tpr_u = tpr[mask]
            fpr_u = fpr[mask]

            J_u = tpr_u - fpr_u
            best_i = int(torch.argmax(J_u).item())
            best_thr = float(scores_u[best_i].item())
            return best_thr, float(J_u[best_i].item()), float(tpr_u[best_i].item()), float(fpr_u[best_i].item())

    def _evaluate_metrics(self, subj_array, total_out, mode):
        """
        Evaluates classification or regression metrics for aggregated subject-level predictions.
        """
        subjects = np.unique(subj_array)

        # 0) 공용 헬퍼: subject별로 모든 seq을 시간축으로 합쳐서 반환
        def _build_subj_tensors_varlen(subj_array, total_out, subjects, t, ta, C):
            """
            Returns:
                subj_logits_list_by_subj: list of tensors, each [T_total_s, ta, C]
                subj_targets_list_by_subj: list of tensors, each [T_total_s, ta]
                kept_subjects: list of subject ids aligned with lists above
            Note:
                각 subject마다 T_total_s(=시퀀스수*t)가 달라도 OK.
            """
            subj_logits_list_by_subj  = []
            subj_targets_list_by_subj = []
            kept_subjects = []

            for subj in subjects:
                # 이 subject의 모든 시퀀스 모으기
                subj_logits_seqs  = [total_out[i][0] for i in range(len(subj_array)) if subj_array[i] == subj]  # 각 [t*ta, C]
                subj_targets_seqs = [total_out[i][1] for i in range(len(subj_array)) if subj_array[i] == subj]  # 각 [t*ta]
                if len(subj_logits_seqs) == 0:
                    continue  # 시퀀스 없으면 스킵

                # [t, ta, C] / [t, ta]로 복원 후 시간축 cat
                seq_logits_3d  = [x.view(t, ta, C) for x in subj_logits_seqs]
                seq_targets_2d = [y.view(t, ta)    for y in subj_targets_seqs]
                logits_cat_3d  = torch.cat(seq_logits_3d,  dim=0)  # [T_total_s, ta, C]
                targets_cat_2d = torch.cat(seq_targets_2d, dim=0)  # [T_total_s, ta]

                subj_logits_list_by_subj.append(logits_cat_3d)
                subj_targets_list_by_subj.append(targets_cat_2d)
                kept_subjects.append(subj)

            return subj_logits_list_by_subj, subj_targets_list_by_subj, kept_subjects



        if self.hparams.downstream_task_type == 'classification':
            t  = self.hparams.img_size[3]       # 예: 30
            ta = self.hparams.num_targets       # 예: 7
            C  = self.hparams.num_classes       # 예: 2
            subj_logits_list_by_subj, subj_targets_list_by_subj, kept_subjects = \
                _build_subj_tensors_varlen(subj_array, total_out, subjects, t, ta, C)

            if len(subj_logits_list_by_subj) == 0:
                # 이 배치/랭크에서 평가할 게 없으면 안전 탈출(로그만 남김)
                self.log(f"{mode}_acc", float('nan'), sync_dist=True)
                self.log(f"{mode}_balacc", float('nan'), sync_dist=True)
                return torch.tensor(0.0, device=next(self.parameters()).device)

            # A-2: 전역(time-step) 지표 계산용 평탄화 (varlen → 리스트 cat)
            # logits_all: [(sum_s T_total_s * ta), C], targets_all: [(sum_s T_total_s * ta)]
            logits_flat_list  = [L.reshape(-1, C) for L in subj_logits_list_by_subj]      # 각 [T_total_s*ta, C]
            targets_flat_list = [Y.reshape(-1)    for Y in subj_targets_list_by_subj]     # 각 [T_total_s*ta]
            logits_all  = torch.cat(logits_flat_list,  dim=0)
            targets_all = torch.cat(targets_flat_list, dim=0)

            with torch.no_grad():
                probs_all = torch.softmax(logits_all.to(dtype=torch.float32), dim=-1)   # [(Σ T_s*ta), C]
                preds_all = probs_all.argmax(dim=-1)                                     # [(Σ T_s*ta)]
                y_np = targets_all.cpu().numpy()
                p_np = preds_all.cpu().numpy()
                acc = accuracy_score(y_np, p_np)
                bal = balanced_accuracy_score(y_np, p_np)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_balacc", bal, sync_dist=True)

            # === [추가] 불균형 대응: Youden threshold + 혼동행렬 기반 지표 (전역) ===
            with torch.no_grad():
                pos_idx = int(self.hparams.positive_class_index)
                pos_scores_all_t = probs_all[:, pos_idx]  # torch tensor, shape [(Σ T_s*ta)]
                y_all_t = targets_all

                # 검증 단계에서 임계값 학습(저장), 테스트/추론 단계에서 사용
                use_thr = 0.5
                if getattr(self.hparams, "use_youden_threshold", False):
                    if mode in ["val", "valid", "validation"] and getattr(self.hparams, "threshold_scope", "per_target") == "global":
                        best_thr, best_J, tpr_b, fpr_b = self._best_threshold_youden(pos_scores_all_t, y_all_t)
                        # 저장(전역)
                        self.eval_thresholds["global"] = float(best_thr)
                        self.log(f"{mode}_thr_global_candidate", float(best_thr), sync_dist=True)

                    # 이번 패스에서 사용할 threshold 선택
                    if getattr(self.hparams, "threshold_scope", "per_target") == "global":
                        use_thr = float(self.eval_thresholds.get("global", 0.5))

                # cutoff 적용 예측
                preds_thr_all_np = (pos_scores_all_t.detach().cpu().numpy() >= use_thr).astype(int)
                y_np_all = y_all_t.detach().cpu().numpy()

                # 혼동행렬 및 파생지표
                cm = confusion_matrix(y_np_all, preds_thr_all_np, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
                    fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
                    fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
                    tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

                def _safe_div(a, b): 
                    return float(a) / float(b) if (b is not None and b != 0) else 0.0

                recall_pos   = _safe_div(tp, tp + fn)   # TPR, 민감도
                specificity  = _safe_div(tn, tn + fp)   # TNR, 특이도
                precision_pos= _safe_div(tp, tp + fp)   # PPV
                npv          = _safe_div(tn, tn + fn)   # NPV
                # F1(양성 클래스)
                f1s = precision_recall_fscore_support(y_np_all, preds_thr_all_np, average=None, labels=[0, 1])[2]
                f1_pos = float(f1s[1]) if f1s.size >= 2 else 0.0
                # MCC
                mcc = matthews_corrcoef(y_np_all, preds_thr_all_np) if (tp + fp + tn + fn) > 0 else 0.0
                prevalence    = _safe_div(tp + fn, tp + fn + tn + fp)   # 실제 양성 비율
                pred_pos_rate = _safe_div(tp + fp, tp + fp + tn + fn)   # 예측 양성 비율
                # AUPRC(양성 기준) — 불균형에서 중요
                auprc_global = average_precision_score(y_np_all, pos_scores_all_t.detach().cpu().numpy()) if np.unique(y_np_all).size == 2 else float("nan")

            # 로깅
            self.log(f"{mode}_thr_global_used", use_thr, sync_dist=True)
            self.log(f"{mode}_TP", tp, sync_dist=True)
            self.log(f"{mode}_FP", fp, sync_dist=True)
            self.log(f"{mode}_TN", tn, sync_dist=True)
            self.log(f"{mode}_FN", fn, sync_dist=True)
            self.log(f"{mode}_recall_pos", recall_pos, sync_dist=True)
            self.log(f"{mode}_specificity", specificity, sync_dist=True)
            self.log(f"{mode}_precision_pos", precision_pos, sync_dist=True)
            self.log(f"{mode}_npv", npv, sync_dist=True)
            self.log(f"{mode}_f1_pos", f1_pos, sync_dist=True)
            self.log(f"{mode}_mcc", mcc, sync_dist=True)
            self.log(f"{mode}_prevalence", prevalence, sync_dist=True)
            self.log(f"{mode}_pred_pos_rate", pred_pos_rate, sync_dist=True)
            self.log(f"{mode}_AUPRC_global", auprc_global, sync_dist=True)
            # === [추가 끝] ===


            # (선택) multi-target per-i 지표
            if (self.hparams.decoder in ['series_decoder','multi_target_decoder']) and self.hparams.num_targets > 1:
                for i in range(ta):
                    lg_list = [L[:, i, :].reshape(-1, C) for L in subj_logits_list_by_subj]  # 각 [T_total_s, C]
                    tg_list = [Y[:, i].reshape(-1)      for Y in subj_targets_list_by_subj]  # 각 [T_total_s]
                    lg = torch.cat(lg_list, dim=0)  # [(Σ T_s), C]
                    tg = torch.cat(tg_list, dim=0)  # [(Σ T_s)]

                    probs = torch.softmax(lg.to(dtype=torch.float32), dim=-1)
                    preds = probs.argmax(dim=-1)

                    y_i = tg.cpu().numpy()
                    p_i = preds.cpu().numpy()
                    acc_i = accuracy_score(y_i, p_i)
                    bal_i = balanced_accuracy_score(y_i, p_i)

                    if C == 2 and np.unique(y_i).size == 2:
                        scores_i = (lg[:, 1] - lg[:, 0]).detach().cpu().numpy()
                        auroc_i  = roc_auc_score(y_i, scores_i)
                        auprc_i  = average_precision_score(y_i, scores_i)
                    else:
                        auroc_i = np.nan
                        auprc_i = np.nan

                    self.log(f"{mode}_acc_{i}", acc_i, sync_dist=True)
                    self.log(f"{mode}_balacc_{i}", bal_i, sync_dist=True)
                    self.log(f"{mode}_AUROC_{i}", auroc_i, sync_dist=True)
                    self.log(f"{mode}_AUPRC_{i}", auprc_i, sync_dist=True)

                    # === [추가] 타깃별 임계값 + 혼동행렬 지표 ===
                    if C == 2 and np.unique(y_i).size == 2:
                        pos_idx = int(self.hparams.positive_class_index)
                        pos_scores_i_t = probs[:, pos_idx]  # torch tensor
                        use_thr_i = 0.5

                        if getattr(self.hparams, "use_youden_threshold", False):
                            # 검증에서 per_target 스코프일 때만 각 타깃별 임계값을 학습
                            if mode in ["val", "valid", "validation"] and getattr(self.hparams, "threshold_scope", "per_target") == "per_target":
                                best_thr_i, best_J_i, _, _ = self._best_threshold_youden(pos_scores_i_t, tg)
                                self.eval_thresholds["per_target"][i] = float(best_thr_i)
                                self.log(f"{mode}_thr_{i}_candidate", float(best_thr_i), sync_dist=True)

                            # 사용할 임계값 선택
                            if getattr(self.hparams, "threshold_scope", "per_target") == "per_target":
                                use_thr_i = float(self.eval_thresholds["per_target"].get(i, 0.5))
                            else:
                                use_thr_i = float(self.eval_thresholds.get("global", 0.5))

                        preds_thr_i_np = (pos_scores_i_t.detach().cpu().numpy() >= use_thr_i).astype(int)
                        cm_i = confusion_matrix(y_i, preds_thr_i_np, labels=[0, 1])
                        if cm_i.shape == (2, 2):
                            tn_i, fp_i, fn_i, tp_i = cm_i.ravel()
                        else:
                            tn_i = cm_i[0, 0] if cm_i.shape[0] > 0 and cm_i.shape[1] > 0 else 0
                            fp_i = cm_i[0, 1] if cm_i.shape[0] > 0 and cm_i.shape[1] > 1 else 0
                            fn_i = cm_i[1, 0] if cm_i.shape[0] > 1 and cm_i.shape[1] > 0 else 0
                            tp_i = cm_i[1, 1] if cm_i.shape[0] > 1 and cm_i.shape[1] > 1 else 0

                        def _safe_div(a, b): 
                            return float(a) / float(b) if (b is not None and b != 0) else 0.0

                        recall_pos_i    = _safe_div(tp_i, tp_i + fn_i)
                        specificity_i   = _safe_div(tn_i, tn_i + fp_i)
                        precision_pos_i = _safe_div(tp_i, tp_i + fp_i)
                        npv_i           = _safe_div(tn_i, tn_i + fn_i)
                        f1s_i = precision_recall_fscore_support(y_i, preds_thr_i_np, average=None, labels=[0, 1])[2]
                        f1_pos_i = float(f1s_i[1]) if f1s_i.size >= 2 else 0.0
                        mcc_i = matthews_corrcoef(y_i, preds_thr_i_np) if (tp_i + fp_i + tn_i + fn_i) > 0 else 0.0

                        # AUPRC 한 번 더(임계값 로깅과 구분)
                        auprc_thr_i = average_precision_score(y_i, pos_scores_i_t.detach().cpu().numpy())

                        # 로깅
                        self.log(f"{mode}_thr_{i}_used", use_thr_i, sync_dist=True)
                        self.log(f"{mode}_TP_{i}", tp_i, sync_dist=True)
                        self.log(f"{mode}_FP_{i}", fp_i, sync_dist=True)
                        self.log(f"{mode}_TN_{i}", tn_i, sync_dist=True)
                        self.log(f"{mode}_FN_{i}", fn_i, sync_dist=True)
                        self.log(f"{mode}_recall_pos_{i}", recall_pos_i, sync_dist=True)
                        self.log(f"{mode}_specificity_{i}", specificity_i, sync_dist=True)
                        self.log(f"{mode}_precision_pos_{i}", precision_pos_i, sync_dist=True)
                        self.log(f"{mode}_npv_{i}", npv_i, sync_dist=True)
                        self.log(f"{mode}_f1_pos_{i}", f1_pos_i, sync_dist=True)
                        self.log(f"{mode}_mcc_{i}", mcc_i, sync_dist=True)
                        self.log(f"{mode}_AUPRC_thr_{i}", auprc_thr_i, sync_dist=True)
                    # === [추가 끝] === 
        
        elif self.hparams.downstream_task_type == 'regression':
            # --- 공통 하이퍼/헬퍼 호출 (가변 길이 지원) ---
            t  = self.hparams.img_size[3]                # window length (e.g., 30)
            ta = self.hparams.num_targets               # #targets per time step
            # 회귀의 경우 보통 마지막 채널은 1이지만, 방어적으로 C를 받아둠
            C  = getattr(self.hparams, "num_classes", 1)

            # subject별: [T_total_s, ta, C], [T_total_s, ta] 의 리스트 반환
            subj_logits_list_by_subj, subj_targets_list_by_subj, kept_subjects = \
                _build_subj_tensors_varlen(subj_array, total_out, subjects, t, ta, C)

            # 평가할 샘플이 없으면 안전 탈출(로그만 남김)
            if len(subj_logits_list_by_subj) == 0:
                self.log(f"{mode}_mse", float('nan'), sync_dist=True)
                self.log(f"{mode}_mae", float('nan'), sync_dist=True)
                self.log(f"{mode}_corrcoef", float('nan'), sync_dist=True)
                self.log(f"{mode}_r2_score", float('nan'), sync_dist=True)
                self.log(f"{mode}_adjusted_mse", float('nan'), sync_dist=True)
                self.log(f"{mode}_adjusted_mae", float('nan'), sync_dist=True)
                return torch.tensor(0.0, device=next(self.parameters()).device)

            # --- 전역(time-step) 평가용 cat (가변 길이 → 리스트 cat) ---
            # logits_all: [(Σ_s T_total_s), ta], targets_all: [(Σ_s T_total_s), ta]
            logits_all_list  = []
            targets_all_list = []
            for L, Y in zip(subj_logits_list_by_subj, subj_targets_list_by_subj):
                # L: [T_total_s, ta, C], Y: [T_total_s, ta]
                # 회귀헤드가 C==1이면 squeeze, 혹 C>1이면 첫 채널만 사용(필요 시 맞게 수정)
                if L.size(-1) == 1:
                    L2 = L.squeeze(-1)                    # [T_total_s, ta]
                else:
                    L2 = L[..., 0]                        # [T_total_s, ta]  (혹은 원하는 채널 선택/평균)
                logits_all_list.append(L2.reshape(-1, ta))
                targets_all_list.append(Y.reshape(-1, ta))

            logits_all  = torch.cat(logits_all_list,  dim=0)   # [(Σ T_s), ta]
            targets_all = torch.cat(targets_all_list, dim=0)   # [(Σ T_s), ta]

            # --- 기본 회귀 지표(전역) ---
            mse = F.mse_loss(logits_all, targets_all)
            mae = F.l1_loss(logits_all, targets_all)

            # 스케일 복원(원척도 지표)
            if self.hparams.label_scaling_method == 'standardization':
                l_adj = logits_all * self.scaler.scale_[0] + self.scaler.mean_[0]
                y_adj = targets_all * self.scaler.scale_[0] + self.scaler.mean_[0]
            elif self.hparams.label_scaling_method == 'minmax':
                rng = (self.scaler.data_max_[0] - self.scaler.data_min_[0])
                l_adj = logits_all * rng + self.scaler.data_min_[0]
                y_adj = targets_all * rng + self.scaler.data_min_[0]
            else:
                l_adj = logits_all
                y_adj = targets_all

            adjusted_mse = F.mse_loss(l_adj, y_adj)
            adjusted_mae = F.l1_loss(l_adj, y_adj)

            # 상관/설명력 (전역)
            pearson = PearsonCorrCoef()
            r2_score = R2Score()
            # numel 보호(샘플 1개면 r2 정의 X)
            if logits_all.numel() >= 2:
                pearson_coef = pearson(logits_all.flatten(), targets_all.flatten())
                r2 = r2_score(logits_all.flatten(), targets_all.flatten())
            else:
                pearson_coef = torch.tensor(0.0, device=logits_all.device, dtype=logits_all.dtype)
                r2 = torch.tensor(0.0, device=logits_all.device, dtype=logits_all.dtype)

            # --- 멀티 타깃일 경우 타깃별 지표 로그(선택) ---
            if ta > 1:
                for i in range(ta):
                    l_i = logits_all[:, i]
                    y_i = targets_all[:, i]

                    mse_i = F.mse_loss(l_i, y_i)
                    mae_i = F.l1_loss(l_i, y_i)

                    # 원척도 복원
                    if self.hparams.label_scaling_method == 'standardization':
                        l_i_adj = l_i * self.scaler.scale_[0] + self.scaler.mean_[0]
                        y_i_adj = y_i * self.scaler.scale_[0] + self.scaler.mean_[0]
                    elif self.hparams.label_scaling_method == 'minmax':
                        rng = (self.scaler.data_max_[0] - self.scaler.data_min_[0])
                        l_i_adj = l_i * rng + self.scaler.data_min_[0]
                        y_i_adj = y_i * rng + self.scaler.data_min_[0]
                    else:
                        l_i_adj, y_i_adj = l_i, y_i

                    adjusted_mse_i = F.mse_loss(l_i_adj, y_i_adj)
                    adjusted_mae_i = F.l1_loss(l_i_adj, y_i_adj)

                    # 상관/설명력(타깃별)
                    if l_i.numel() >= 2:
                        pearson_i = pearson(l_i, y_i)
                        r2_i = r2_score(l_i, y_i)
                    else:
                        pearson_i = torch.tensor(0.0, device=l_i.device, dtype=l_i.dtype)
                        r2_i = torch.tensor(0.0, device=l_i.device, dtype=l_i.dtype)

                    self.log(f"{mode}_mse_{i}", mse_i, sync_dist=True)
                    self.log(f"{mode}_mae_{i}", mae_i, sync_dist=True)
                    self.log(f"{mode}_adjusted_mse_{i}", adjusted_mse_i, sync_dist=True)
                    self.log(f"{mode}_adjusted_mae_{i}", adjusted_mae_i, sync_dist=True)
                    self.log(f"{mode}_corrcoef_{i}", pearson_i, sync_dist=True)
                    self.log(f"{mode}_r2_score_{i}", r2_i, sync_dist=True)

            # --- 전역 로그 ---
            self.log(f"{mode}_mse", mse, sync_dist=True)
            self.log(f"{mode}_mae", mae, sync_dist=True)
            self.log(f"{mode}_adjusted_mse", adjusted_mse, sync_dist=True)
            self.log(f"{mode}_adjusted_mae", adjusted_mae, sync_dist=True)
            self.log(f"{mode}_corrcoef", pearson_coef, sync_dist=True)
            self.log(f"{mode}_r2_score", r2, sync_dist=True)




    def training_step(self, batch, batch_idx):
        """
        Performs a single training step by calculating and returning the loss for the given batch.
        """
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """
        Processes a single validation batch to compute logits and targets, 
        returning subject IDs and corresponding predictions for evaluation.
        """
        subj, logits, target = self._compute_logits(batch) #(b, num_classes)
        if self.hparams.decoder == 'series_decoder': # (batch, T, E) -> (batch, T*E)
            output = [(logit.cpu().detach(), targets.cpu()) for logit, targets in zip(logits, target)] # target is not single value, item() cannot be invoked
        else:
            output = [(logit.cpu().detach(), targets.cpu().item()) for logit, targets in zip(logits, target)]
        return (subj, output) # output은 배치 개수로 구성된 list. output[0]의 경우, [torch.Size([210, 2]), torch.Size([210])]

    def validation_epoch_end(self, outputs):
        """
        Aggregates and processes validation and test outputs at the end of an epoch. 
        Evaluates metrics for both datasets and optionally saves model predictions for future analysis.
        """
        if self.valid_only: # kimbo change
            outputs_valid = outputs  # outputs 자체가 validation 출력 리스트라 가정
            outputs_test = []   
        outputs_valid = outputs[0]
        outputs_test = outputs[1]
        subj_valid = []
        subj_test = []
        out_valid_list = []
        out_test_list = []
        for subj, out in outputs_valid:
            subj_valid += subj
            out_valid_list.append(out)
        for subj, out in outputs_test:
            subj_test += subj
            out_test_list.append(out)
        subj_valid = np.array(subj_valid)
        subj_test = np.array(subj_test)
        total_out_valid = [item for sublist in out_valid_list for item in sublist]
        if not self.valid_only:
            total_out_test = [item for sublist in out_test_list for item in sublist]


        # save model predictions if it is needed for future analysis
        # self._save_predictions(subj_valid,total_out_valid,mode="valid")
        # self._save_predictions(subj_test,total_out_test, mode="test") 
                
        # evaluate 
        self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
        if self.hparams.valid_only == False:
            self._evaluate_metrics(subj_test, total_out_test, mode="test")
            
    # If you use loggers other than Neptune you may need to modify this
    def _save_predictions(self,total_subjs,total_out, mode):
        self.subject_accuracy = {}
        for subj, output in zip(total_subjs,total_out):
            if self.hparams.downstream_task == 'sex':
                score = torch.sigmoid(output[0]).item()
            else:
                score = output[0].item()

            if subj not in self.subject_accuracy:
                self.subject_accuracy[subj] = {'score': [score], 'mode':mode, 'truth':output[1], 'count':1}
            else:
                self.subject_accuracy[subj]['score'].append(score)
                self.subject_accuracy[subj]['count']+=1
        
        if self.hparams.strategy == None : 
            pass
        elif 'ddp' in self.hparams.strategy and len(self.subject_accuracy) > 0:
            world_size = torch.distributed.get_world_size()
            total_subj_accuracy = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(total_subj_accuracy,self.subject_accuracy) # gather and broadcast to whole ranks     
            accuracy_dict = {}
            for dct in total_subj_accuracy:
                for subj, metric_dict in dct.items():
                    if subj not in accuracy_dict:
                        accuracy_dict[subj] = metric_dict
                    else:
                        accuracy_dict[subj]['score']+=metric_dict['score']
                        accuracy_dict[subj]['count']+=metric_dict['count']
            self.subject_accuracy = accuracy_dict
        if self.trainer.is_global_zero:
            for subj_name,subj_dict in self.subject_accuracy.items():
                subj_pred = np.mean(subj_dict['score'])
                subj_error = np.std(subj_dict['score'])
                subj_truth = subj_dict['truth'].item()
                subj_count = subj_dict['count']
                subj_mode = subj_dict['mode'] # train, val, test

                # only save samples at rank 0 (total iterations/world_size numbers are saved) 
                os.makedirs(os.path.join('predictions',self.hparams.id), exist_ok=True)
                with open(os.path.join('predictions',self.hparams.id,'iter_{}.txt'.format(self.current_epoch)),'a+') as f:
                    f.write('subject:{} ({})\ncount: {} outputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_count,subj_pred,subj_error,subj_truth))

            with open(os.path.join('predictions',self.hparams.id,'iter_{}.pkl'.format(self.current_epoch)),'wb') as fw:
                pickle.dump(self.subject_accuracy, fw)

    def test_step(self, batch, batch_idx):
        """
        Processes a single test batch to compute logits and targets, 
        returning subject IDs and corresponding predictions for evaluation.
        """
        subj, logits, target = self._compute_logits(batch) #(b, num_classes)
        if self.hparams.decoder == 'series_decoder': # (batch, T, E) -> (batch, T*E)
            output = [(logit.cpu().detach(), targets.cpu()) for logit, targets in zip(logits, target)] # target is not single value, item() cannot be invoked
        else:
            output = [(logit.cpu().detach(), targets.cpu().item()) for logit, targets in zip(logits, target)]
        return (subj, output)

    def test_epoch_end(self, outputs):
        """
        Aggregates test outputs at the end of an epoch, consolidating subject IDs and predictions for evaluation.
        """
        subj_test = [] 
        out_test_list = []
        for subj, out in outputs:
            subj_test += subj
            out_test_list.append(out)

        subj_test = np.array(subj_test)
        total_out_test = [item for sublist in out_test_list for item in sublist]
                    
        self._evaluate_metrics(subj_test, total_out_test, mode="test")
    
    def on_train_epoch_start(self) -> None:
        """
        Initializes GPU timing events and timing variables to measure training performance 
        at the start of each training epoch.
        """
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.total_time = 0
        self.repetitions = 200
        self.gpu_warmup = 50
        self.timings=np.zeros((self.repetitions,1))
        return super().on_train_epoch_start()
    
    def on_train_batch_start(self, batch, batch_idx):
        """
        Records GPU start timing for selected batches during training 
        to perform scalability checks if enabled.
        """
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.starter.record()
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, out, batch, batch_idx):
        """
        Records GPU end timing and calculates performance metrics such as throughput, mean time, 
        and standard deviation for selected batches during training if scalability checks are enabled.
        """
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.ender.record()
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender) / 1000
                self.total_time += curr_time
                self.timings[batch_idx-self.gpu_warmup] = curr_time
            elif (batch_idx-self.gpu_warmup) == self.repetitions:
                mean_syn = np.mean(self.timings)
                std_syn = np.std(self.timings)
                
                Throughput = (self.repetitions*self.hparams.batch_size*int(self.hparams.num_nodes) * int(self.hparams.devices))/self.total_time
                
                self.log(f"Throughput", Throughput, sync_dist=False)
                self.log(f"mean_time", mean_syn, sync_dist=False)
                self.log(f"std_time", std_syn, sync_dist=False)
                print('mean_syn:',mean_syn)
                print('std_syn:',std_syn)
                
        return super().on_train_batch_end(out, batch, batch_idx)

    def configure_optimizers(self):
        """
        Configures the optimizer (AdamW or SGD) and optionally a learning rate scheduler 
        with warm-up and cosine annealing for training.
        """
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum
            )
        else:
            print("Error: Input a correct optimizer name (default: AdamW)")
        
        if self.hparams.use_scheduler:
            print()
            print("training steps: " + str(self.trainer.estimated_stepping_batches))
            print("using scheduler")
            print()
            total_iterations = self.trainer.estimated_stepping_batches # ((number of samples/batch size)/number of gpus) * num_epochs
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05) # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 1
            
            sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr,min_lr=1e-9, warmup_steps=warmup, gamma=gamma)
            print('total iterations:',self.trainer.estimated_stepping_batches * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        # training related
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for optimizer")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=1.0, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        
        # pretraining-related
        group.add_argument("--use_contrastive", action='store_true', help="whether to use contrastive learning (specify --contrastive_type argument as well)")
        group.add_argument("--contrastive_type", default=0, type=int, help="combination of contrastive losses to use [1: Use the Instance contrastive loss function, 2: Use the local-local temporal contrastive loss function, 3: Use the sum of both loss functions]")
        group.add_argument("--pretraining", action='store_true', help="whether to use pretraining")
        group.add_argument("--augment_during_training", action='store_true', help="whether to augment input images during training")
        group.add_argument("--augment_only_affine", action='store_true', help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true', help="whether to only apply intensity augmentation")
        group.add_argument("--temperature", default=0.1, type=float, help="temperature for NTXentLoss")
        
        # model related
        group.add_argument("--model", type=str, default="none", help="which model to be used")
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--embed_dim", type=int, default=24, help="embedding size (recommend to use 24, 36, 48)")
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int, help="window size from the second layers")
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int, help="first window size")
        group.add_argument("--patch_size", nargs="+", default=[6, 6, 6, 1], type=int, help="patch size")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int, help="depth of layers in each stage")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int, help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2, help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", action='store_true', help="whether to use full-scale multi-head self-attention at the last layers")
        group.add_argument("--clf_head_version", type=str, default="v1", help="clf head version, v2 has a hidden layer")
        group.add_argument("--attn_drop_rate", type=float, default=0, help="dropout rate of attention layers")

        # others
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None, help="Slurm code/PBS code. Use this argument if you want to save process codes to your log")
        group.add_argument("--pos_weight", action='store_true', help="positive class weight for imbalanced datasets") # kimbo change

        # decoder related
        group.add_argument("--num_classes", type=int, default=2, help="Number of distinct target classes")
        group.add_argument("--decoder", type=str, default="single_target_decoder", help="Which decoder to use: (i) single_target_decoder - predict a single value via regression or classification | (ii) series_decoder: predict a series of values (one per timeframe) via regression")
        group.add_argument("--num_targets", type=int, default=7, help="Number of targets to predict in series_decoder")
        # parser.add_argument("--valid_only", action='store_true', help="disable running _evaluate_metrics(mode='test') at validation stage") # kimbo change

        # classification metric related
        group.add_argument("--use_youden_threshold", action='store_true', help="whether to use Youden's J statistic to determine the optimal threshold for binary classification") # kimbo change
        group.add_argument("--threshold_scope", type=str, default="per_target",  choices=['per_target', 'global'], help="Scope of thresholding: 'per_target' applies thresholds individually for each target; 'global' applies a single threshold across all targets.")
        group.add_argument("--positive_class_index", type=int, default=1, help="Index of the positive class for binary classification tasks.")


        return parser