import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle

from torchmetrics import PearsonCorrCoef # Accuracy,
from torchmetrics.regression import R2Score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
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
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from einops import rearrange
from torchmetrics import R2Score, PearsonCorrCoef
from torchmetrics.regression import ConcordanceCorrCoef
from torchmetrics.functional import concordance_corrcoef



sys.path.append('/global/cfs/cdirs/m4750/kimbo/pytorch-softdtw-cuda')
from soft_dtw_cuda import SoftDTW

class LitClassifier(pl.LightningModule):
    
    def __init__(self,data_module, **kwargs):
        super().__init__()

         # ✅ 1. 하이퍼파라미터 먼저 안전하게 저장
        self.data_module = data_module

        base_hparams = self._filter_hparams(data_module, kwargs)
        base_hparams.setdefault("derivative_lambda", kwargs.get("derivative_lambda", 0.1))
        base_hparams.setdefault("softdtw_lambda", kwargs.get("softdtw_lambda", 0.1))
        base_hparams.setdefault("softdtw_gamma",  kwargs.get("softdtw_gamma", 0.05))
        base_hparams.setdefault("ccc_lambda",     kwargs.get("ccc_lambda", 0.1))
        base_hparams.setdefault("efdm_lambda", kwargs.get("efdm_lambda", 1.0))

        self.save_hyperparameters(base_hparams)

        # ✅ 2. 이제 self.hparams.xxx 접근 가능
        self.loss_weights = torch.nn.Parameter(
            torch.ones(self.hparams.num_targets), requires_grad=False
        )
        self.log_vars = torch.nn.Parameter(
            torch.zeros(self.hparams.num_targets), requires_grad=False
        )

        if self.hparams.loss_type in [
            'learnable_weighted_mse',
            'intensity_learnable_weighted_mse',
            'log_intensity_learnable_weighted_mse',
            'normalized_intensity_learnable_weighted_mse']:
            self.loss_weights.requires_grad = True

        elif self.hparams.loss_type == 'uncertainty_weighted_mse':
            self.log_vars.requires_grad = True

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
            # target_values가 정규화되었는지 확인
            print("Normalized target 확인 (평균, 표준편차):")
            print("mean:", normalized_target_values.mean(axis=0))
            print("std:", normalized_target_values.std(axis=0))

        else:
            print("⚠️ No train_dataset provided — skipping target normalization")
            self.scaler = None  # fallback: not used


        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)
        self.output_head = load_model(self.hparams.decoder, self.hparams)
        self.metric = Metrics()
        self.valid_only = kwargs.get("valid_only", False) # kimbo change

    def _filter_hparams(self, data_module, kwargs):
        import copy
        import wandb

        def is_pickleable(v):
            try:
                copy.deepcopy(v)
                return True
            except Exception:
                return False

        # ✅ 1. data_module과 None 제거
        hparams = {
            k: v for k, v in kwargs.items()
            if k != "data_module" and v is not None
        }

        # ✅ 2. wandb 객체 제거
        hparams = {
            k: v for k, v in hparams.items()
            if not isinstance(v, wandb.sdk.wandb_run.Run)
        }

        # ✅ 3. id 필드 문자열화
        if "id" in hparams:
            hparams["id"] = str(hparams["id"])

        # ✅ 4. DDP 환경에서만 Pickle 검증
        if torch.cuda.device_count() > 1:
            hparams = {
                k: v for k, v in hparams.items()
                if is_pickleable(v)
            }

        return hparams
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.load_state_dict(checkpoint['state_dict'], strict=False)

    
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
        Computes logits and normalized targets for classification or regression.
        Compatible with single_target_decoder for sequence-to-single prediction.
        """
        fmri, subj, target_value, tr, sex = batch.values()
       
        if augment_during_training:
            if mode == 'train': # kimbo change
                fmri = self.augment(fmri)

        feature = self.model(fmri)

        # Classification task
        if self.hparams.downstream_task_type == 'classification':
            logits = self.output_head(feature).squeeze() # (b,num_classes)  /  (b,t,num_targets,num_classes)
            target = target_value.float().squeeze()      # (b,num_classes)  /  (b,t,num_targets,num_classes)
            if self.hparams.decoder == 'series_decoder':
                logits = rearrange(logits, 'b t ta c -> b (t ta) c')
                target = rearrange(target, 'b t ta -> b (t ta)')

        # Regression task
        elif self.hparams.downstream_task_type == 'regression':
            logits = self.output_head(feature) # [B, E] (already flattened if single_target_decoder)
            unnormalized_target = target_value.float() # [B, E] or [B, 1]
            
            if self.hparams.decoder == 'series_decoder': # (batch, T, E) -> (batch, T*E)
                logits = logits.view(logits.size(0), -1)
                unnormalized_target = unnormalized_target.view(unnormalized_target.size(0), -1)
            
            if self.hparams.label_scaling_method == 'standardization': # default
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])
            
        return subj, logits, target
    
    # LitClassifier 클래스 안 아무 곳(예: _calculate_loss 위)에 추가
    def _derivative(self, x, dim=1):
        """
        1st-order finite difference along time dim.
        x: (B, T, E) 텐서
        반환: (B, T-1, E)
        """
        return x[:, 1:, :] - x[:, :-1, :]
    
    def _ccc_loss(self, pred, true, eps=1e-8):
        if pred.dim() == 3:
            pred = pred.reshape(pred.size(0), -1)
            true = true.reshape(true.size(0), -1)
        μp, μt = pred.mean(1, keepdim=True), true.mean(1, keepdim=True)
        σp2 = pred.var(1, unbiased=False, keepdim=True)
        σt2 = true.var(1, unbiased=False, keepdim=True)
        cov = ((pred - μp) * (true - μt)).mean(1, keepdim=True)
        ccc = (2 * cov) / (σp2 + σt2 + (μp - μt).pow(2) + eps)
        return 1 - ccc.mean()
    
    def _softdtw_loss(self, pred, target, gamma=0.05):
        if pred.dim() == 3:
            B, T, E = pred.shape
            total_loss = 0.0
            for i in range(E):
                for b in range(B):
                    dist = SoftDTW(gamma=gamma, use_cuda=True)(pred[b, :, i].detach().cpu().numpy(),
                                                target[b, :, i].detach().cpu().numpy())
                    total_loss += dist
            return total_loss / (B * E)
        elif pred.dim() == 2:
            B, T = pred.shape
            total_loss = 0.0
            for b in range(B):
                dist = SoftDTW(gamma=gamma, use_cuda=True)(pred[b].detach().cpu().numpy(),
                                            target[b].detach().cpu().numpy())
                total_loss += dist
            return total_loss / B
        else:
            raise ValueError("Unsupported tensor shape for SoftDTW loss")

    @staticmethod
    def _fdsm_core(pred_flat, target_flat, bins, scaling):
        if scaling == 'minmax':
            bin_edges = torch.linspace(0, 1, bins + 1, device=target_flat.device)
        else:  # standardization
            bin_edges = torch.linspace(-4, 4, bins + 1, device=target_flat.device)

        # bin_ids = torch.bucketize(target_flat, bin_edges, right=False) - 1
        bin_ids = torch.bucketize(target_flat.contiguous(), bin_edges.contiguous(), right=False) - 1
        bin_ids = bin_ids.clamp(min=0, max=bins - 1)

        bin_counts = torch.bincount(bin_ids.flatten(), minlength=bins).float()
        bin_weights = 1.0 / (bin_counts + 1e-6)
        sample_weights = bin_weights[bin_ids]

        error = (pred_flat - target_flat).pow(2)
        weighted_error = sample_weights * error
        return weighted_error.mean()


    @staticmethod
    def fdsm_loss(preds, targets, bins=20, scaling='standardization'):
        """
        Supports:
        - [B, T, E]
        - [B, T]
        - [B]
        """
        assert preds.shape == targets.shape, "Shape mismatch between preds and targets"

        if preds.dim() == 1:  # [B] → 단일 감정 (scalar)
            pred_flat = preds
            target_flat = targets
        elif preds.dim() == 2:  # [B, T]
            pred_flat = preds.reshape(-1)
            target_flat = targets.reshape(-1)
        elif preds.dim() == 3:  # [B, T, E]
            B, T, E = preds.shape
            total_loss = 0.0
            for i in range(E):
                pred_i = preds[..., i].reshape(-1)
                target_i = targets[..., i].reshape(-1)
                total_loss += LitClassifier._fdsm_core(pred_i, target_i, bins, scaling)
            return total_loss / E
        else:
            raise ValueError(f"Unsupported input shape: {preds.shape}")

        return LitClassifier._fdsm_core(pred_flat, target_flat, bins, scaling)
    

    @staticmethod
    def efdm_loss(features, targets, expected_stats_list, reduction='mean'):
        """
        Computes EFDM loss per emotion dimension.

        Args:
            features: [B, T, D]
            targets: [B, T, E]
            expected_stats_list: List[Dict] of length E, each with 'mean' and 'var'
        
        Returns:
            total_loss (scalar), list of (per-emotion loss)
        """
        assert targets.shape[:2] == features.shape[:2], "B, T mismatch"
        B, T, D = features.shape
        E = targets.shape[-1]
        loss_list = []

        # Flatten features
        features_flat = features.view(-1, D)  # [B*T, D]

        for i in range(E):
            stats = expected_stats_list[i]
            target_i = targets[..., i].reshape(-1)  # not used, but possible for masking

            target_mean = stats['mean'].to(features.device)
            target_var = stats['var'].to(features.device)

            batch_mean = features_flat.mean(dim=0)
            batch_var = features_flat.var(dim=0, unbiased=False)

            mean_diff = (batch_mean - target_mean).pow(2).mean()
            var_diff = (batch_var - target_var).pow(2).mean()

            loss_i = (mean_diff + var_diff) / 2 if reduction == 'mean' else (mean_diff + var_diff)
            loss_list.append(loss_i)

        total_loss = sum(loss_list) / E
        return total_loss, loss_list



    def _calculate_loss(self, batch, mode):
        subj, logits, target = self._compute_logits(
            batch, augment_during_training=self.hparams.augment_during_training, mode=mode
        )

        result_dict = {}

        if self.hparams.downstream_task_type == 'classification':
            if self.hparams.decoder == 'series_decoder':
                logits = rearrange(logits, 'b tta c -> (b tta) c')
                target = target.flatten()
            loss = F.cross_entropy(logits, target.long())
            acc = self.metric.get_accuracy(logits, target.float().squeeze())
            result_dict.update({
                f"{mode}_loss": loss,
                f"{mode}_acc": acc,
            })

        elif self.hparams.downstream_task_type == 'regression':
            if self.hparams.decoder == 'series_decoder':
                B, TE = logits.shape
                E = self.hparams.num_targets
                T = TE // E
                logits = logits.view(B, T, E)
                target = target.view(B, T, E)

                loss_list = []

                for i in range(E):
                    pred = logits[:, :, i]       # [B, T]
                    tgt = target[:, :, i]        # [B, T]

                    if self.hparams.loss_type == 'efdm':
                        features = self.model(batch["fmri"])  # [B, T, D]
                        expected_stats_list = self.efdm_stats
                        efdm_total, efdm_per_emotion = self.efdm_loss(features, target, expected_stats_list)
                        result_dict[f"{mode}_efdm_loss"] = efdm_total
                        for i, loss_i in enumerate(efdm_per_emotion):
                            result_dict[f"{mode}_efdm_loss_{i}"] = loss_i
                        loss_list.append(self.hparams.efdm_lambda * efdm_total) 

                    elif self.hparams.loss_type == 'fdsm':
                        scaling = self.hparams.label_scaling_method  # 'standardization' 또는 'minmax'
                        fdsm_i = self.fdsm_loss(pred, tgt, bins=10, scaling=scaling)

                        result_dict[f"{mode}_fdsm_loss_{i}"] = fdsm_i
                        loss_list.append(fdsm_i)
                    
                    else:
                        # 기존 MSE or intensity-weighted MSE 등
                        error = (logits[:, :, i] - target[:, :, i]) ** 2

                        if self.hparams.loss_type == 'intensity_learnable_weighted_mse':
                            intensity = target[:, :, i].abs().detach()
                            error = error * intensity
                        elif self.hparams.loss_type == 'log_intensity_learnable_weighted_mse':
                            intensity = torch.log1p(target[:, :, i].abs()).detach()
                            error = error * intensity
                        elif self.hparams.loss_type == 'normalized_intensity_learnable_weighted_mse':
                            intensity = target[:, :, i].abs().detach()
                            intensity = intensity / (intensity.mean() + 1e-6)
                            error = error * intensity
                        elif self.hparams.loss_type == 'intensity_weighted_mse':
                            intensity = target[:, :, i].abs().detach()
                            error = error * intensity
                            mse_i = error.mean()

                        if self.hparams.loss_type in [
                            'intensity_learnable_weighted_mse',
                            'log_intensity_learnable_weighted_mse',
                            'normalized_intensity_learnable_weighted_mse',
                            'learnable_weighted_mse']:
                            mse_i = self.loss_weights[i] * error.mean()
                        elif self.hparams.loss_type == 'uncertainty_weighted_mse':
                            precision = torch.exp(-self.log_vars[i])
                            mse_i = precision * error.mean() + self.log_vars[i]
                        else:
                            mse_i = error.mean()

                        result_dict[f"{mode}_mse_emotion_{i}"] = mse_i
                        loss_list.append(mse_i)

                base_loss = sum(loss_list) / E
                loss = base_loss
                result_dict[f"{mode}_loss"] = loss

                # Derivative loss
                d_pred = self._derivative(logits)
                d_target = self._derivative(target)
                der_loss = F.mse_loss(d_pred, d_target)
                λ_deriv = self.hparams.derivative_lambda
                loss += λ_deriv * der_loss
                result_dict[f"{mode}_derivative_mse"] = der_loss

                # Soft-DTW
                λ_dtw = self.hparams.softdtw_lambda if self.hparams.loss_type in ["softdtw_mse", "softdtw_ccc_mse"] else 0.0
                if λ_dtw > 0:
                    softdtw_term = self._softdtw_loss(logits, target, gamma=self.hparams.softdtw_gamma)
                    loss += λ_dtw * softdtw_term
                    result_dict[f"{mode}_softdtw"] = softdtw_term

                # CCC
                λ_ccc = self.hparams.ccc_lambda if self.hparams.loss_type in ["ccc_mse", "softdtw_ccc_mse"] else 0.0
                if λ_ccc > 0:
                    ccc_term = self._ccc_loss(logits, target)
                    loss += λ_ccc * ccc_term
                    result_dict[f"{mode}_ccc"] = ccc_term                

            else:
                logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits
                target = target.squeeze(-1) if target.shape[-1] == 1 else target

                # 분기 1: scalar prediction (예: [B])
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    loss = F.mse_loss(logits.squeeze(), target.squeeze())
                    result_dict[f"{mode}_mse"] = loss
                    result_dict[f"{mode}_loss"] = loss
                    l1 = F.l1_loss(logits.squeeze(), target.squeeze())
                    result_dict[f"{mode}_l1_loss"] = l1

                # 분기 2: multitarget regression (예: [B, E])
                else:
                    loss_list = []
                    E = logits.shape[-1]

                    # ✅ EFDM 추가: 감정 루프 전에 1번만
                    if self.hparams.loss_type == 'efdm' and hasattr(self, "efdm_stats"):
                        with torch.no_grad():
                            features = self.model(batch["fmri"])  # [B, D] or [B, T, D]
                            expected_stats_list = self.efdm_stats
                            efdm_total, efdm_per_emotion = self.efdm_loss(features, target, expected_stats_list)

                        result_dict[f"{mode}_efdm_loss"] = efdm_total
                        for i, loss_i in enumerate(efdm_per_emotion):
                            result_dict[f"{mode}_efdm_loss_{i}"] = loss_i
                        loss_list.append(self.hparams.efdm_lambda * efdm_total)

                        print(f"[EFDM DEBUG] step={self.global_step} | mode={mode} | efdm_total={efdm_total.item():.4f}")
                        for i, loss_i in enumerate(efdm_per_emotion):
                            print(f"  [EFDM emotion {i}] loss={loss_i.item():.4f}")

                        loss_list.append(self.hparams.efdm_lambda * efdm_total)

                    for i in range(E):
                        pred = logits[:, i]
                        tgt = target[:, i]

                        mse_i = F.mse_loss(pred, tgt)
                        mae_i = F.l1_loss(pred, tgt)

                        result_dict[f"{mode}_mse_emotion_{i}"] = mse_i
                        result_dict[f"{mode}_mae_emotion_{i}"] = mae_i
                        loss_list.append(mse_i)

                    base_loss = sum(loss_list) / E
                    loss = base_loss
                    result_dict[f"{mode}_loss"] = loss

        self.log_dict(
            result_dict,
            prog_bar=True,
            sync_dist=False,
            add_dataloader_idx=False,
            on_step=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size
        )

        return loss


    # def on_fit_start(self):
    #     import os, torch

    #     # 1. split 이름 파싱 및 공백 제거
    #     split_csv = getattr(self.data_module, "split_name", "split_unknown.csv")
    #     split_tag = os.path.splitext(split_csv)[0].strip()  # "split_seed42" 등
    #     stats_fname = f"efdm_stats_{split_tag}.pth"
    #     stats_path = os.path.join(self.hparams.default_root_dir, stats_fname)

    #     # 2. 이미 저장된 통계가 있으면 로드 후 종료
    #     if os.path.exists(stats_path):
    #         print(f"📂 EFDM stats already exist for [{split_tag}]. Loading from cache...")
    #         self.efdm_stats = torch.load(stats_path, map_location="cpu")
    #         return  # ⛔ 계산 스킵

    #     # 3. 존재하지 않으면 계산 시작
    #     print(f"🧮 Computing EFDM stats for split [{split_tag}] ...")
    #     all_feats = []

    #     train_loader = self.data_module.train_dataloader()
    #     self.model.eval()

    #     with torch.no_grad():
    #         for batch in train_loader:
    #             fmri = batch["fmri_sequence"].to(self.device).float()  # [B, C, D, H, W, T]
    #             feats = self.model(fmri)                               # [B, T, D]
    #             feats = feats.permute(0, 2, 1).contiguous()            # [B, D, T]
    #             flat_feats = feats.view(-1, feats.size(-1))            # [B*T, D]
    #             all_feats.append(flat_feats)

    #     # 4. 평균 및 분산 계산
    #     flat = torch.cat(all_feats, dim=0)
    #     mean = flat.mean(dim=0)
    #     var  = flat.var(dim=0, unbiased=False)
    #     self.efdm_stats = {"mean": mean, "var": var}

    #     # 5. 통계 저장
    #     os.makedirs(self.hparams.default_root_dir, exist_ok=True)
    #     torch.save(self.efdm_stats, stats_path)
    #     print(f"✅ EFDM stats saved to {stats_path}")

    def on_fit_start(self):
        import os, torch

        # 1. split 이름 파싱 및 경로 설정
        split_csv = getattr(self.data_module, "split_name", "split_unknown.csv")
        split_tag = os.path.splitext(split_csv)[0].strip()

        # 마스킹 여부에 따라 파일 이름 다르게 설정
        use_mask = getattr(self.hparams, "efdm_mask", False)
        mask_tag = "_masked" if use_mask else "_all"

        stats_fname = f"efdm_stats_{split_tag}{mask_tag}.pth"
        stats_path = os.path.join(self.hparams.default_root_dir, stats_fname)

        # 2. 통계 캐시가 이미 존재하면 로딩 후 종료
        if os.path.exists(stats_path):
            print(f"📂 EFDM stats already exist for [{split_tag}{mask_tag}]. Loading from cache...")
            self.efdm_stats = torch.load(stats_path, map_location="cpu")
            return

        # 3. 새로 계산 시작
        print(f"🧮 Computing EFDM stats for split [{split_tag}] (mask: {use_mask}) ...")
        all_feats = []

        train_loader = self.data_module.train_dataloader()
        self.model.eval()

        # 4. target 기준 마스킹 조건 처리
        all_targets = []
        if use_mask:
            with torch.no_grad():
                for batch in train_loader:
                    target = batch["target"]  # [B, T, E]
                    all_targets.append(target.view(-1, target.shape[-1]))  # [B*T, E]
            all_targets = torch.cat(all_targets, dim=0)  # [N, E]
            target_medians = all_targets.median(dim=0).values  # [E]
            print(f"📊 Median threshold per target: {target_medians}")

        # 5. feature 추출 및 마스킹 적용
        with torch.no_grad():
            for batch in train_loader:
                fmri   = batch["fmri_sequence"].to(self.device).float()  # [B, C, D, H, W, T]
                target = batch["target"].to(self.device).float()         # [B, T, E]

                feats = self.model(fmri)                     # [B, T, D]
                feats = feats.permute(0, 2, 1).contiguous()  # [B, D, T]
                flat_feats = feats.view(-1, feats.size(-1))  # [B*T, D]
                flat_tgt   = target.view(-1, target.shape[-1])  # [B*T, E]

                if use_mask:
                    med = target_medians.to(self.device)     # [E]
                    mask = (flat_tgt > med).any(dim=-1)      # [B*T]
                    flat_feats = flat_feats[mask]

                all_feats.append(flat_feats)

        # 6. 통계 계산
        flat = torch.cat(all_feats, dim=0)
        mean = flat.mean(dim=0)
        var  = flat.var(dim=0, unbiased=False)
        self.efdm_stats = {"mean": mean, "var": var}

        # 7. 저장
        os.makedirs(self.hparams.default_root_dir, exist_ok=True)
        torch.save(self.efdm_stats, stats_path)
        print(f"✅ EFDM stats saved to {stats_path}")





    def _evaluate_metrics(self, subj_array, total_out, mode, best=False):
        mode_str = mode if not best else f'best_{mode}'
        subjects = np.unique(subj_array)

        subj_avg_logits, subj_targets = [], []

        for subj in subjects:
            subj_logits = [total_out[i][0] for i in range(len(subj_array)) if subj_array[i] == subj]
            subj_target = [total_out[i][1] for i in range(len(subj_array)) if subj_array[i] == subj][0]
            subj_targets.append(subj_target)

            if self.hparams.decoder == 'series_decoder':
                # subj_logits: list of [1, T, 7]
                subj_avg_logits.append(subj_logits)
            else:  # single_target_decoder
                # subj_logits: list of [7]
                subj_avg_logits.append(torch.mean(torch.stack(subj_logits), dim=0))  # → [7]

        # Stack tensors
        if self.hparams.decoder == 'series_decoder':
            subj_avg_logits = [x[0] for x in subj_avg_logits]  # remove dummy dim: [1, T, 7] → [T, 7]
            subj_avg_logits = torch.stack(subj_avg_logits)     # [B, T, 7]
            subj_targets = torch.stack(subj_targets)           # [B, T, 7]
            logits = subj_avg_logits.view(-1, self.hparams.num_targets)
            targets = subj_targets.view(-1, self.hparams.num_targets)

        else:  # single_target_decoder
            subj_avg_logits = torch.stack(subj_avg_logits)     # [B, E] or [B]
            subj_targets = torch.stack(subj_targets)           # [B, E] or [B]
            logits, targets = subj_avg_logits, subj_targets

            # flatten to [B] if scalar
            if logits.ndim == 1 or logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
                targets = targets.squeeze(-1)

        # Metric logging
        if self.hparams.downstream_task_type == 'regression':
            if self.hparams.decoder == 'series_decoder':
                # [B, T, 7] → flatten
                logits = subj_avg_logits.view(-1, self.hparams.num_targets)
                targets = subj_targets.view(-1, self.hparams.num_targets)
                self._log_base_metrics(logits, targets, mode_str)
                self._log_per_target_metrics(logits, targets, mode_str)

            else:
                logits, targets = subj_avg_logits, subj_targets  # [B, 7]
                # 안전하게 squeeze
                if logits.ndim == 2 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                    targets = targets.squeeze(-1)

                # log base metrics (항상 실행)
                self._log_base_metrics(logits, targets, mode_str)

                # multitarget일 때만 per-target metrics 실행
                if logits.ndim == 2 and logits.shape[-1] > 1:
                    self._log_per_target_metrics(logits, targets, mode_str)

            # EFDM: 항상 decoder 관계없이 실행
            if self.hparams.loss_type == "efdm" and hasattr(self, "efdm_stats"):
                self._log_efdm_metrics(mode_str)

    def _log_base_metrics(self, logits, targets, mode_str):
        # logits, targets: [B, 7]
        mse = F.mse_loss(logits, targets)
        mae = F.l1_loss(logits, targets)

        if self.hparams.label_scaling_method == 'standardization':
            logits_adj = logits * self.scaler.scale_[0] + self.scaler.mean_[0]
            targets_adj = targets * self.scaler.scale_[0] + self.scaler.mean_[0]
        else:  # minmax
            scale = self.scaler.data_max_[0] - self.scaler.data_min_[0]
            logits_adj = logits * scale + self.scaler.data_min_[0]
            targets_adj = targets * scale + self.scaler.data_min_[0]

        adjusted_mse = F.mse_loss(logits_adj, targets_adj)
        adjusted_mae = F.l1_loss(logits_adj, targets_adj)

        pearson = PearsonCorrCoef()(logits.flatten(), targets.flatten())
        ccc = concordance_corrcoef(logits.flatten(), targets.flatten())
        r2_score_val = R2Score()(logits.flatten(), targets.flatten()) if len(logits) >= 2 else 0

        for name, val in zip(["mse", "mae", "adjusted_mse", "adjusted_mae", "corrcoef", "r2_score", "ccc"],
                            [mse, mae, adjusted_mse, adjusted_mae, pearson, r2_score_val, ccc]):
            self.log(f"{mode_str}_{name}", val, sync_dist=True)

    
    def _log_per_target_metrics(self, logits, targets, mode_str):
        # logits, targets: [B, 7] or [B*T, 7]
        nt = self.hparams.num_targets
        for i in range(nt):
            l_i = logits[:, i]
            t_i = targets[:, i]

            mse_i = F.mse_loss(l_i, t_i)
            mae_i = F.l1_loss(l_i, t_i)
            # 안전 처리
            if l_i.numel() < 2:
                r2_i = torch.tensor(0.0, device=l_i.device)
                corr_i = torch.tensor(0.0, device=l_i.device)
                ccc_i  = torch.tensor(0.0, device=l_i.device)
            else:
                r2_i = R2Score()(l_i, t_i)
                corr_i = PearsonCorrCoef()(l_i, t_i)
                ccc_i  = concordance_corrcoef(l_i, t_i)

            if self.hparams.label_scaling_method == 'standardization':
                l_adj = l_i * self.scaler.scale_[0] + self.scaler.mean_[0]
                t_adj = t_i * self.scaler.scale_[0] + self.scaler.mean_[0]
                fdsm = self.fdsm_loss(l_i, t_i, scaling='standardization')
            else:
                scale = self.scaler.data_max_[0] - self.scaler.data_min_[0]
                l_adj = l_i * scale + self.scaler.data_min_[0]
                t_adj = t_i * scale + self.scaler.data_min_[0]
                fdsm = self.fdsm_loss(l_i, t_i, scaling='minmax')

            adj_mse_i = F.mse_loss(l_adj, t_adj)
            adj_mae_i = F.l1_loss(l_adj, t_adj)

            for name, val in zip(
                ["mse", "mae", "adjusted_mse", "adjusted_mae", "corrcoef", "r2_score", "ccc", "fdsm_loss"],
                [mse_i, mae_i, adj_mse_i, adj_mae_i, corr_i, r2_i, ccc_i, fdsm]
            ):
                self.log(f"{mode_str}_{name}_{i}", val, sync_dist=True)


    def _log_efdm_metrics(self, mode: str):
        """
        EFDM (expected feature distribution matching) 지표를 계산해 로깅합니다.
        - mode: 'val' 또는 'best_val' 등 로그 키에 사용될 모드 문자열
        """
        # 1) DataModule에서 반환된 모든 DataLoader를 가져옵니다.
        val_loaders = self.data_module.val_dataloader()  # List[DataLoader]
        all_feats, all_tgts = [], []

        with torch.no_grad():
            # 2) 각 DataLoader, 각 배치 순회
            for loader in val_loaders:
                for batch in loader:
                    # ── 배치에서 필요한 텐서 꺼내기 ──
                    fmri_seq     = batch["fmri_sequence"].to(self.device).float()  # [B, C, D, H, W, T]
                    target_value = batch["target"].to(self.device)                # [B, T, E] or [B, E]

                    # ── 3) feature 추출 및 시퀀스 차원 정렬 ──
                    feats = self.model(fmri_seq)                                   # [B, D] or [B, T, D]
                    # 토큰 차원이 2번째 축일 때 (B, D, N) → (B, N, D)
                    if feats.ndim == 3:
                        feats = feats.permute(0, 2, 1).contiguous()               # [B, T, D]
                        B, T, D = feats.shape
                        flat = feats.reshape(B * T, D)                           # [B*T, D]
                    else:
                        # 만약 [B, D] 형태라면 시퀀스 차원이 없으므로 그대로
                        flat = feats                                              # [B, D]
                    all_feats.append(flat)

                    # ── 4) target 펼치기 ──
                    if target_value.ndim == 3:
                        B, T, E = target_value.shape
                        tgt_flat = target_value.reshape(B * T, E)                 # [B*T, E]
                    else:
                        tgt_flat = target_value                                   # [B, E]
                    all_tgts.append(tgt_flat)

        # ── 5) 전체 concat ──
        features_flat = torch.cat(all_feats, dim=0)  # [N, D]
        targets_flat  = torch.cat(all_tgts,  dim=0)  # [N, E]

        # ── 6) 감정(emotion)별로 EFD M 비교 ──
        for i in range(self.hparams.num_targets):
            expected = self.efdm_stats[i]
            # 감정 i가 활성화된 프레임만 골라낼 mask
            mask = targets_flat[:, i] > 0.0
            feats_i = features_flat[mask] if mask.any() else features_flat

            # mean / var shift 계산
            mean_shift = (feats_i.mean(dim=0) - expected["mean"].to(self.device)).pow(2).mean()
            var_shift  = (feats_i.var(dim=0, unbiased=False) - expected["var"].to(self.device)).pow(2).mean()

            # ── 7) 로깅 ──
            self.log(f"{mode}_efdm_mean_shift_{i}", mean_shift, sync_dist=True)
            self.log(f"{mode}_efdm_var_shift_{i}",  var_shift,  sync_dist=True)



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
            output = []
            for logit, targets in zip(logits, target):
                logit = logit.cpu().detach()
                targets = targets.cpu()

                if targets.numel() == 1:
                    output.append((logit, targets.item()))  # scalar case
                else:
                    output.append((logit, targets))         # multitask (e.g. [7])

        return (subj, output)

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
        subj, logits, target = self._compute_logits(batch)  # (B, ...)

        if self.hparams.decoder == 'series_decoder':
            # logits: [B, T*E], target: [B, T*E]
            output = [
                (logit.cpu().detach(), targets.cpu())
                for logit, targets in zip(logits, target)
            ]
        else:
            # single_target_decoder: scalar or multitask
            output = []
            for logit, targets in zip(logits, target):
                logit = logit.cpu().detach()
                targets = targets.cpu()

                if targets.numel() == 1:
                    output.append((logit, targets.item()))  # scalar prediction
                else:
                    output.append((logit, targets))         # multitask prediction

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
        
        # decoder related
        group.add_argument("--num_classes", type=int, default=2, help="Number of distinct target classes")
        group.add_argument(
            "--decoder",
            type=str,
            default="single_target_scalar",
            help=(
                "Which decoder to use:\n"
                "(i) single_target_scalar - predict a single scalar value (e.g., age, sex) via regression or classification\n"
                "(ii) single_target_multitask - predict multiple independent scalar values (e.g., 7 emotion scores) via regression\n"
                "(iii) series_decoder - predict a time series of values (one per timeframe) via regression"
            )
        )
        group.add_argument("--num_targets", type=int, default=7, help="Number of targets to predict in series_decoder")

        # loss related
        group.add_argument("--loss_type", type=str, default="mean_mse",
                   choices=[
                    "mean_mse", "weighted_mse_norm", "weighted_mse_var", "weighted_mse_both",
                    "learnable_weighted_mse", "uncertainty_weighted_mse",
                    "intensity_weighted_mse", "intensity_learnable_weighted_mse",
                    "log_intensity_learnable_weighted_mse", "normalized_intensity_learnable_weighted_mse",
                    "softdtw_mse", "ccc_mse", "softdtw_ccc_mse", 
                    "fdsm", "efdm"],
                   help="Loss function type for series decoder: basic or weighted")
        group.add_argument(
            "--derivative_lambda",
            type=float,
            default=0.0,                 # 0이면 비활성
            help="가치 변화율(1차 차분) MSE 가중치 λ; 0이면 파생 손실을 사용하지 않음"
        )
        # ② Soft-DTW · CCC 하이퍼파라미터 기본값
        group.add_argument("--softdtw_lambda", type=float, default=0.1,
                        help="softdtw_mse, softdtw_ccc_mse 사용 시 가중치 λ")
        group.add_argument("--softdtw_gamma", type=float, default=0.05,
                        help="Soft-DTW γ (부드러움 정도)")
        group.add_argument("--ccc_lambda", type=float, default=0.1,
                        help="ccc_mse, softdtw_ccc_mse 사용 시 가중치 λ")
        group.add_argument("--fdsm_lambda", type=float, default=0.3,
                   help="fdsm loss의 smoothing 가중치 λ (MAE + λ × smooth)")
        group.add_argument("--efdm_lambda", type=float, default=1.0,
                   help="Weight for EFDM loss when used as part of total loss.")
        group.add_argument("--efdm_mask", action="store_true",
                   help="Apply median-based masking for EFDM stats (use only target > median)")


        return parser