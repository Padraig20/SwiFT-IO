import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle

from torchmetrics import PearsonCorrCoef # Accuracy,
from torchmetrics.regression import R2Score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import monai.transforms as monai_t

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from .models.load_model import load_model
from .utils.metrics import Metrics
from .utils.lr_scheduler import CosineAnnealingWarmUpRestarts
from .utils.learnable_losses import (
    PerEmotionLearnableWeightedMSE,
    UncertaintyWeightedMSE,
    FocalMSELoss,
    WeightedFocalMSELoss,
    NormalizedFocalMSELoss,
    TweedieLoss
)

from einops import rearrange

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wandb 
import copy
import pdb
class LitClassifier(pl.LightningModule):

    # Emotion names for logging (matches data_module.py emotion order)
    EMOTION_NAMES = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']

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

        # ✅ 6. 안전한 값만 `save_hyperparameters()`에 전달
        self.save_hyperparameters(hparams)

        # # you should define target_values at the Dataset classes
        # target_values = data_module.train_dataset.target_values
        # if self.hparams.label_scaling_method == 'standardization':
        #     scaler = StandardScaler()
        #     normalized_target_values = scaler.fit_transform(target_values)
        #     print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        # elif self.hparams.label_scaling_method == 'minmax': 
        #     scaler = MinMaxScaler()
        #     normalized_target_values = scaler.fit_transform(target_values)
        #     print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        # self.scaler = scaler

        # you should define target_values at the Dataset classes
        if data_module and hasattr(data_module, "train_dataset"):
            target_values = data_module.train_dataset.target_values

            # Filter out invalid values (object dtype or non-numeric)
            import numpy as np
            if target_values.dtype == np.object_:
                print(f"⚠️ Warning: target_values has object dtype. Converting to float and filtering invalid values...")
                # Try to convert to float, filtering out invalid entries
                valid_mask = np.ones(len(target_values), dtype=bool)
                for i, val in enumerate(target_values):
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        valid_mask[i] = False
                        print(f"  Skipping invalid value at index {i}: {val}")

                target_values = target_values[valid_mask].astype(np.float32)
                print(f"  Filtered {np.sum(~valid_mask)} invalid values. Remaining: {len(target_values)}")

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

        # Initialize optimal thresholds (will be calculated on validation set)
        self.optimal_thresholds = None

        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)
        
        self.output_head = load_model(self.hparams.decoder, self.hparams)

        self.metric = Metrics()

        self.valid_only = kwargs.get("valid_only", False) # kimbo change

        # Initialize learnable loss functions for regression tasks
        if self.hparams.downstream_task_type == 'regression':
            loss_type = self.hparams.get('regression_loss_type', 'mse')
            if loss_type == 'per_emotion_weighted':
                print(f"\n{'='*80}")
                print("Using Per-Emotion Learnable Weighted MSE Loss")
                print(f"{'='*80}\n")
                self.learnable_loss = PerEmotionLearnableWeightedMSE(
                    num_emotions=self.hparams.num_targets,
                    init_zero_weight=self.hparams.get('init_zero_weight', 0.1),
                    init_nonzero_weight=self.hparams.get('init_nonzero_weight', 5.0)
                )
            elif loss_type == 'uncertainty_weighted':
                print(f"\n{'='*80}")
                print("Using Uncertainty-based Weighted MSE Loss")
                print(f"{'='*80}\n")
                self.learnable_loss = UncertaintyWeightedMSE(
                    num_emotions=self.hparams.num_targets,
                    init_log_var=self.hparams.get('init_log_var', 0.0)
                )
            elif loss_type == 'focal_mse':
                print(f"\n{'='*80}")
                print("Using Focal MSE Loss")
                print(f"  Gamma: {self.hparams.get('focal_gamma', 2.0)}")
                print(f"{'='*80}\n")
                self.learnable_loss = FocalMSELoss(
                    gamma=self.hparams.get('focal_gamma', 2.0)
                )
            elif loss_type == 'weighted_focal_mse':
                print(f"\n{'='*80}")
                print("Using Weighted Focal MSE Loss")
                print(f"  Gamma: {self.hparams.get('focal_gamma', 2.0)}")
                print(f"  Zero weight: {self.hparams.get('zero_weight', 1.0)}")
                print(f"  Non-zero weight: {self.hparams.get('nonzero_weight', 5.0)}")
                print(f"{'='*80}\n")
                self.learnable_loss = WeightedFocalMSELoss(
                    gamma=self.hparams.get('focal_gamma', 2.0),
                    zero_weight=self.hparams.get('zero_weight', 1.0),
                    nonzero_weight=self.hparams.get('nonzero_weight', 5.0)
                )
            elif loss_type == 'normalized_focal_mse':
                print(f"\n{'='*80}")
                print("Using Normalized Focal MSE Loss (Scale-Robust)")
                print(f"  Gamma: {self.hparams.get('focal_gamma', 1.0)}")
                print(f"  Epsilon: {self.hparams.get('focal_eps', 1e-6)}")
                print(f"  Scale-invariant: Handles different emotion ranges (Positive: 0-27, Sad: 0-5)")
                print(f"{'='*80}\n")
                self.learnable_loss = NormalizedFocalMSELoss(
                    gamma=self.hparams.get('focal_gamma', 1.0),
                    eps=self.hparams.get('focal_eps', 1e-6)
                )
            elif loss_type == 'tweedie':
                print(f"\n{'='*80}")
                print("Using Tweedie Loss for Zero-Inflated Regression")
                print(f"  Power parameter p: {self.hparams.get('tweedie_p', 1.5)}")
                print(f"  (1 < p < 2, p=1.5 recommended for zero-inflated data)")
                print(f"{'='*80}\n")
                self.learnable_loss = TweedieLoss(
                    p=self.hparams.get('tweedie_p', 1.5)
                )
            else:
                print(f"\nUsing standard MSE loss\n")
                self.learnable_loss = None
        else:
            self.learnable_loss = None

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
                fmri = self.augment(fmri)

        feature = self.model(fmri)

        # Classification task
        if self.hparams.downstream_task_type == 'classification':
            logits = self.output_head(feature)  # (b, num_classes) or (b, t, num_targets, num_classes)
            target = target_value.float().squeeze()  # (b, num_classes) or (b, t, num_targets)
            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
                logits = rearrange(logits, 'b t ta c -> b (t ta) c')
                target = rearrange(target, 'b t ta -> b (t ta)')
            else:
                # For non-series decoders, squeeze if needed
                if logits.dim() > 2:
                    logits = logits.squeeze()
                if target.dim() > 2:
                    target = target.squeeze()
        # Regression task
        elif self.hparams.downstream_task_type == 'regression':

            logits = self.output_head(feature) # (b,1) or (b, num_targets)
            unnormalized_target = target_value.float() # (b,1) or (b, t, num_targets)

            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']: # (batch, T, E) -> (batch, T*E)
                logits = logits.view(logits.size(0), -1)
                unnormalized_target = unnormalized_target.view(unnormalized_target.size(0), -1)
            elif self.hparams.decoder == 'lstm_regression_head':
                # LSTM outputs single prediction per sequence, average target across time
                # unnormalized_target shape: (batch, time, num_targets)
                # logits shape: (batch, num_targets)
                if unnormalized_target.dim() == 3:
                    unnormalized_target = unnormalized_target.mean(dim=1)  # (batch, num_targets)

            if self.hparams.label_scaling_method == 'standardization': # default
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])
            
        return subj, logits, target
    
    # def _calculate_loss(self, batch, mode):
    #     """
    #     Calculates the loss and performance metrics for classification or regression tasks. 
    #     Logs the results for monitoring during training or evaluation.
    #     """
    #     subj, logits, target = self._compute_logits(batch, augment_during_training = self.hparams.augment_during_training)

    #     if self.hparams.downstream_task_type == 'classification':
    #         if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']: # [b, (t ta), c] -> [(b t ta), c]
    #             logits = rearrange(logits, 'b tta c -> (b tta) c')
    #             target = target.flatten() # (b,c) -> (b*c)
    #         loss = F.cross_entropy(logits, target.long()) # target is float
    #         acc = self.metric.get_accuracy(logits, target.float().squeeze())
    #         result_dict = {
    #             f"{mode}_loss": loss,
    #             f"{mode}_acc": acc,
    #         }

    #     elif self.hparams.downstream_task_type == 'regression':
    #         loss = F.mse_loss(logits.squeeze(), target.squeeze())
    #         l1 = F.l1_loss(logits.squeeze(), target.squeeze())
    #         result_dict = {
    #             f"{mode}_loss": loss,
    #             f"{mode}_mse": loss,
    #             f"{mode}_l1_loss": l1
    #         }
    #     self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)
    #     return loss
    
    def _calculate_loss(self, batch, mode):
        subj, logits, target = self._compute_logits(batch, augment_during_training=self.hparams.augment_during_training, mode=mode)

        result_dict = {}

        if self.hparams.downstream_task_type == 'classification':
            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
                logits = rearrange(logits, 'b tta c -> (b tta) c')
                target = target.flatten()
            loss = F.cross_entropy(logits, target.long())
            acc = self.metric.get_accuracy(logits, target.float().squeeze())
            result_dict.update({
                f"{mode}_loss": loss,
                f"{mode}_acc": acc,
            })

        elif self.hparams.downstream_task_type == 'regression':
            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
                B, TE = logits.shape  # (B, T*E)
                E = self.hparams.num_targets
                T = TE // E
                logits = logits.view(B, T, E)
                target = target.view(B, T, E)

                # Use learnable loss if available
                if self.learnable_loss is not None:
                    loss = self.learnable_loss(logits, target)

                    # Still log per-emotion MSE for monitoring with emotion names
                    for i in range(E):
                        emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"
                        mse_i = F.mse_loss(logits[:, :, i], target[:, :, i])
                        result_dict[f"{mode}_mse_{emotion_name}"] = mse_i
                else:
                    # Standard MSE loss (original behavior)
                    loss_list = []
                    for i in range(E):
                        emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"
                        mse_i = F.mse_loss(logits[:, :, i], target[:, :, i])
                        result_dict[f"{mode}_mse_{emotion_name}"] = mse_i
                        loss_list.append(mse_i)
                    loss = sum(loss_list) / E
            else:
                # Use learnable loss if available
                if self.learnable_loss is not None:
                    loss = self.learnable_loss(logits.squeeze(), target.squeeze())
                else:
                    loss = F.mse_loss(logits.squeeze(), target.squeeze())
                result_dict[f"{mode}_mse"] = loss

            l1 = F.l1_loss(logits.squeeze(), target.squeeze())
            result_dict[f"{mode}_loss"] = loss
            result_dict[f"{mode}_l1_loss"] = l1

        # ✅ loss 및 각 감정별 mse를 wandb에 기록
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


    def _evaluate_metrics(self, subj_array, total_out, mode, best=False):
        """
        Evaluates classification or regression metrics for aggregated subject-level predictions. 
        Logs accuracy, balanced accuracy, and AUROC for classification tasks, and MSE, MAE, and correlation coefficients for regression tasks, including metrics on the original scale.
        """
        mode_str = mode if best == False else 'best_'+mode # kimbo change
        subjects = np.unique(subj_array)
        
        subj_avg_logits = []
        subj_targets = []
        for subj in subjects:
            subj_logits = [total_out[i][0] for i in range(len(subj_array)) if subj_array[i] == subj]
            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']: # do not calculate the average logits
                subj_avg_logits.append(subj_logits)
            else:
                subj_avg_logits.append(torch.mean(torch.stack(subj_logits), dim=0))
            subj_targets.append([total_out[i][1] for i in range(len(subj_array)) if subj_array[i] == subj][0])

        if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
            subj_avg_logits = [i[0] for i in subj_avg_logits] # unpack single values from the list
            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.stack(subj_targets)
        else:
            subj_avg_logits = torch.stack(subj_avg_logits)
            # Check if targets are already tensors (multi-target regression)
            if len(subj_targets) > 0 and isinstance(subj_targets[0], torch.Tensor):
                subj_targets = torch.stack(subj_targets)
            else:
                subj_targets = torch.tensor(subj_targets)
    
        if self.hparams.downstream_task_type == 'classification':
            
            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
                subj_avg_logits = rearrange(subj_avg_logits, 'b tta c -> (b tta) c')
                subj_targets = subj_targets.flatten()
                
            num_classes = subj_avg_logits.shape[1]
            
            probabilities = F.softmax(subj_avg_logits.to(dtype=torch.float32), dim=1) # (b,num_classes), require 32 bit precision
            predictions = probabilities.argmax(dim=1) # (b)
            
            predictions_np = predictions.cpu().numpy()
            targets_np = subj_targets.cpu().numpy()

            accuracy = accuracy_score(targets_np, predictions_np)
            balanced_accuracy = balanced_accuracy_score(targets_np, predictions_np)

            # ROC AUC calculation (handle case where only one class is present)
            try:
                if num_classes == 2:
                    roc_auc = roc_auc_score(targets_np, predictions_np)
                else:
                    targets_one_hot = label_binarize(targets_np, classes=np.arange(num_classes))
                    roc_auc = roc_auc_score(targets_one_hot, probabilities.cpu().detach().numpy(), multi_class='ovr')
            except ValueError as e:
                # Only one class present in y_true (common in sanity check with 2 batches)
                print(f"[WARNING] ROC AUC calculation failed: {e}")
                print(f"[WARNING] Unique classes in targets: {np.unique(targets_np)}")
                roc_auc = float('nan')  # Mark as undefined rather than guessing 0.5

            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
                
                # evaluate multiple targets separately
                t = self.hparams.img_size[3]

                subj_avg_logits = rearrange(subj_avg_logits, '(b t ta) c -> b t ta c', t=t, ta=self.hparams.num_targets, c=self.hparams.num_classes)
                subj_targets = rearrange(subj_targets, '(b t ta) -> b t ta', t=t, ta=self.hparams.num_targets)
            
                for i in range(self.hparams.num_targets):
                    logits_group = subj_avg_logits[:,:,i]  # Shape: [batch_size, temporal_size, num_classes]
                    target_group = subj_targets[..., i]
                    
                    probabilities = F.softmax(logits_group.to(dtype=torch.float32), dim=-1) # (b, temporal_size, num_classes), require 32 bit precision

                    # Use optimal threshold if available, otherwise use argmax (threshold=0.5)
                    if self.optimal_thresholds and i in self.optimal_thresholds and mode == 'test':
                        opt_thr = self.optimal_thresholds[i]
                        predictions = (probabilities[..., 1] >= opt_thr).long()  # (b, temporal_size)
                        if self.trainer.is_global_zero:  # Only print once in distributed training
                            print(f"  [INFO] Emotion {i} ({mode}): Using optimal threshold {opt_thr:.4f}")
                    else:
                        predictions = probabilities.argmax(dim=-1) # (b, temporal_size)
                    
                    predictions_np = predictions.flatten().cpu().numpy()
                    targets_np = target_group.flatten().cpu().numpy()
                    
                    accuracy_group = accuracy_score(targets_np, predictions_np)
                    balanced_accuracy_group = balanced_accuracy_score(targets_np, predictions_np)

                    # ROC AUC calculation per emotion (handle case where only one class is present)
                    try:
                        if num_classes == 2:
                            roc_auc_group = roc_auc_score(targets_np, predictions_np)
                        else:
                            targets_one_hot = label_binarize(targets_np, classes=np.arange(num_classes))
                            roc_auc_group = roc_auc_score(targets_one_hot, rearrange(probabilities, 'b t c -> (b t) c').cpu().detach().numpy(), multi_class='ovr')
                    except ValueError as e:
                        # Only one class present in y_true for this emotion (common in sanity check)
                        print(f"[WARNING] ROC AUC calculation failed for emotion {i}: {e}")
                        print(f"[WARNING] Unique classes for emotion {i}: {np.unique(targets_np)}")
                        roc_auc_group = float('nan')  # Mark as undefined rather than guessing 0.5

                    emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"
                    self.log(f"{mode_str}_acc_{emotion_name}", accuracy_group, sync_dist=True)
                    self.log(f"{mode_str}_balacc_{emotion_name}", balanced_accuracy_group, sync_dist=True)
                    self.log(f"{mode_str}_AUROC_{emotion_name}", roc_auc_group, sync_dist=True)
                
            self.log(f"{mode_str}_acc", accuracy, sync_dist=True)
            self.log(f"{mode_str}_balacc", balanced_accuracy, sync_dist=True)
            self.log(f"{mode_str}_AUROC", roc_auc, sync_dist=True)
 
        # regression target is normalized
        elif self.hparams.downstream_task_type == 'regression':
            subj_avg_logits = subj_avg_logits.squeeze(-1)
            mse = F.mse_loss(subj_avg_logits, subj_targets)
            mae = F.l1_loss(subj_avg_logits, subj_targets)
            
            # reconstruct to original scale
            if self.hparams.label_scaling_method == 'standardization': # default
                adjusted_mse = F.mse_loss(subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0], subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0])
                adjusted_mae = F.l1_loss(subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0], subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                adjusted_mse = F.mse_loss(subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
                adjusted_mae = F.l1_loss(subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
            pearson = PearsonCorrCoef()
            r2_score = R2Score()
            
            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
                pearson_coef = pearson(subj_avg_logits.flatten(), subj_targets.flatten())
                r2 = r2_score(subj_avg_logits.flatten(), subj_targets.flatten()) if len(subj_avg_logits) >=2 else 0
            else:
                pearson_coef = pearson(subj_avg_logits, subj_targets)
                r2 = r2_score(subj_avg_logits, subj_targets) if len(subj_avg_logits) >=2 else 0
            
            if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
                
                # evaluate multiple targets separately
                t = self.hparams.img_size[3]
            
                subj_avg_logits = subj_avg_logits.view(-1, t ,self.hparams.num_targets) # (b, t*num_targets) -> (b, t, num_targets)
                subj_targets = subj_targets.view(-1, t ,self.hparams.num_targets) # (b, t*num_targets) -> (b, t, num_targets)
            
                for i in range(self.hparams.num_targets):
                    emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"
                    logits_group = subj_avg_logits[..., i]  # Shape: [batch_size, temporal_size]
                    target_group = subj_targets[..., i]

                    # Flatten for easier computation
                    logits_flat = logits_group.flatten()
                    target_flat = target_group.flatten()

                    # Overall metrics (original behavior)
                    mse_group = F.mse_loss(logits_flat, target_flat)
                    mae_group = F.l1_loss(logits_flat, target_flat)
                    pearson_coef_group = pearson(logits_flat, target_flat)
                    r2_group = r2_score(logits_flat, target_flat)

                    # Adjusted metrics (original scale)
                    if self.hparams.label_scaling_method == 'standardization': # default
                        adjusted_mse_group = F.mse_loss(logits_flat * self.scaler.scale_[0] + self.scaler.mean_[0], target_flat * self.scaler.scale_[0] + self.scaler.mean_[0])
                        adjusted_mae_group = F.l1_loss(logits_flat * self.scaler.scale_[0] + self.scaler.mean_[0], target_flat * self.scaler.scale_[0] + self.scaler.mean_[0])
                    elif self.hparams.label_scaling_method == 'minmax':
                        adjusted_mse_group = F.mse_loss(logits_flat * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], target_flat * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
                        adjusted_mae_group = F.l1_loss(logits_flat * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], target_flat * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])

                    # ========================================================================
                    # STRATIFIED METRICS: Non-zero and Zero samples (New!)
                    # ========================================================================

                    # Move to CPU/numpy for easier masking
                    target_np = target_flat.cpu().numpy()
                    logits_np = logits_flat.cpu().numpy()

                    # Create masks
                    mask_zero = (target_np == 0)
                    mask_nonzero = ~mask_zero

                    n_total = len(target_np)
                    n_zero = mask_zero.sum()
                    n_nonzero = mask_nonzero.sum()

                    # Log sample counts
                    self.log(f"{mode_str}_n_total_{emotion_name}", float(n_total), sync_dist=True)
                    self.log(f"{mode_str}_n_zero_{emotion_name}", float(n_zero), sync_dist=True)
                    self.log(f"{mode_str}_n_nonzero_{emotion_name}", float(n_nonzero), sync_dist=True)
                    self.log(f"{mode_str}_pct_zero_{emotion_name}", float(n_zero / n_total * 100) if n_total > 0 else 0.0, sync_dist=True)

                    # Non-zero metrics (핵심!)
                    if n_nonzero > 1:
                        target_nonzero = target_flat[torch.from_numpy(mask_nonzero)]
                        logits_nonzero = logits_flat[torch.from_numpy(mask_nonzero)]

                        # Non-zero MAE, MSE, RMSE
                        nonzero_mae = F.l1_loss(logits_nonzero, target_nonzero)
                        nonzero_mse = F.mse_loss(logits_nonzero, target_nonzero)
                        nonzero_rmse = torch.sqrt(nonzero_mse)

                        # Non-zero Pearson correlation
                        nonzero_pearson = pearson(logits_nonzero, target_nonzero)

                        # Adjusted non-zero metrics (original scale)
                        if self.hparams.label_scaling_method == 'standardization':
                            logits_nonzero_adj = logits_nonzero * self.scaler.scale_[0] + self.scaler.mean_[0]
                            target_nonzero_adj = target_nonzero * self.scaler.scale_[0] + self.scaler.mean_[0]
                        elif self.hparams.label_scaling_method == 'minmax':
                            logits_nonzero_adj = logits_nonzero * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]
                            target_nonzero_adj = target_nonzero * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]

                        nonzero_adjusted_mae = F.l1_loss(logits_nonzero_adj, target_nonzero_adj)
                        nonzero_adjusted_rmse = torch.sqrt(F.mse_loss(logits_nonzero_adj, target_nonzero_adj))

                        # Log non-zero metrics
                        self.log(f"{mode_str}_nonzero_mae_{emotion_name}", nonzero_mae, sync_dist=True)
                        self.log(f"{mode_str}_nonzero_mse_{emotion_name}", nonzero_mse, sync_dist=True)
                        self.log(f"{mode_str}_nonzero_rmse_{emotion_name}", nonzero_rmse, sync_dist=True)
                        self.log(f"{mode_str}_nonzero_pearson_{emotion_name}", nonzero_pearson, sync_dist=True)
                        self.log(f"{mode_str}_nonzero_adjusted_mae_{emotion_name}", nonzero_adjusted_mae, sync_dist=True)
                        self.log(f"{mode_str}_nonzero_adjusted_rmse_{emotion_name}", nonzero_adjusted_rmse, sync_dist=True)

                    # Zero metrics
                    if n_zero > 0:
                        logits_zero = logits_flat[torch.from_numpy(mask_zero)]

                        # Zero MAE (how close to 0 are predictions on zero targets)
                        zero_mae = torch.abs(logits_zero).mean()
                        zero_mean_pred = logits_zero.mean()
                        zero_std_pred = logits_zero.std()

                        # Adjusted zero metrics (original scale)
                        if self.hparams.label_scaling_method == 'standardization':
                            logits_zero_adj = logits_zero * self.scaler.scale_[0] + self.scaler.mean_[0]
                        elif self.hparams.label_scaling_method == 'minmax':
                            logits_zero_adj = logits_zero * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]

                        zero_adjusted_mae = torch.abs(logits_zero_adj).mean()

                        # Log zero metrics
                        self.log(f"{mode_str}_zero_mae_{emotion_name}", zero_mae, sync_dist=True)
                        self.log(f"{mode_str}_zero_mean_pred_{emotion_name}", zero_mean_pred, sync_dist=True)
                        self.log(f"{mode_str}_zero_std_pred_{emotion_name}", zero_std_pred, sync_dist=True)
                        self.log(f"{mode_str}_zero_adjusted_mae_{emotion_name}", zero_adjusted_mae, sync_dist=True)

                    # ========================================================================
                    # Log original overall metrics (for backward compatibility)
                    # ========================================================================
                    self.log(f"{mode_str}_corrcoef_{emotion_name}", pearson_coef_group, sync_dist=True)
                    self.log(f"{mode_str}_r2_score_{emotion_name}", r2_group, sync_dist=True)
                    self.log(f"{mode_str}_mse_{emotion_name}", mse_group, sync_dist=True)
                    self.log(f"{mode_str}_mae_{emotion_name}", mae_group, sync_dist=True)
                    self.log(f"{mode_str}_adjusted_mse_{emotion_name}", adjusted_mse_group, sync_dist=True)
                    self.log(f"{mode_str}_adjusted_mae_{emotion_name}", adjusted_mae_group, sync_dist=True)
            
            self.log(f"{mode_str}_corrcoef", pearson_coef, sync_dist=True)
            self.log(f"{mode_str}_r2_score", r2, sync_dist=True)
            self.log(f"{mode_str}_mse", mse, sync_dist=True)
            self.log(f"{mode_str}_mae", mae, sync_dist=True)
            self.log(f"{mode_str}_adjusted_mse", adjusted_mse, sync_dist=True) 
            self.log(f"{mode_str}_adjusted_mae", adjusted_mae, sync_dist=True)

    def _calculate_optimal_thresholds(self, subj_array, total_out):
        """Calculate optimal thresholds using Youden Index on validation data"""
        if self.hparams.downstream_task_type != 'classification':
            return

        if self.hparams.decoder not in ['series_decoder', 'lstm_series_regression_head']:
            return

        if self.hparams.num_classes != 2:
            print("[INFO] Youden threshold only supported for binary classification")
            return

        subjects = np.unique(subj_array)

        subj_avg_logits = []
        subj_targets = []
        for subj in subjects:
            subj_logits = [total_out[i][0] for i in range(len(subj_array)) if subj_array[i] == subj]
            subj_avg_logits.append(subj_logits)
            subj_targets.append([total_out[i][1] for i in range(len(subj_array)) if subj_array[i] == subj][0])

        subj_avg_logits = [i[0] for i in subj_avg_logits]
        subj_avg_logits = torch.stack(subj_avg_logits)
        subj_targets = torch.stack(subj_targets)

        print("\n" + "="*80)
        print("CALCULATING OPTIMAL THRESHOLDS (Youden Index)")
        print("="*80)

        t = self.hparams.img_size[3]

        # First flatten: (b, t*ta, c) -> (b*t*ta, c)
        subj_avg_logits = rearrange(subj_avg_logits, 'b tta c -> (b tta) c')
        subj_targets = subj_targets.flatten()

        # Then reshape: (b*t*ta, c) -> (b, t, ta, c)
        subj_avg_logits = rearrange(subj_avg_logits, '(b t ta) c -> b t ta c',
                                   t=t, ta=self.hparams.num_targets, c=self.hparams.num_classes)
        subj_targets = rearrange(subj_targets, '(b t ta) -> b t ta',
                                t=t, ta=self.hparams.num_targets)

        optimal_thresholds = {}

        for i in range(self.hparams.num_targets):
            logits_emotion = subj_avg_logits[:, :, i, :]  # (batch, time, num_classes)
            targets_emotion = subj_targets[:, :, i]  # (batch, time)

            # Get probabilities for class 1
            probs = F.softmax(logits_emotion.to(dtype=torch.float32), dim=-1)
            probs_pos = probs[..., 1]

            # Flatten
            probs_flat = probs_pos.flatten().cpu().numpy()
            targets_flat = targets_emotion.flatten().cpu().numpy()

            # Remove NaN
            valid_mask = ~np.isnan(targets_flat) & ~np.isnan(probs_flat)
            probs_flat = probs_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]

            if len(targets_flat) == 0:
                print(f"  Emotion {i}: No valid samples - using default threshold 0.5")
                optimal_thresholds[i] = 0.5
                continue

            # Check if both classes present
            unique_classes = np.unique(targets_flat)
            if len(unique_classes) < 2:
                print(f"  Emotion {i}: Only one class ({unique_classes}) - using default threshold 0.5")
                optimal_thresholds[i] = 0.5
                continue

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(targets_flat, probs_flat)

            # Youden index = TPR - FPR
            j_scores = tpr - fpr
            optimal_idx = j_scores.argmax()
            opt_thr = float(thresholds[optimal_idx])

            optimal_thresholds[i] = opt_thr
            print(f"  Emotion {i}: Optimal threshold = {opt_thr:.4f} (J={j_scores[optimal_idx]:.4f})")

        self.optimal_thresholds = optimal_thresholds
        print("="*80 + "\n")

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

        # Debug: print decoder type and target shape
        if batch_idx == 0:
            print(f"[DEBUG] decoder type: {self.hparams.decoder}")
            print(f"[DEBUG] target shape: {target.shape if hasattr(target, 'shape') else type(target)}")
            print(f"[DEBUG] logits shape: {logits.shape if hasattr(logits, 'shape') else type(logits)}")

        # Handle multi-target regression (LSTM series regression head, series decoder)
        # Check if target is multi-dimensional per sample
        if self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
            output = [(logit.cpu().detach(), targets.cpu()) for logit, targets in zip(logits, target)]
        elif self.hparams.decoder == 'lstm_regression_head':
            # LSTM regression head outputs single value per sequence
            output = [(logit.cpu().detach(), targets.cpu()) for logit, targets in zip(logits, target)]
        else:
            # Single value target - check if it's actually a scalar
            output = []
            for logit, targets in zip(logits, target):
                if targets.numel() == 1:
                    output.append((logit.cpu().detach(), targets.cpu().item()))
                else:
                    # Multi-element target, keep as tensor
                    output.append((logit.cpu().detach(), targets.cpu()))
        return (subj, output)

    def validation_epoch_end(self, outputs):
        """
        Aggregates and processes validation and test outputs at the end of an epoch.
        Evaluates metrics for both datasets and optionally saves model predictions for future analysis.
        """
        # Fix: valid_only should properly handle single dataloader output
        if self.valid_only:
            outputs_valid = outputs
            outputs_test = []
        else:
            outputs_valid = outputs[0]
            outputs_test = outputs[1]

        subj_valid = []
        subj_test = []
        out_valid_list = []
        out_test_list = []

        for subj, out in outputs_valid:
            subj_valid += subj
            out_valid_list.append(out)

        if not self.valid_only:
            for subj, out in outputs_test:
                subj_test += subj
                out_test_list.append(out)

        subj_valid = np.array(subj_valid)
        total_out_valid = [item for sublist in out_valid_list for item in sublist]

        if not self.valid_only:
            subj_test = np.array(subj_test)
            total_out_test = [item for sublist in out_test_list for item in sublist]

        # save model predictions if it is needed for future analysis
        # self._save_predictions(subj_valid,total_out_valid,mode="valid")
        # if not self.valid_only:
        #     self._save_predictions(subj_test,total_out_test, mode="test")

        # evaluate
        self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")

        # ========================================================================
        # Log summary metrics (average across emotions) for regression tasks
        # ========================================================================
        if self.hparams.downstream_task_type == 'regression' and self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
            # Collect non-zero metrics from trainer's callback_metrics
            nonzero_maes = []
            nonzero_pearsons = []
            nonzero_rmses = []

            for i in range(self.hparams.num_targets):
                emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"

                # Try to get metrics from logged values (they might be in callback_metrics)
                nonzero_mae_key = f"valid_nonzero_mae_{emotion_name}"
                nonzero_pearson_key = f"valid_nonzero_pearson_{emotion_name}"
                nonzero_rmse_key = f"valid_nonzero_rmse_{emotion_name}"

                # Use trainer's callback_metrics if available
                if hasattr(self.trainer, 'callback_metrics'):
                    if nonzero_mae_key in self.trainer.callback_metrics:
                        nonzero_maes.append(self.trainer.callback_metrics[nonzero_mae_key].item())
                    if nonzero_pearson_key in self.trainer.callback_metrics:
                        nonzero_pearsons.append(self.trainer.callback_metrics[nonzero_pearson_key].item())
                    if nonzero_rmse_key in self.trainer.callback_metrics:
                        nonzero_rmses.append(self.trainer.callback_metrics[nonzero_rmse_key].item())

            # Log average metrics
            if len(nonzero_maes) > 0:
                avg_nonzero_mae = np.mean(nonzero_maes)
                std_nonzero_mae = np.std(nonzero_maes)
                self.log("valid_avg_nonzero_mae", avg_nonzero_mae, sync_dist=True)
                self.log("valid_std_nonzero_mae", std_nonzero_mae, sync_dist=True)

            if len(nonzero_pearsons) > 0:
                avg_nonzero_pearson = np.mean(nonzero_pearsons)
                std_nonzero_pearson = np.std(nonzero_pearsons)
                self.log("valid_avg_nonzero_pearson", avg_nonzero_pearson, sync_dist=True)
                self.log("valid_std_nonzero_pearson", std_nonzero_pearson, sync_dist=True)

            if len(nonzero_rmses) > 0:
                avg_nonzero_rmse = np.mean(nonzero_rmses)
                std_nonzero_rmse = np.std(nonzero_rmses)
                self.log("valid_avg_nonzero_rmse", avg_nonzero_rmse, sync_dist=True)
                self.log("valid_std_nonzero_rmse", std_nonzero_rmse, sync_dist=True)

            # Print summary if on global rank 0
            if self.trainer.is_global_zero and len(nonzero_maes) > 0:
                print(f"\n{'='*80}")
                print("STRATIFIED METRICS SUMMARY (Validation)")
                print(f"{'='*80}")
                print(f"Avg Non-zero MAE:     {avg_nonzero_mae:.4f} ± {std_nonzero_mae:.4f}")
                if len(nonzero_pearsons) > 0:
                    print(f"Avg Non-zero Pearson: {avg_nonzero_pearson:.4f} ± {std_nonzero_pearson:.4f}")
                if len(nonzero_rmses) > 0:
                    print(f"Avg Non-zero RMSE:    {avg_nonzero_rmse:.4f} ± {std_nonzero_rmse:.4f}")
                print(f"{'='*80}\n")

        # Log learnable loss weights/uncertainties
        if self.learnable_loss is not None and self.trainer.is_global_zero:
            if isinstance(self.learnable_loss, PerEmotionLearnableWeightedMSE):
                weights = self.learnable_loss.get_weights()
                print(f"\n{'='*80}")
                print("Learned Weights (Per-Emotion Weighted MSE)")
                print(f"{'='*80}")
                print(f"Zero weights:     {weights['zero_weights']}")
                print(f"Non-zero weights: {weights['nonzero_weights']}")
                print(f"{'='*80}\n")

                # Log to wandb with emotion names
                for i in range(self.hparams.num_targets):
                    emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"
                    self.log(f"weight_zero_{emotion_name}", weights['zero_weights'][i], sync_dist=True)
                    self.log(f"weight_nonzero_{emotion_name}", weights['nonzero_weights'][i], sync_dist=True)

            elif isinstance(self.learnable_loss, UncertaintyWeightedMSE):
                uncertainties = self.learnable_loss.get_uncertainties()
                print(f"\n{'='*80}")
                print("Learned Uncertainties (Uncertainty-based Weighted MSE)")
                print(f"{'='*80}")
                print(f"Log variances: {uncertainties['log_vars']}")
                print(f"Sigmas (σ):    {uncertainties['sigmas']}")
                print(f"{'='*80}\n")

                # Log to wandb with emotion names
                for i in range(self.hparams.num_targets):
                    emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"
                    self.log(f"uncertainty_logvar_{emotion_name}", uncertainties['log_vars'][i], sync_dist=True)
                    self.log(f"uncertainty_sigma_{emotion_name}", uncertainties['sigmas'][i], sync_dist=True)

        # Calculate optimal thresholds from validation set
        self._calculate_optimal_thresholds(subj_valid, total_out_valid)

        if not self.valid_only:
            self._evaluate_metrics(subj_test, total_out_test, mode="test")

            # ========================================================================
            # Log summary metrics for test set (average across emotions)
            # ========================================================================
            if self.hparams.downstream_task_type == 'regression' and self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
                # Collect non-zero metrics from trainer's callback_metrics
                nonzero_maes = []
                nonzero_pearsons = []
                nonzero_rmses = []

                for i in range(self.hparams.num_targets):
                    emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"

                    # Try to get metrics from logged values
                    nonzero_mae_key = f"test_nonzero_mae_{emotion_name}"
                    nonzero_pearson_key = f"test_nonzero_pearson_{emotion_name}"
                    nonzero_rmse_key = f"test_nonzero_rmse_{emotion_name}"

                    # Use trainer's callback_metrics if available
                    if hasattr(self.trainer, 'callback_metrics'):
                        if nonzero_mae_key in self.trainer.callback_metrics:
                            nonzero_maes.append(self.trainer.callback_metrics[nonzero_mae_key].item())
                        if nonzero_pearson_key in self.trainer.callback_metrics:
                            nonzero_pearsons.append(self.trainer.callback_metrics[nonzero_pearson_key].item())
                        if nonzero_rmse_key in self.trainer.callback_metrics:
                            nonzero_rmses.append(self.trainer.callback_metrics[nonzero_rmse_key].item())

                # Log average metrics
                if len(nonzero_maes) > 0:
                    avg_nonzero_mae = np.mean(nonzero_maes)
                    std_nonzero_mae = np.std(nonzero_maes)
                    self.log("test_avg_nonzero_mae", avg_nonzero_mae, sync_dist=True)
                    self.log("test_std_nonzero_mae", std_nonzero_mae, sync_dist=True)

                if len(nonzero_pearsons) > 0:
                    avg_nonzero_pearson = np.mean(nonzero_pearsons)
                    std_nonzero_pearson = np.std(nonzero_pearsons)
                    self.log("test_avg_nonzero_pearson", avg_nonzero_pearson, sync_dist=True)
                    self.log("test_std_nonzero_pearson", std_nonzero_pearson, sync_dist=True)

                if len(nonzero_rmses) > 0:
                    avg_nonzero_rmse = np.mean(nonzero_rmses)
                    std_nonzero_rmse = np.std(nonzero_rmses)
                    self.log("test_avg_nonzero_rmse", avg_nonzero_rmse, sync_dist=True)
                    self.log("test_std_nonzero_rmse", std_nonzero_rmse, sync_dist=True)

                # Print summary if on global rank 0
                if self.trainer.is_global_zero and len(nonzero_maes) > 0:
                    print(f"\n{'='*80}")
                    print("STRATIFIED METRICS SUMMARY (Test)")
                    print(f"{'='*80}")
                    print(f"Avg Non-zero MAE:     {avg_nonzero_mae:.4f} ± {std_nonzero_mae:.4f}")
                    if len(nonzero_pearsons) > 0:
                        print(f"Avg Non-zero Pearson: {avg_nonzero_pearson:.4f} ± {std_nonzero_pearson:.4f}")
                    if len(nonzero_rmses) > 0:
                        print(f"Avg Non-zero RMSE:    {avg_nonzero_rmse:.4f} ± {std_nonzero_rmse:.4f}")
                    print(f"{'='*80}\n")
            
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
        if self.hparams.decoder in ['series_decoder', 'lstm_regression_head', 'lstm_series_regression_head']: # (batch, T, E) -> (batch, T*E)
            output = [(logit.cpu().detach(), targets.cpu().detach()) for logit, targets in zip(logits, target)] # target is not single value, item() cannot be invoked
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

        # ========================================================================
        # Log summary metrics for test set (average across emotions)
        # ========================================================================
        if self.hparams.downstream_task_type == 'regression' and self.hparams.decoder in ['series_decoder', 'lstm_series_regression_head']:
            # Collect non-zero metrics from trainer's callback_metrics
            nonzero_maes = []
            nonzero_pearsons = []
            nonzero_rmses = []

            for i in range(self.hparams.num_targets):
                emotion_name = self.EMOTION_NAMES[i] if i < len(self.EMOTION_NAMES) else f"emotion_{i}"

                # Try to get metrics from logged values
                nonzero_mae_key = f"test_nonzero_mae_{emotion_name}"
                nonzero_pearson_key = f"test_nonzero_pearson_{emotion_name}"
                nonzero_rmse_key = f"test_nonzero_rmse_{emotion_name}"

                # Use trainer's callback_metrics if available
                if hasattr(self.trainer, 'callback_metrics'):
                    if nonzero_mae_key in self.trainer.callback_metrics:
                        nonzero_maes.append(self.trainer.callback_metrics[nonzero_mae_key].item())
                    if nonzero_pearson_key in self.trainer.callback_metrics:
                        nonzero_pearsons.append(self.trainer.callback_metrics[nonzero_pearson_key].item())
                    if nonzero_rmse_key in self.trainer.callback_metrics:
                        nonzero_rmses.append(self.trainer.callback_metrics[nonzero_rmse_key].item())

            # Log average metrics
            if len(nonzero_maes) > 0:
                avg_nonzero_mae = np.mean(nonzero_maes)
                std_nonzero_mae = np.std(nonzero_maes)
                self.log("test_avg_nonzero_mae", avg_nonzero_mae, sync_dist=True)
                self.log("test_std_nonzero_mae", std_nonzero_mae, sync_dist=True)

            if len(nonzero_pearsons) > 0:
                avg_nonzero_pearson = np.mean(nonzero_pearsons)
                std_nonzero_pearson = np.std(nonzero_pearsons)
                self.log("test_avg_nonzero_pearson", avg_nonzero_pearson, sync_dist=True)
                self.log("test_std_nonzero_pearson", std_nonzero_pearson, sync_dist=True)

            if len(nonzero_rmses) > 0:
                avg_nonzero_rmse = np.mean(nonzero_rmses)
                std_nonzero_rmse = np.std(nonzero_rmses)
                self.log("test_avg_nonzero_rmse", avg_nonzero_rmse, sync_dist=True)
                self.log("test_std_nonzero_rmse", std_nonzero_rmse, sync_dist=True)

            # Print summary if on global rank 0
            if self.trainer.is_global_zero and len(nonzero_maes) > 0:
                print(f"\n{'='*80}")
                print("STRATIFIED METRICS SUMMARY (Test - Final)")
                print(f"{'='*80}")
                print(f"Avg Non-zero MAE:     {avg_nonzero_mae:.4f} ± {std_nonzero_mae:.4f}")
                if len(nonzero_pearsons) > 0:
                    print(f"Avg Non-zero Pearson: {avg_nonzero_pearson:.4f} ± {std_nonzero_pearson:.4f}")
                if len(nonzero_rmses) > 0:
                    print(f"Avg Non-zero RMSE:    {avg_nonzero_rmse:.4f} ± {std_nonzero_rmse:.4f}")
                print(f"{'='*80}\n")
    
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
        group.add_argument("--decoder", type=str, default="single_target_decoder", help="Which decoder to use: (i) single_target_decoder - predict a single value via regression or classification | (ii) series_decoder: predict a series of values (one per timeframe) via regression")
        group.add_argument("--num_targets", type=int, default=7, help="Number of targets to predict in series_decoder")
        # parser.add_argument("--valid_only", action='store_true', help="disable running _evaluate_metrics(mode='test') at validation stage") # kimbo change

        # learnable loss related (for regression)
        group.add_argument("--regression_loss_type", type=str, default="mse",
                          choices=["mse", "per_emotion_weighted", "uncertainty_weighted", "focal_mse", "weighted_focal_mse", "normalized_focal_mse", "tweedie"],
                          help="Loss function type for regression: 'mse' (standard), 'per_emotion_weighted' (learnable weights for zero/non-zero), 'uncertainty_weighted' (uncertainty-based weighting), 'focal_mse' (focal loss for hard samples), 'weighted_focal_mse' (focal + zero/non-zero weighting), 'normalized_focal_mse' (scale-robust focal loss), 'tweedie' (Tweedie loss for zero-inflated data)")
        group.add_argument("--init_zero_weight", type=float, default=0.1,
                          help="Initial weight for zero values in per_emotion_weighted loss (default: 0.1)")
        group.add_argument("--init_nonzero_weight", type=float, default=5.0,
                          help="Initial weight for non-zero values in per_emotion_weighted loss (default: 5.0)")
        group.add_argument("--init_log_var", type=float, default=0.0,
                          help="Initial log variance for uncertainty_weighted loss (default: 0.0, i.e., variance=1.0)")
        group.add_argument("--focal_gamma", type=float, default=2.0,
                          help="Gamma parameter for focal losses (default: 2.0). Higher gamma = more focus on hard samples")
        group.add_argument("--focal_eps", type=float, default=1e-6,
                          help="Epsilon for numerical stability in normalized_focal_mse loss (default: 1e-6)")
        group.add_argument("--zero_weight", type=float, default=1.0,
                          help="Weight for zero targets in weighted_focal_mse loss (default: 1.0)")
        group.add_argument("--nonzero_weight", type=float, default=5.0,
                          help="Weight for non-zero targets in weighted_focal_mse loss (default: 5.0)")
        group.add_argument("--tweedie_p", type=float, default=1.5,
                          help="Power parameter p for Tweedie loss (1 < p < 2, default: 1.5). Controls zero-inflation modeling")

        return parser