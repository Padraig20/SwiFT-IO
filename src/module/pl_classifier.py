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

class LitClassifier(pl.LightningModule):
    
    def __init__(self,data_module, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs) # save hyperparameters except data_module (data_module cannot be pickled as a checkpoint)
       
        # you should define target_values at the Dataset classes
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
        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)
        
        self.output_head = load_model(self.hparams.decoder, self.hparams)

        self.metric = Metrics()

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
    
    def _compute_logits(self, batch, augment_during_training=None):
        """
        Processes a batch of data to compute logits for either classification or regression tasks. 
        Applies optional augmentation during training and handles label scaling for regression tasks.
        """
        fmri, subj, target_value, tr, sex = batch.values()
       
        if augment_during_training:
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
            
            logits = self.output_head(feature) # (b,1)
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
                logits = rearrange(logits, 'b tta c -> (b tta) c')
                target = target.flatten() # (b,c) -> (b*c)
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

    def _evaluate_metrics(self, subj_array, total_out, mode):
        """
        Evaluates classification or regression metrics for aggregated subject-level predictions. 
        Logs accuracy, balanced accuracy, and AUROC for classification tasks, and MSE, MAE, and correlation coefficients for regression tasks, including metrics on the original scale.
        """
        subjects = np.unique(subj_array)
        
        subj_avg_logits = []
        subj_targets = []
        for subj in subjects:
            subj_logits = [total_out[i][0] for i in range(len(subj_array)) if subj_array[i] == subj]
            if self.hparams.decoder == 'series_decoder': # do not calculate the average logits
                subj_avg_logits.append(subj_logits)
            else:
                subj_avg_logits.append(torch.mean(torch.stack(subj_logits), dim=0))
            subj_targets.append([total_out[i][1] for i in range(len(subj_array)) if subj_array[i] == subj][0])
    
        if self.hparams.decoder == 'series_decoder':
            subj_avg_logits = [i[0] for i in subj_avg_logits] # unpack single values from the list
            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.stack(subj_targets)
        else:
            subj_avg_logits = torch.stack(subj_avg_logits)
            subj_targets = torch.tensor(subj_targets)
    
        if self.hparams.downstream_task_type == 'classification':
            
            if self.hparams.decoder == 'series_decoder':
                subj_avg_logits = rearrange(subj_avg_logits, 'b tta c -> (b tta) c')
                subj_targets = subj_targets.flatten()
                
            num_classes = subj_avg_logits.shape[1]
            
            probabilities = F.softmax(subj_avg_logits.to(dtype=torch.float32), dim=1) # (b,num_classes), require 32 bit precision
            predictions = probabilities.argmax(dim=1) # (b)
            
            predictions_np = predictions.cpu().numpy()
            targets_np = subj_targets.cpu().numpy()

            accuracy = accuracy_score(targets_np, predictions_np)
            balanced_accuracy = balanced_accuracy_score(targets_np, predictions_np)

            if num_classes == 2:
                roc_auc = roc_auc_score(targets_np, predictions_np)
            else: 
                targets_one_hot = label_binarize(targets_np, classes=np.arange(num_classes))
                roc_auc = roc_auc_score(targets_one_hot, probabilities.cpu().detach().numpy(), multi_class='ovr')

            if self.hparams.decoder == 'series_decoder':
                
                # evaluate multiple targets separately
                t = self.hparams.img_size[3]

                subj_avg_logits = rearrange(subj_avg_logits, '(b t ta) c -> b t ta c', t=t, ta=self.hparams.num_targets, c=self.hparams.num_classes)
                subj_targets = rearrange(subj_targets, '(b t ta) -> b t ta', t=t, ta=self.hparams.num_targets)
            
                for i in range(self.hparams.num_targets):
                    logits_group = subj_avg_logits[:,:,i]  # Shape: [batch_size, temporal_size, num_classes]
                    target_group = subj_targets[..., i]
                    
                    probabilities = F.softmax(logits_group.to(dtype=torch.float32), dim=-1) # (b, temporal_size, num_classes), require 32 bit precision
                    predictions = probabilities.argmax(dim=-1) # (b, temporal_size)
                    
                    predictions_np = predictions.flatten().cpu().numpy()
                    targets_np = target_group.flatten().cpu().numpy()
                    
                    accuracy_group = accuracy_score(targets_np, predictions_np)
                    balanced_accuracy_group = balanced_accuracy_score(targets_np, predictions_np)
                    
                    if num_classes == 2:
                        roc_auc_group = roc_auc_score(targets_np, predictions_np)
                    else: 
                        targets_one_hot = label_binarize(targets_np, classes=np.arange(num_classes))
                        roc_auc_group = roc_auc_score(targets_one_hot, rearrange(probabilities, 'b t c -> (b t) c').cpu().detach().numpy(), multi_class='ovr')
                    
                    self.log(f"{mode}_acc_{i}", accuracy_group, sync_dist=True)
                    self.log(f"{mode}_balacc_{i}", balanced_accuracy_group, sync_dist=True)
                    self.log(f"{mode}_AUROC_{i}", roc_auc_group, sync_dist=True)
                
            self.log(f"{mode}_acc", accuracy, sync_dist=True)
            self.log(f"{mode}_balacc", balanced_accuracy, sync_dist=True)
            self.log(f"{mode}_AUROC", roc_auc, sync_dist=True)

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
            
            if self.hparams.decoder == 'series_decoder':
                pearson_coef = pearson(subj_avg_logits.flatten(), subj_targets.flatten())
                r2 = r2_score(subj_avg_logits.flatten(), subj_targets.flatten()) if len(subj_avg_logits) >=2 else 0
            else:
                pearson_coef = pearson(subj_avg_logits, subj_targets)
                r2 = r2_score(subj_avg_logits, subj_targets) if len(subj_avg_logits) >=2 else 0
            
            if self.hparams.decoder == 'series_decoder':
                
                # evaluate multiple targets separately
                t = self.hparams.img_size[3]
            
                subj_avg_logits = subj_avg_logits.view(-1, t ,self.hparams.num_targets) # (b, t*num_targets) -> (b, t, num_targets)
                subj_targets = subj_targets.view(-1, t ,self.hparams.num_targets) # (b, t*num_targets) -> (b, t, num_targets)
            
                for i in range(self.hparams.num_targets):
                    logits_group = subj_avg_logits[..., i]  # Shape: [batch_size, temporal_size]
                    target_group = subj_targets[..., i]
                
                    mse_group = F.mse_loss(logits_group, target_group)  # target is float
                    mae_group = F.l1_loss(logits_group, target_group)
                
                    pearson_coef_group = pearson(logits_group.flatten(), target_group.flatten())
                    r2_group = r2_score(logits_group.flatten(), target_group.flatten()) 

                    if self.hparams.label_scaling_method == 'standardization': # default
                        adjusted_mse_group = F.mse_loss(logits_group * self.scaler.scale_[0] + self.scaler.mean_[0], target_group * self.scaler.scale_[0] + self.scaler.mean_[0])
                        adjusted_mae_group = F.l1_loss(logits_group * self.scaler.scale_[0] + self.scaler.mean_[0], target_group * self.scaler.scale_[0] + self.scaler.mean_[0])
                    elif self.hparams.label_scaling_method == 'minmax':
                        adjusted_mse_group = F.mse_loss(logits_group * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], target_group * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
                        adjusted_mae_group = F.l1_loss(logits_group * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], target_group * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])

                    self.log(f"{mode}_corrcoef_{i}", pearson_coef_group, sync_dist=True)
                    self.log(f"{mode}_r2_score_{i}", r2_group, sync_dist=True)
                    self.log(f"{mode}_mse_{i}", mse_group, sync_dist=True)
                    self.log(f"{mode}_mae_{i}", mae_group, sync_dist=True)
                    self.log(f"{mode}_adjusted_mse_{i}", adjusted_mse_group, sync_dist=True)
                    self.log(f"{mode}_adjusted_mae_{i}", adjusted_mae_group, sync_dist=True)
            
            self.log(f"{mode}_corrcoef", pearson_coef, sync_dist=True)
            self.log(f"{mode}_r2_score", r2, sync_dist=True)
            self.log(f"{mode}_mse", mse, sync_dist=True)
            self.log(f"{mode}_mae", mae, sync_dist=True)
            self.log(f"{mode}_adjusted_mse", adjusted_mse, sync_dist=True) 
            self.log(f"{mode}_adjusted_mae", adjusted_mae, sync_dist=True)

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
        return (subj, output)

    def validation_epoch_end(self, outputs):
        """
        Aggregates and processes validation and test outputs at the end of an epoch. 
        Evaluates metrics for both datasets and optionally saves model predictions for future analysis.
        """
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
        total_out_test = [item for sublist in out_test_list for item in sublist]

        # save model predictions if it is needed for future analysis
        # self._save_predictions(subj_valid,total_out_valid,mode="valid")
        # self._save_predictions(subj_test,total_out_test, mode="test") 
                
        # evaluate 
        self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
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
        
        # decoder related
        group.add_argument("--num_classes", type=int, default=2, help="Number of distinct target classes")
        group.add_argument("--decoder", type=str, default="single_target_decoder", help="Which decoder to use: (i) single_target_decoder - predict a single value via regression or classification | (ii) series_decoder: predict a series of values (one per timeframe) via regression")
        group.add_argument("--num_targets", type=int, default=7, help="Number of targets to predict in series_decoder")
        
        return parser