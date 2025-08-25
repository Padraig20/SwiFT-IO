import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from .datasets import Dummy, HBN
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # self.setup() 

        #pl.seed_everything(seed=self.hparams.data_seed)

    def get_dataset(self):
        if self.hparams.dataset_name == "Dummy":
            return Dummy
        elif self.hparams.dataset_name == "HBN":
            return HBN
        else:
            raise NotImplementedError(f"Dataset {self.hparams.dataset_name} not implemented")

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        #subj_idx = np.array([str(x[0]) for x in subj_list])
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        # print(S)
        print('unique subjects:',len(S))  
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx
    
    def determine_stratified_split(self, subject_dict, seed, stratified_params, metadata_csv_path,
                                train_split_size=0.7, val_split_size=0.15):

        df = pd.read_csv(metadata_csv_path)
        df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(str)
        subject_ids = set(str(sid) for sid in subject_dict)
        df = df[df["SUBJECT_ID"].isin(subject_ids)].copy()

        if df.empty:
            raise ValueError("No matching SUBJECT_IDs found in metadata.")

        X = df["SUBJECT_ID"].values

        val_test_split = 1.0 - train_split_size
        test_size = (1.0 - train_split_size - val_split_size) / val_test_split

        if not stratified_params:
            train_ids, temp_ids = train_test_split(X, test_size=val_test_split, random_state=seed)
            val_ids, test_ids = train_test_split(temp_ids, test_size=test_size, random_state=seed)
            return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()

        Y = []
        for col in stratified_params:
            if np.issubdtype(df[col].dtype, np.number):
                binned = pd.qcut(df[col], q=4, labels=False, duplicates='drop')  # bin continuous (4 quartiles)
                Y.append(binned.values)
            else:
                encoded = pd.factorize(df[col])[0]  # encode categorical
                Y.append(encoded)

        Y = np.vstack(Y).T

        if Y.shape[1] == 1:
            # single-column stratification => sklearn
            stratify_labels = Y[:, 0]
            train_ids, temp_ids, _, temp_labels = train_test_split(
                X, stratify_labels, test_size=val_test_split, random_state=seed, stratify=stratify_labels
            )
            val_ids, test_ids = train_test_split(
                temp_ids, test_size=test_size, random_state=seed, stratify=temp_labels
            )
        else:
            # multi-label stratification => iterative-stratification
            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_test_split, random_state=seed)
            train_idx, temp_idx = next(msss.split(X, Y))
            X_temp, Y_temp = X[temp_idx], Y[temp_idx]

            msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            val_idx, test_idx = next(msss2.split(X_temp, Y_temp))

            train_ids = X[train_idx]
            val_ids = X_temp[val_idx]
            test_ids = X_temp[test_idx]

        return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()

    def prepare_data(self):
        # This function is only called at global rank==0
        return
    
    # filter subjects with metadata and pair subject names with their target values (+ sex)
    def make_subject_dict(self):
        # output: {'subj1':[target1,target2],'subj2':[target1,target2]...}
        img_root = os.path.join(self.hparams.image_path, 'img')
        final_dict = dict()
        
        if self.hparams.dataset_name == 'HBN':
            emotions = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
            contents = ['Closeup', 'Body', 'Face', 'NumberCharacters', 'SpokenWords', 'WrittenWords']
            features = ['Brightness', 'SaliencyFraction', 'Sharpness', 'Vibrance', 'Loudness', 'Motion', 'Tempo', 'LowLevelChange']

            if self.hparams.decoder == 'single_target_decoder':
                if self.hparams.downstream_task == 'sex': task_name = 'sex'
                elif self.hparams.downstream_task == 'age': task_name = 'age'
                else: raise ValueError('downstream task not supported')
                
                meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HBN_metadata_240501_CJB.csv"))
                if task_name == 'sex':
                    meta_task = meta_data[['SUBJECT_ID',task_name]].dropna()
                else:
                    meta_task = meta_data[['SUBJECT_ID',task_name,'sex']].dropna()

                for subject in os.listdir(img_root):
                    if subject in meta_task['SUBJECT_ID'].values:
                        target = meta_task[meta_task["SUBJECT_ID"]==subject][task_name].values[0]
                        sex = meta_task[meta_task["SUBJECT_ID"]==subject]["sex"].values[0]
                        final_dict[subject]=[sex,target]

            elif self.hparams.decoder == 'series_decoder':

                if self.hparams.downstream_task == 'emotions': task_name = emotions
                elif self.hparams.downstream_task == 'contents': task_name = contents
                elif self.hparams.downstream_task == 'features': task_name = features
                else: raise ValueError('downstream task not supported')

                
                if self.hparams.downstream_task_type == 'regression' and self.hparams.adjust_hrf == True: # kimbo change
                    task_name = [x + "_conv" for x in task_name]  # kimbo change
                elif self.hparams.downstream_task_type == 'regression' and self.hparams.adjust_hrf == False:  # kimbo change
                    task_name = [x for x in task_name]  # kimbo change
                elif self.hparams.downstream_task_type == 'classification' and self.hparams.adjust_hrf == True:  # kimbo change
                    task_name = [x + "_conv_binary" for x in task_name]  # kimbo change
                elif self.hparams.downstream_task_type == 'classification' and self.hparams.adjust_hrf == False:  # kimbo change
                    task_name = [x + "_binary" for x in task_name]  # kimbo change
                else:
                    raise ValueError('downstream task type not supported')
                
                if self.hparams.input_type == 'movieDM':
                    meta_data = pd.read_csv("/pscratch/sd/k/kimbo/SwiFT-IO-2/SwiFT-IO/metadata/DespicableMe_summary_codes_1.2Hz_intuitivenames_270819.csv") # TODO change later
                    
                elif self.hparams.input_type == 'movieTP':
                    meta_data = pd.read_csv("/pscratch/sd/k/kimbo/SwiFT-IO/metadata/ThePresent_summary_codes_1.2Hz_intuitivenames_260120.csv")
                
                meta_task = meta_data[task_name + ['frame']].dropna() 
                
                for subject in os.listdir(img_root):
                        sex = 1 # arbitrary value, not used
                        target = meta_task[task_name].values
                        target = target[np.argsort(meta_task['frame'].values)]
                        final_dict[subject] = (sex, target)
        
        return final_dict

    def setup(self, stage=None):
        # this function will be called at each devices
        Dataset = self.get_dataset()
        params = {
                "root": self.hparams.image_path,
                "sequence_length": self.hparams.sequence_length,
                "contrastive":self.hparams.use_contrastive,
                "contrastive_type":self.hparams.contrastive_type,
                "stride_between_seq": self.hparams.stride_between_seq,
                "stride_within_seq": self.hparams.stride_within_seq,
                "with_voxel_norm": self.hparams.with_voxel_norm,
                "downstream_task": self.hparams.downstream_task,
                "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
                "input_type": self.hparams.input_type,
                "label_scaling_method" : self.hparams.label_scaling_method,
                "decoder": self.hparams.decoder,
                "adjust_hrf": self.hparams.adjust_hrf,
                "dtype":'float16', 
                "input_offset": self.hparams.input_offset} # kimbo change
        
        subject_dict = self.make_subject_dict()
        
        metadata_csv_path = "/pscratch/sd/k/kimbo/SwiFT-IO/data_behavior/split_fixed_1.w.Dx.csv"
        # now split the data
        train_names, val_names, test_names = self.determine_stratified_split(subject_dict, self.hparams.dataset_split_seed, self.hparams.stratified_params,
                                                                             metadata_csv_path, self.hparams.train_split, self.hparams.val_split)
                
        if self.hparams.bad_subj_path:
            bad_subjects = open(self.hparams.bad_subj_path, "r").readlines()
            for bad_subj in bad_subjects:
                bad_subj = bad_subj.strip()
                if bad_subj in list(subject_dict.keys()):
                    print(f'removing bad subject: {bad_subj}')
                    del subject_dict[bad_subj]
        
        if self.hparams.limit_training_samples:
            train_names = np.random.choice(train_names, size=self.hparams.limit_training_samples, replace=False, p=None)

        def get_params(train):
                return {
                    "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                    "num_workers": self.hparams.num_workers,
                    "drop_last": True,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "shuffle": train,
                }
        
        print(f"현재 실행중인 stage: {stage}")
        
        if stage in (None, "fit"):  # train + val
            train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
            val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
            test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}

            self.train_dataset = Dataset(**params, subject_dict=train_dict, use_augmentations=False, train=True)
            self.val_dataset = Dataset(**params, subject_dict=val_dict, use_augmentations=False, train=False)
            self.test_dataset = Dataset(**params, subject_dict=test_dict, use_augmentations=False, train=False)
            
            self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
            self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
            self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))
            print("number of train_subj:", len(train_dict))
            print("number of val_subj:", len(val_dict))
            print("length of train_idx:", len(self.train_dataset.data))  
            print("length of val_idx:", len(self.val_dataset.data))
            print("number of test_subj:", len(test_dict))
            print("length of test_idx:", len(self.test_dataset.data))

        if stage in ("test", "predict"): # kimbo change
            test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}
            self.test_dataset = Dataset(**params, subject_dict=test_dict, use_augmentations=False, train=False)
            self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))
            print("number of test_subj:", len(test_dict))
            print("length of test_idx:", len(self.test_dataset.data))        

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        # return self.val_loader
        # currently returns validation and test set to track them during training
        return [self.val_loader, self.test_loader]

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        
        # dataset split parameters
        group.add_argument("--dataset_split_seed", type=int, default=777)
        group.add_argument("--stratified_params", nargs="+", default=None, type=str, help="stratified parameters for dataset split")
        
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"], help="label normalization strategy for a regression task (mean and std are automatically calculated using train set)")
        group.add_argument("--image_path", default=None, help="path to image datasets preprocessed for SwiFT")
        group.add_argument("--bad_subj_path", default=None, help="path to txt file that contains subjects with bad fMRI quality")
        group.add_argument("--input_type", default="movieDM",choices=['rest','task', 'movieDM', 'movieTP'],help='refer to datasets.py')
        group.add_argument("--train_split", default=0.7, type=float)
        group.add_argument("--val_split", default=0.15, type=float)
        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--eval_batch_size", type=int, default=16)
        group.add_argument("--img_size", nargs="+", default=[96, 96, 96, 20], type=int, help="image size (adjust the fourth dimension according to your --sequence_length argument)")
        group.add_argument("--sequence_length", type=int, default=20)
        group.add_argument("--stride_between_seq", type=float, default=1.0, help="Fractional stride (0.5 = 50% overlap between sequences). Will be multiplied by sample_duration and rounded internally.")
        group.add_argument("--stride_within_seq", type=int, default=1, help="skip some fMRI volumes within fMRI sub-sequences")
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--with_voxel_norm", action='store_true')
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--limit_training_samples", type=int, default=None, help="use if you want to limit training samples")
        group.add_argument("--adjust_hrf", action='store_true', help="use for HRF effect adjustmenet. shifts start fMRI timeframe from 0TR to 7TR")
        group.add_argument("--input_offset", default=0, type = int, help="Shifts the starting point of the fMRI input sequence") # kimbo change

        return parser