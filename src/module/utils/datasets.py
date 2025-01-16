# 4D_fMRI_Transformer
import os
import glob
import torch
from torch.utils.data import Dataset

# import augmentations
import numpy as np
import torchio as tio
import random

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration),1)
        self.data = self._set_data(self.root, self.subject_dict)

        transforms_dict = {
            tio.RandomMotion(): 0.25,
            tio.RandomBlur(): 0.25,
            tio.RandomNoise(): 0.25,
            tio.RandomGamma(): 0.25,
        }
        self.transform = tio.Compose([
            tio.RandomAffine(),
            tio.OneOf(transforms_dict),
        ])
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        y = []
        if self.shuffle_time_sequence: # shuffle whole sequences
            load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0,num_frames)),sample_duration//self.stride_within_seq)]
        else:
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            
        if self.with_voxel_norm:
            load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
        for fname in load_fnames:
            img_path = os.path.join(subject_path, fname)
            y_i = torch.load(img_path).unsqueeze(0)
            y.append(y_i)
        y = torch.cat(y, dim=4)
        return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")
    
class Dummy(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=1000)
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        
        # classification
        #num_classes = 2
        #target = torch.randint(0,num_classes,(1,)).float()
        
        # regression
        #target = torch.rand(1)
        
        # series decoder regression
        #num_targets = 7
        #target = torch.randn(20, num_targets)
        
        # series decoder classification
        num_targets = 7
        num_classes = 2
        target = torch.randint(0,num_classes,(20,num_targets)).float()
        
        return {
                "fmri_sequence": y,
                "subject_name": subj,
                "target": target,
                "TR": 0,
                "sex": sex,
            } 
    
class HBN(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Number of sequences: {len(self.data)}")
        self.count_unique_subjects()  # 고유한 subject 개수 출력
    
    def count_unique_subjects(self):
        # self.data에서 모든 subject_name을 추출하고 고유한 값의 개수를 셈
        unique_subjects = set([tup[1] for tup in self.data])  # self.data에서 두 번째 요소가 subject_name
        print(f"Number of unique subjects: {len(unique_subjects)}")

    def _set_data(self, root, subject_dict):
        data = []
        
        img_root = os.path.join(root, 'img') 

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            num_frames = len(glob.glob(os.path.join(subject_path,'frame_*'))) # voxel mean & std
            #num_frames = 250 # TODO remove this line, just for testing
            session_duration = num_frames - self.sample_duration + 1
            
            for start_frame in range(0, session_duration, self.stride):
                if hasattr(target,'__len__'):
                    target = target[start_frame:min(start_frame+self.sample_duration,num_frames)]
                                    
                data_tuple = (i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target, sex)
                data.append(data_tuple)

        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3) 
        y = torch.nn.functional.pad(y, (7, 8, 0, 1, 7, 8), value=background_value) # adjust this padding level according to your data 
        y = y.permute(0,2,3,4,1) 

        if hasattr(self, 'time_as_channel') and self.time_as_channel:
            y = y.permute(0,4,1,2,3).squeeze()
        print(type(y), type(subject_name), type(target), type(start_frame), type(sex))
        return {
            "fmri_sequence": y,
            "subject_name": subject_name,
            "target": target,
            "TR": start_frame,
            "sex": sex,
            } 