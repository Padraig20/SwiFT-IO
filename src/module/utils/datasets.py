# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset

# import augmentations
import numpy as np
import torchio as tio
import random
import glob
import re
import pdb
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
            y_i = torch.load(img_path, weights_only=True).unsqueeze(0) # weights only (?)
            y.append(y_i)
        y = torch.cat(y, dim=4)
        return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")

class HBN(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = kwargs.get('decoder')
        print(f"Number of sequences: {len(self.data)}")
        self.count_unique_subjects()
    
    def count_unique_subjects(self):
        unique_subjects = set([tup[1] for tup in self.data])
        print(f"Number of unique subjects: {len(unique_subjects)}")

    def _set_data(self, root, subject_dict):
        data = []
        
        img_root = os.path.join(root, 'img') 

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            num_frames = len([ # exclude e.g. frame_seq-10_610_620.pt
                f for f in glob.glob(os.path.join(subject_path, 'frame_*.pt'))
                if re.search(r'frame_\d+\.pt$', os.path.basename(f))
            ])
            
            session_duration = num_frames - self.sample_duration + 1 #- start_TR: kimbo change
            start_frame_adjustment = 7 if not self.adjust_hrf else 0   # kimbo change: If HRF correction is not applied, shift the start frame by an additional 7 TRs
            for start_frame in range(start_frame_adjustment, session_duration, self.stride):
                if start_frame < 0:
                        continue
                input_start_frame = start_frame + self.input_offset
                output_start_frame = start_frame
                if not (0 <= input_start_frame < session_duration and 0 <= output_start_frame < session_duration):
                    continue
                if self.decoder == 'series_decoder':
                    data_tuple = (i,
                                subject_name,
                                subject_path,
                                input_start_frame,
                                self.sample_duration,
                                num_frames,
                                target[output_start_frame:min(output_start_frame+self.sample_duration,num_frames)],
                                sex)
                elif self.decoder == 'single_target_multitask':
                    target_frame_idx = start_frame + self.sample_duration
                    if not (0 <= target_frame_idx < num_frames):
                        continue
                    target_frame = target[target_frame_idx]  # shape: [num_emotions]
                    data_tuple = (i,
                                  subject_name,
                                  subject_path,
                                  start_frame,
                                  self.sample_duration,
                                  num_frames,
                                  target_frame,
                                  sex)
                elif self.decoder == 'single_target_scalar':
                    data_tuple = (i,
                                  subject_name,
                                  subject_path,
                                  start_frame,
                                  self.stride,
                                  num_frames,
                                  target,
                                  sex)
                else:
                    raise ValueError("Invalid decoder")
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # input_offset 적용 후 범위 체크
        input_start_frame = start_frame + self.input_offset
        if not (0 <= input_start_frame < num_frames):  # 범위를 벗어나면 건너뛰기
            return None
        y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
        # print(f"Loaded sequence for subject {subject_name} at start frame {start_frame} with shape {y.shape}")

        if y is None: 
            return None # 이후 코드가 실행되지 않음

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3) 
        y = torch.nn.functional.pad(y, (7, 8, 0, 1, 7, 8), value=background_value) # adjust this padding level according to your data 
        y = y.permute(0,2,3,4,1) 

        return {
                    "fmri_sequence": y,
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex,
                } 
    
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