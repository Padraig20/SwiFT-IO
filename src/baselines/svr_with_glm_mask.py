"""
SVR with GLM Significant Voxel Mask for fMRI emotion prediction

This module provides SVR baseline using only GLM-identified significant voxels
from 2nd level analysis, allowing comparison with whole-brain approaches.

Two reduction modes:
1. glm_pca: GLM sig voxels -> PCA -> SVR (temporal information preserved)
2. glm_direct: GLM sig voxels -> time-average -> SVR (simpler, comparable to time_avg)

Author: For comparison with SwiFT-IO
"""

import os
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Optional, Literal
from tqdm import tqdm
import torch

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import multiprocessing

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. Install with: pip install nibabel")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .nonzero_metrics import NonZeroMetricsCalculator


class SVRWithGLMMask:
    """
    SVR baseline using GLM significant voxel masks

    Uses significant voxels identified from 2nd level GLM analysis
    to reduce feature dimension while preserving task-relevant information.

    Two modes:
    - 'glm_pca': Apply PCA to masked voxels, concat across time
    - 'glm_direct': Time-average masked voxels (single static pattern)
    """

    # Default path to GLM masks (from 2nd level analysis)
    GLM_MASK_DIR = '/scratch/connectome/kimbo/GLM-Baseline-Test/results/full_analysis/smooth_motion/threshold_nonparam/sig_masks_for_ridge/'

    EMOTION_NAMES = ['Anger', 'Happy', 'Fear', 'Sad', 'Excited', 'Positive', 'Negative']
    EMOTION_FILE_MAP = {
        'Anger': 'anger',
        'Happy': 'happy',
        'Fear': 'fear',
        'Sad': 'sad',
        'Excited': 'excited',
        'Positive': 'positive',
        'Negative': 'negative'
    }

    def __init__(self,
                 num_emotions: int = 7,
                 sequence_length: int = 20,
                 reduction_mode: Literal['glm_pca', 'glm_direct'] = 'glm_pca',
                 pca_components: int = 100,
                 mask_type: str = 'union',
                 glm_mask_dir: str = None,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 standardize: bool = True,
                 use_wandb: bool = False):
        """
        Initialize SVR with GLM mask

        Args:
            num_emotions: Number of emotions to predict (default: 7)
            sequence_length: Length of fMRI sequence
            reduction_mode:
                - 'glm_pca': GLM sig voxels -> PCA -> SVR
                - 'glm_direct': GLM sig voxels -> time-average -> SVR
            pca_components: Number of PCA components (for glm_pca mode)
            mask_type: 'union' (all emotions' sig voxels combined)
            glm_mask_dir: Path to GLM mask directory
            kernel: SVR kernel ('linear', 'rbf', 'poly')
            C: SVR regularization parameter
            epsilon: SVR epsilon parameter
            standardize: Whether to standardize features
            use_wandb: Whether to log to wandb
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for GLM mask loading. "
                            "Install with: pip install nibabel")

        self.num_emotions = num_emotions
        self.sequence_length = sequence_length
        self.reduction_mode = reduction_mode
        self.pca_components = pca_components
        self.mask_type = mask_type
        self.glm_mask_dir = glm_mask_dir or self.GLM_MASK_DIR
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.standardize = standardize
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # GLM masks
        self.union_mask = None
        self.union_mask_flat = None
        self.mask_indices = None
        self.n_masked_voxels = 0
        self.per_emotion_masks = {}

        # SVR models (one per emotion)
        self.models = [
            SVR(kernel=kernel, C=C, epsilon=epsilon, cache_size=1000, verbose=False)
            for _ in range(num_emotions)
        ]

        # Scalers for feature standardization
        self.scalers = [StandardScaler() for _ in range(num_emotions)] if standardize else None

        # PCA model (for glm_pca mode)
        if reduction_mode == 'glm_pca':
            self.pca_model = IncrementalPCA(n_components=pca_components, batch_size=1000)
        else:
            self.pca_model = None

        # Metrics calculator
        self.metrics_calculator = None  # Initialized with scaler during fit

        self.fitted = False
        self.feature_dim = None

        print(f"\n{'='*80}")
        print(f"SVR with GLM Mask Initialized")
        print(f"{'='*80}")
        print(f"  Reduction mode: {reduction_mode}")
        if reduction_mode == 'glm_pca':
            print(f"  PCA components: {pca_components}")
        print(f"  Mask type: {mask_type}")
        print(f"  Kernel: {kernel}, C: {C}, epsilon: {epsilon}")
        print(f"{'='*80}")

    def load_glm_masks(self):
        """
        Load GLM significant voxel masks from NIfTI files

        Expected files:
        - union_mask.nii.gz: Combined mask of all emotions
        - sig_mask_anger.nii.gz, sig_mask_happy.nii.gz, etc.
        """
        print(f"\n{'='*80}")
        print("Loading GLM Significant Voxel Masks")
        print(f"{'='*80}")
        print(f"  Directory: {self.glm_mask_dir}")

        # Load union mask
        union_path = os.path.join(self.glm_mask_dir, 'union_mask.nii.gz')
        if not os.path.exists(union_path):
            raise FileNotFoundError(f"Union mask not found: {union_path}")

        union_nii = nib.load(union_path)
        self.union_mask = union_nii.get_fdata() > 0  # Binary mask
        self.union_mask_shape = self.union_mask.shape

        # Flatten and get indices
        self.union_mask_flat = self.union_mask.flatten()
        self.mask_indices = np.where(self.union_mask_flat)[0]
        self.n_masked_voxels = len(self.mask_indices)

        print(f"\n  Union mask shape: {self.union_mask_shape}")
        print(f"  Total voxels: {np.prod(self.union_mask_shape):,}")
        print(f"  Significant voxels: {self.n_masked_voxels:,} "
              f"({100*self.n_masked_voxels/np.prod(self.union_mask_shape):.1f}%)")

        # Load per-emotion masks
        print(f"\n  Per-emotion significant voxels:")
        for emotion_name, file_key in self.EMOTION_FILE_MAP.items():
            mask_path = os.path.join(self.glm_mask_dir, f'sig_mask_{file_key}.nii.gz')
            if os.path.exists(mask_path):
                mask_nii = nib.load(mask_path)
                mask_data = mask_nii.get_fdata() > 0
                self.per_emotion_masks[emotion_name] = mask_data
                n_voxels = mask_data.sum()
                print(f"    {emotion_name}: {n_voxels:,}")
            else:
                print(f"    {emotion_name}: (file not found)")

        print(f"\n{'='*80}")

    def _apply_mask(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply GLM union mask to a 3D volume

        Args:
            volume: 3D array of shape (X, Y, Z) or flattened

        Returns:
            Masked voxels only (n_masked_voxels,)
        """
        if volume.ndim == 3:
            volume_flat = volume.flatten()
        else:
            volume_flat = volume

        return volume_flat[self.mask_indices]

    def reduce_features(self, fmri_seq: np.ndarray) -> np.ndarray:
        """
        Apply GLM mask and dimensionality reduction to fMRI sequence

        Args:
            fmri_seq: 4D array (seq_len, X, Y, Z) or (X, Y, Z, seq_len)
                      or (X, Y, Z, 1) for single frame

        Returns:
            Reduced feature vector
        """
        # Handle different input shapes
        if fmri_seq.ndim == 4:
            # Check if last dim is 1 (single frame with channel)
            if fmri_seq.shape[-1] == 1:
                fmri_seq = fmri_seq.squeeze(-1)  # (X, Y, Z)
                fmri_seq = fmri_seq[np.newaxis, ...]  # (1, X, Y, Z)
            # Check if need to transpose (X, Y, Z, T) -> (T, X, Y, Z)
            elif fmri_seq.shape[-1] < fmri_seq.shape[0]:
                fmri_seq = np.transpose(fmri_seq, (3, 0, 1, 2))

        seq_len = fmri_seq.shape[0]

        # Apply mask to each timepoint
        masked_frames = []
        for t in range(seq_len):
            masked = self._apply_mask(fmri_seq[t])
            masked_frames.append(masked)

        # Stack: (seq_len, n_masked_voxels)
        masked_seq = np.stack(masked_frames)

        if self.reduction_mode == 'glm_pca':
            # Apply PCA to each timepoint, concat across time
            reduced_timepoints = []
            for t in range(seq_len):
                reduced = self.pca_model.transform(masked_seq[t].reshape(1, -1))
                reduced_timepoints.append(reduced.flatten())

            # Final: (seq_len * pca_components,)
            features = np.concatenate(reduced_timepoints)

        elif self.reduction_mode == 'glm_direct':
            # Time-average the masked voxels
            # Final: (n_masked_voxels,)
            features = masked_seq.mean(axis=0)

        return features

    def fit_pca_on_masked_data(self, dataloader, output_dir: str = None):
        """
        Fit IncrementalPCA on GLM-masked training data

        Args:
            dataloader: Training dataloader
            output_dir: Directory for checkpoints
        """
        if self.reduction_mode != 'glm_pca':
            return

        # Check for existing checkpoint
        pca_checkpoint_path = os.path.join(output_dir, 'glm_pca_model_checkpoint.pkl') if output_dir else None

        if pca_checkpoint_path and os.path.exists(pca_checkpoint_path):
            print(f"\n  Loading PCA from checkpoint: {pca_checkpoint_path}")
            with open(pca_checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.pca_model = checkpoint['pca_model']
                print(f"  Loaded PCA with {checkpoint['total_frames']:,} frames")
                print(f"  Variance explained: {checkpoint['variance_explained']:.2%}")
            return

        print(f"\n{'='*80}")
        print("Fitting IncrementalPCA on GLM-masked training data...")
        print(f"{'='*80}")

        total_frames = 0

        for batch in tqdm(dataloader, desc="Fitting PCA on masked voxels"):
            fmri_data = batch['fmri_sequence']

            # Handle tensor
            if isinstance(fmri_data, torch.Tensor):
                fmri_data = fmri_data.numpy()

            # Shape: (B, 1, X, Y, Z, S) or (B, X, Y, Z, S)
            if fmri_data.ndim == 6:
                fmri_data = fmri_data.squeeze(1)  # Remove channel

            batch_size = fmri_data.shape[0]
            seq_len = fmri_data.shape[-1]

            # Process each sample
            for b in range(batch_size):
                sample = fmri_data[b]  # (X, Y, Z, S)

                # Apply mask to each timepoint
                for t in range(seq_len):
                    if sample.ndim == 4:
                        frame = sample[:, :, :, t]  # (X, Y, Z)
                    else:
                        frame = sample[t]  # Already (X, Y, Z)

                    masked = self._apply_mask(frame)
                    self.pca_model.partial_fit(masked.reshape(1, -1).astype(np.float32))
                    total_frames += 1

        variance_explained = self.pca_model.explained_variance_ratio_.sum()
        print(f"\n  PCA fitted on {total_frames:,} masked frames")
        print(f"  Variance explained: {variance_explained:.2%}")

        # Save checkpoint
        if pca_checkpoint_path:
            os.makedirs(os.path.dirname(pca_checkpoint_path), exist_ok=True)
            checkpoint = {
                'pca_model': self.pca_model,
                'total_frames': total_frames,
                'variance_explained': variance_explained
            }
            with open(pca_checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  Saved PCA checkpoint: {pca_checkpoint_path}")

        print(f"{'='*80}")

    def prepare_data_from_dataloader(self, dataloader, mode: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training/evaluation data with GLM mask applied

        Args:
            dataloader: PyTorch dataloader
            mode: 'train', 'valid', or 'test'

        Returns:
            X: (n_samples, feature_dim)
            Y: (n_samples, num_emotions)
        """
        X_list = []
        Y_list = []

        print(f"\n  Preparing {mode} data with GLM mask ({self.reduction_mode})...")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {mode}")):
            fmri_data = batch['fmri_sequence']
            target_data = batch['target']

            # Handle tensors
            if isinstance(fmri_data, torch.Tensor):
                fmri_data = fmri_data.numpy()
            if isinstance(target_data, torch.Tensor):
                target_data = target_data.numpy()

            # Shape: (B, 1, X, Y, Z, S) -> (B, X, Y, Z, S)
            if fmri_data.ndim == 6:
                fmri_data = fmri_data.squeeze(1)

            batch_size = fmri_data.shape[0]

            # Print shape on first batch
            if batch_idx == 0:
                print(f"    First batch fMRI shape: {fmri_data.shape}")
                print(f"    First batch target shape: {target_data.shape}")

            for b in range(batch_size):
                sample = fmri_data[b]  # (X, Y, Z, S)
                targets = target_data[b]  # (S, num_emotions)

                # Reduce features
                features = self.reduce_features(sample)

                # Average targets across time
                targets_mean = targets.mean(axis=0)  # (num_emotions,)

                X_list.append(features)
                Y_list.append(targets_mean)

        X = np.stack(X_list).astype(np.float32)
        Y = np.stack(Y_list).astype(np.float32)

        print(f"    {mode} data: X={X.shape}, Y={Y.shape}")

        if self.feature_dim is None:
            self.feature_dim = X.shape[1]
            print(f"    Feature dimension: {self.feature_dim:,}")

        return X, Y

    def _train_single_emotion(self, e: int, X_train: np.ndarray, Y_train: np.ndarray) -> Dict:
        """
        Train SVR for a single emotion

        Args:
            e: Emotion index
            X_train: Training features
            Y_train: Training targets

        Returns:
            Dictionary with model, scaler, and metrics
        """
        y_train = Y_train[:, e]

        if self.standardize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            scaler = None
            X_train_scaled = X_train

        # Fit SVR
        model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon,
                   cache_size=1000, verbose=False)
        model.fit(X_train_scaled, y_train)

        # Training predictions
        y_pred = model.predict(X_train_scaled)

        mse = mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)

        return {
            'emotion_idx': e,
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'mae': mae
        }

    def fit(self, train_dataloader, scaler=None, output_dir: str = None) -> Dict[str, float]:
        """
        Fit SVR models on training data

        Args:
            train_dataloader: PyTorch dataloader for training
            scaler: StandardScaler from data_module (for non-zero metrics)
            output_dir: Directory for checkpoints

        Returns:
            Training metrics dictionary
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize metrics calculator
        self.metrics_calculator = NonZeroMetricsCalculator(
            scaler=scaler,
            label_scaling_method='standardization'
        )

        print(f"\n{'='*80}")
        print(f"Training SVR with GLM Mask ({self.reduction_mode} mode)")
        print(f"{'='*80}")

        # Step 1: Load GLM masks
        self.load_glm_masks()

        # Step 2: Fit PCA if using glm_pca mode
        self.fit_pca_on_masked_data(train_dataloader, output_dir)

        # Step 3: Prepare training data
        data_checkpoint_path = os.path.join(output_dir, 'glm_train_data_checkpoint.pkl') if output_dir else None

        if data_checkpoint_path and os.path.exists(data_checkpoint_path):
            print(f"\n  Loading training data from checkpoint: {data_checkpoint_path}")
            with open(data_checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                X_train = checkpoint['X_train']
                Y_train = checkpoint['Y_train']
                self.feature_dim = checkpoint['feature_dim']
            print(f"    Loaded X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        else:
            X_train, Y_train = self.prepare_data_from_dataloader(train_dataloader, mode='train')

            # Save checkpoint
            if data_checkpoint_path:
                print(f"\n  Saving training data checkpoint...")
                checkpoint = {
                    'X_train': X_train,
                    'Y_train': Y_train,
                    'feature_dim': self.feature_dim
                }
                with open(data_checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"    Saved: {data_checkpoint_path}")
                print(f"    Size: {os.path.getsize(data_checkpoint_path) / 1024**2:.1f} MB")

        # Step 4: Train SVR for each emotion
        print(f"\n  Training SVR models for each emotion...")

        # Determine number of parallel jobs
        n_jobs = min(
            int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count())),
            self.num_emotions,
            3  # Limit to avoid OOM
        )
        print(f"    Using {n_jobs} parallel workers")

        # Parallel training
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._train_single_emotion)(e, X_train, Y_train)
            for e in range(self.num_emotions)
        )

        # Process results
        train_metrics = {}
        for result in results:
            e = result['emotion_idx']
            self.models[e] = result['model']
            if self.standardize:
                self.scalers[e] = result['scaler']

            mse = result['mse']
            mae = result['mae']
            emotion_name = self.EMOTION_NAMES[e]

            train_metrics[f'train_mse_{emotion_name}'] = mse
            train_metrics[f'train_mae_{emotion_name}'] = mae

            print(f"    {emotion_name}: MSE={mse:.4f}, MAE={mae:.4f}")

            # Save emotion checkpoint
            if output_dir:
                ckpt_path = os.path.join(output_dir, f'glm_svr_emotion_{e}_checkpoint.pkl')
                with open(ckpt_path, 'wb') as f:
                    pickle.dump({
                        'model': self.models[e],
                        'scaler': self.scalers[e] if self.standardize else None,
                        'mse': mse,
                        'mae': mae
                    }, f)

        # Overall metrics
        train_metrics['train_mse'] = np.mean([train_metrics[f'train_mse_{n}']
                                              for n in self.EMOTION_NAMES])
        train_metrics['train_mae'] = np.mean([train_metrics[f'train_mae_{n}']
                                              for n in self.EMOTION_NAMES])

        print(f"\n  Overall Training - MSE: {train_metrics['train_mse']:.4f}, "
              f"MAE: {train_metrics['train_mae']:.4f}")

        self.fitted = True
        print(f"{'='*80}")

        return train_metrics

    def evaluate(self, dataloader, mode: str = 'test') -> Dict[str, float]:
        """
        Evaluate with full non-zero metrics

        Args:
            dataloader: PyTorch dataloader
            mode: 'valid' or 'test'

        Returns:
            Dictionary with all metrics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        print(f"\n{'='*80}")
        print(f"Evaluating on {mode} set with Non-Zero Metrics")
        print(f"{'='*80}")

        # Prepare data
        X_eval, Y_eval = self.prepare_data_from_dataloader(dataloader, mode=mode)

        # Predict for each emotion
        Y_pred = np.zeros_like(Y_eval)

        for e in range(self.num_emotions):
            if self.standardize:
                X_scaled = self.scalers[e].transform(X_eval)
            else:
                X_scaled = X_eval

            Y_pred[:, e] = self.models[e].predict(X_scaled)

        # Calculate metrics for each emotion
        all_metrics = {}

        for e in range(self.num_emotions):
            emotion_metrics = self.metrics_calculator.calculate_metrics(
                y_true_scaled=Y_eval[:, e],
                y_pred_scaled=Y_pred[:, e],
                emotion_idx=e,
                mode_str=mode
            )
            all_metrics.update(emotion_metrics)

        # Add summary metrics
        summary = self.metrics_calculator.calculate_summary_metrics(all_metrics, mode)
        all_metrics.update(summary)

        # Print summary table
        self.metrics_calculator.print_summary(all_metrics, mode)

        return all_metrics

    def save(self, save_path: str):
        """Save all model components"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state = {
            'models': self.models,
            'scalers': self.scalers,
            'pca_model': self.pca_model,
            'num_emotions': self.num_emotions,
            'sequence_length': self.sequence_length,
            'reduction_mode': self.reduction_mode,
            'pca_components': self.pca_components,
            'mask_type': self.mask_type,
            'glm_mask_dir': self.glm_mask_dir,
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'standardize': self.standardize,
            'feature_dim': self.feature_dim,
            'n_masked_voxels': self.n_masked_voxels,
            'union_mask_shape': self.union_mask_shape if hasattr(self, 'union_mask_shape') else None
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"\nSVR model saved to {save_path}")

    def load(self, load_path: str):
        """Load model components"""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)

        self.models = state['models']
        self.scalers = state['scalers']
        self.pca_model = state['pca_model']
        self.num_emotions = state['num_emotions']
        self.sequence_length = state['sequence_length']
        self.reduction_mode = state['reduction_mode']
        self.pca_components = state['pca_components']
        self.mask_type = state['mask_type']
        self.glm_mask_dir = state['glm_mask_dir']
        self.kernel = state['kernel']
        self.C = state['C']
        self.epsilon = state['epsilon']
        self.standardize = state['standardize']
        self.feature_dim = state['feature_dim']
        self.n_masked_voxels = state.get('n_masked_voxels', 0)

        # Need to reload masks for evaluation
        self.load_glm_masks()

        self.fitted = True
        print(f"\nSVR model loaded from {load_path}")
