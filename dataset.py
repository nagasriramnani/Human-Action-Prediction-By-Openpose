import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class KTHSkeletonDataset(Dataset):
    def __init__(self, root_dir, sequence_length=32, mode='train', split_ratio=0.8):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.mode = mode
        
        self.classes = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = [] 
        self._load_dataset(split_ratio)
        
    def _load_dataset(self, split_ratio):
        all_samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            files = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
            label = self.class_to_idx[cls_name]
            for f in files:
                all_samples.append((f, label))
        
        np.random.seed(42)
        np.random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * split_ratio)
        if self.mode == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
            
    def __len__(self):
        return len(self.samples)
    
    def augment(self, data):
        """Apply random augmentations to skeleton data (T, J, C)"""
        # 1. Random Noise (Increased)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.02, data.shape) # Was 0.01
            data += noise
            
        # 2. Random Scaling (Increased range)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2) # Was 0.9-1.1
            data *= scale
            
        # 3. Random Rotation (Increased range)
        if np.random.rand() < 0.5:
            theta = np.radians(np.random.uniform(-30, 30)) # Was -15 to 15
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array(((c, -s), (s, c)))
            # Center of mass
            center = data.mean(axis=(0, 1))
            data_centered = data - center
            # Rotate
            data_rotated = np.dot(data_centered, rotation_matrix)
            data = data_rotated + center
            
        return data

    def interpolate(self, data):
        """
        Fill zero values (missing keypoints) using linear interpolation.
        data: (T, J, C)
        """
        T, J, C = data.shape
        for j in range(J):
            for c in range(C):
                # Find valid (non-zero) indices
                valid_indices = np.where(data[:, j, c] != 0)[0]
                
                # If no valid points, skip (can't interpolate)
                if len(valid_indices) == 0:
                    continue
                    
                # If some missing, interpolate
                if len(valid_indices) < T:
                    # Create interpolation function
                    # We interpolate over the entire range [0, T-1]
                    # using the known values at valid_indices
                    data[:, j, c] = np.interp(
                        np.arange(T), 
                        valid_indices, 
                        data[valid_indices, j, c]
                    )
        return data

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path) # (T, J, 2)
        
        # Interpolate missing points (Fix for poor OpenPose detection)
        data = self.interpolate(data)
        
        # Augmentation (only for training)
        if self.mode == 'train':
            data = self.augment(data)
        
        T, J, C = data.shape
        if T < self.sequence_length:
            pad_len = self.sequence_length - T
            padding = np.zeros((pad_len, J, C), dtype=data.dtype)
            data = np.concatenate((data, padding), axis=0)
        elif T > self.sequence_length:
            if self.mode == 'train':
                start = np.random.randint(0, T - self.sequence_length)
            else:
                start = (T - self.sequence_length) // 2
            data = data[start : start + self.sequence_length]
            
        data = data.transpose(2, 0, 1) # (C, T, J)
        return torch.from_numpy(data).float(), torch.tensor(label).long()

def get_dataloaders(root_dir, batch_size=32, sequence_length=32):
    train_dataset = KTHSkeletonDataset(root_dir, sequence_length, mode='train')
    val_dataset = KTHSkeletonDataset(root_dir, sequence_length, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
