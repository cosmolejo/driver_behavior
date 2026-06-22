import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List

# ==========================
# 1. Dataset
# ==========================
class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sequence_length: int = -1,
        sample_one_each: int = 1, 
        stride: int = 8,
        transform=None
    ):
        """
        root_dir: path to train/val folder
        sequence_length: number of frames per video (-1 to use full video)
        transform: torchvision transforms to apply on each frame
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.sample_one_each = sample_one_each
        self.stride = stride
        self.transform = transform
        self.samples = []  # List of (video_path, label)
        self.class_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(sorted(os.listdir(root_dir)))
        }
        
        self._build_index()

    def _build_index(self):
         for cls_name, label in self.class_to_idx.items():
            cls_path = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for session_name in os.listdir(cls_path):
                session_path = os.path.join(cls_path, session_name)
                if not os.path.isdir(session_path):
                    continue
                for video_name in os.listdir(session_path):
                    video_path = os.path.join(session_path, video_name, "face")
                    if os.path.isdir(video_path):
                        frame_files = sorted(os.listdir(video_path))
                        num_frames = len(frame_files)

                        # create sliding windows
                        for start in range(0, num_frames - self.sequence_length + 1, self.stride):
                            self.samples.append(
                                (video_path, label, start)
                            )

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, start):
        frame_files = sorted(os.listdir(video_path))
        end = start + self.sequence_length
        sequence_files = frame_files[start:(end if end < len(frame_files) else len(frame_files)):self.sample_one_each]
        frames = []
        for f in sequence_files:
            img = cv2.imread(os.path.join(video_path, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)

            frames.append(img)

        return torch.stack(frames, dim=1)  # (C, T, H, W)

    def __getitem__(self, idx):
        video_path, label, start = self.samples[idx]
        video_tensor = self._load_frames(video_path, start)
        return video_tensor, torch.tensor(label, dtype=torch.long), Path(video_path).parts[-5:-2]