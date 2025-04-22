import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


class HMDB51(data.Dataset):
    """HMDB51 Dataset for FSCIL."""

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, index=None, base_sess=True, frames_per_clip=16,
                 step_between_clips=8, fold=1):
        super(HMDB51, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.fold = fold

        # Setup default transforms
        if self.transform is None:
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        # Load dataset
        self.data = []
        self.targets = []
        self._load_dataset(index, base_sess)

    def _load_dataset(self, index, base_sess):
        """Load video clips based on specified indices and session type."""
        # Video folder paths based on train/test split
        split_folder = os.path.join(self.root, 'splits', f'fold_{self.fold}')
        video_folder = os.path.join(self.root, 'videos')

        # Load class indices
        if index is None:
            if base_sess:
                # Base session with first N classes
                index = np.arange(31)  # 31 base classes for HMDB51
            else:
                # For incremental sessions
                index = np.arange(31, 51)  # Remaining classes

        # Load videos and labels
        for class_idx in index:
            class_name = self._get_class_name(class_idx)
            split_file = 'train' if self.train else 'test'
            split_path = os.path.join(split_folder, f'{class_name}_{split_file}.txt')

            if not os.path.exists(split_path):
                continue

            with open(split_path, 'r') as f:
                for line in f:
                    video_name = line.strip().split(' ')[0]
                    video_path = os.path.join(video_folder, class_name, video_name)

                    # Extract frames or store video path
                    self.data.append(video_path)
                    self.targets.append(class_idx)

    def _get_class_name(self, class_idx):
        """Convert class index to class name."""
        # Mapping class indices to actual folder names
        class_names = sorted(os.listdir(os.path.join(self.root, 'videos')))
        return class_names[class_idx]

    def _load_video(self, video_path):
        """Load video frames from path."""
        frames = []
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        # Sample frames based on frames_per_clip
        if len(frames) >= self.frames_per_clip:
            start_idx = random.randint(0, len(frames) - self.frames_per_clip) if self.train else 0
            sampled_frames = frames[start_idx:start_idx + self.frames_per_clip]
        else:
            # Pad with last frame if not enough frames
            sampled_frames = frames + [frames[-1]] * (self.frames_per_clip - len(frames))

        return sampled_frames

    def __getitem__(self, index):
        """Get video clip and label."""
        video_path = self.data[index]
        target = self.targets[index]

        # Load video frames
        frames = self._load_video(video_path)

        if frames is None:
            # Fallback for corrupted videos
            return self.__getitem__((index + 1) % len(self.data))

        # Apply transform to each frame
        if self.transform:
            transformed_frames = []
            for frame in frames:
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame)
                transformed_frames.append(self.transform(pil_frame))

            # Stack frames: (T, C, H, W) -> (C, T, H, W)
            clip = torch.stack(transformed_frames, dim=0).permute(1, 0, 2, 3)
        else:
            clip = torch.tensor(np.array(frames))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)