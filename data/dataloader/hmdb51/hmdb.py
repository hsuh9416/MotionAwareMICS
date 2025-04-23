import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import HMDB51

import av
import numpy as np
from PIL import Image


class HMDB51(data.Dataset):
    """HMDB51 Dataset for FSCIL, adapted to use torchvision's download functionality."""

    def __init__(self, root='data', train=True, transform=None, target_transform=None,
                 download=False, index=None, base_sess=True, frames_per_clip=16,
                 step_between_clips=8, fold=1, clip_length=16):
        super(HMDB51, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.fold = fold
        self.clip_length = clip_length

        # Setup default transforms if none provided
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

        # Set paths for data
        self.data_dir = os.path.join(self.root, 'hmdb51')
        self.video_dir = os.path.join(self.data_dir , 'videos')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # Download dataset if requested
        if download:
            self._download_hmdb51()

        # Load dataset
        self.data = []
        self.targets = []
        self._load_dataset(index, base_sess)

    def _download_hmdb51(self):
        """Download HMDB51 dataset using torchvision."""
        try:

            # Use torchvision's HMDB51 if available
            from torchvision.datasets import HMDB51 as TorchHMDB51

            # Download dataset (torchvision will handle the downloading)
            _ = TorchHMDB51(
                root=self.data_dir,
                annotation_path=os.path.join(self.data_dir, f'split_{self.fold}.txt'),
                frames_per_clip=self.frames_per_clip,
                step_between_clips=self.step_between_clips,
                fold=self.fold,
                train=True,
                download=True,
                num_workers=4
            )

            print(f"HMDB51 dataset downloaded to {self.data_dir}")
        except Exception as e:
            print(f"Error downloading HMDB51 via torchvision: {str(e)}")
            print(
                "Please download HMDB51 manually from http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/")
            print(f"and extract it to {self.data_dir}")

    def _load_dataset(self, index, base_sess):
        """Load video clips based on specified indices and session type."""
        # Define class mappings
        self.all_classes = sorted([d for d in os.listdir(os.path.join(self.data_dir, 'videos'))
                                   if os.path.isdir(os.path.join(self.data_dir, 'videos', d))])

        # Determine which classes to include based on session
        if index is None:
            if base_sess:
                # Base session with first 31 classes
                index = np.arange(31)
            else:
                # For incremental sessions
                index = np.arange(31, min(51, len(self.all_classes)))  # Remaining classes up to 51

        # Create split file for train/test
        split_type = 1 if self.train else 2  # 1 for train, 2 for test

        # Load videos and labels for specified classes
        for class_idx in index:
            if class_idx >= len(self.all_classes):
                continue

            class_name = self.all_classes[class_idx]
            class_dir = os.path.join(self.data_dir, 'videos', class_name)

            if not os.path.exists(class_dir):
                continue

            # Get all video files for this class
            video_files = [f for f in os.listdir(class_dir) if f.endswith(('.avi', '.mp4'))]

            # Split videos into train/test based on fold and split_type
            # For simplicity, we'll use a deterministic split based on file order
            # In a real implementation, you'd use the official splits provided with HMDB51
            videos = sorted(video_files)
            num_videos = len(videos)

            if self.train:
                # Use first 70% for training
                selected_videos = videos[:int(0.7 * num_videos)]
            else:
                # Use remaining 30% for testing
                selected_videos = videos[int(0.7 * num_videos):]

            # Add selected videos to our dataset
            for video_name in selected_videos:
                video_path = os.path.join(class_dir, video_name)
                self.data.append(video_path)
                self.targets.append(class_idx)

    def extract_frames(self, video_path):
        """Extract frames from a video file using PyAV."""
        frames = []
        try:
            container = av.open(video_path)
            # Get video stream
            stream = container.streams.video[0]

            # Calculate frame indices to sample evenly throughout the video
            total_frames = stream.frames
            if total_frames == 0:  # Some videos might not report correct frame count
                # Count frames manually
                total_frames = sum(1 for _ in container.decode(stream))
                # Reset container to start
                container = av.open(video_path)
                stream = container.streams.video[0]

            if total_frames <= self.clip_length:
                # If video is too short, duplicate frames
                indices = list(range(total_frames)) + [total_frames - 1] * (self.clip_length - total_frames)
            else:
                # Sample frames evenly
                indices = np.linspace(0, total_frames - 1, self.clip_length, dtype=int)

            # Extract frames at the calculated indices
            for i, frame in enumerate(container.decode(stream)):
                if i in indices:
                    img = frame.to_image().convert("RGB")
                    frames.append(img)

                if len(frames) == self.clip_length:
                    break

            # If we couldn't extract enough frames, duplicate the last frame
            while len(frames) < self.clip_length:
                frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))

            container.close()

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {str(e)}")
            # Return empty frames as fallback
            frames = [Image.new('RGB', (224, 224)) for _ in range(self.clip_length)]

        return frames

    def __getitem__(self, index):
        """Get video clip and label."""
        video_path = self.data[index]
        target = self.targets[index]

        # Extract frames from video
        frames = self.extract_frames(video_path)

        # Apply transforms to each frame
        if self.transform:
            transformed_frames = []
            for frame in frames:
                transformed_frames.append(self.transform(frame))

            # Stack frames: (T, C, H, W) -> (C, T, H, W)
            clip = torch.stack(transformed_frames, dim=0).permute(1, 0, 2, 3)
        else:
            # Convert PIL images to tensors
            clip = torch.stack([transforms.ToTensor()(frame) for frame in frames], dim=0).permute(1, 0, 2, 3)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)