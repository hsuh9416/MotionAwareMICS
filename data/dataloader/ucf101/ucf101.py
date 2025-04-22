from PIL import Image
import os
import os.path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset

class UCF101Dataset(VisionDataset):
    """UCF101 Action Recognition Dataset adapted for Few-Shot Class-Incremental Learning.

    Args:
        root (string): Root directory of dataset where directory 'UCF101' exists
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet.
        index (list): List of class indices to include
        base_sess (bool): If True, selects base session classes
        frames_per_clip (int): Number of frames to extract per clip
        step_between_clips (int): Step size between clips
        fold (int): Which fold to use (1, 2, or 3)
        autoaug (bool): Whether to use data augmentation
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, index=None, base_sess=None, frames_per_clip=16,
                 step_between_clips=8, fold=1):

        super(UCF101Dataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.fold = fold

        # Set appropriate transforms based on dataset requirements
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomCrop(112),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Initialize the UCF101 dataset with torchvision
        try:
            self.ucf_dataset = torchvision.datasets.UCF101(
                root=os.path.join(self.root, 'UCF101'),
                annotation_path=os.path.join(self.root, 'ucfTrainTestlist'),
                frames_per_clip=self.frames_per_clip,
                step_between_clips=self.step_between_clips,
                fold=self.fold,
                train=self.train,
                transform=None  # We'll apply our transforms later
            )
        except Exception as e:
            print(f"Error loading UCF101 dataset: {e}")
            if download:
                print("Attempting to download UCF101 dataset...")
                # Implement download code if needed
            self.ucf_dataset = None
            return

        # Extract data and targets
        self.data = []  # Will store video clips
        self.targets = []  # Will store class labels

        # Process the dataset
        for i in range(len(self.ucf_dataset)):
            try:
                video, label = self.ucf_dataset[i]
                # video shape should be [T, C, H, W]
                self.data.append(video)
                self.targets.append(label)
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

        # Convert to numpy arrays for consistency with CIFAR format
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        # Filter classes based on index
        if base_sess:
            self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)
        else:  # new Class session
            if train:
                self.data, self.targets = self.NewClassSelector(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)

        # Load class names for UCF101
        self.classes = self._load_class_names()
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def SelectfromDefault(self, data, targets, index):
        """Select samples from specific classes."""
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            if len(data_tmp) == 0:
                data_tmp = [data[j] for j in ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp.extend([data[j] for j in ind_cl])
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp

    def NewClassSelector(self, data, targets, index):
        """Select a few samples for new classes (for few-shot learning)."""
        data_tmp = []
        targets_tmp = []
        ind_list = [int(i) for i in index]
        ind_np = np.array(ind_list)
        # Reshape for N-way K-shot format (assuming 5-way 5-shot)
        index = ind_np.reshape((5, 5))
        for i in index:
            ind_cl = i
            if len(data_tmp) == 0:
                data_tmp = [data[j] for j in ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp.extend([data[j] for j in ind_cl])
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp

    def _load_class_names(self):
        """Load class names from UCF101 dataset."""
        try:
            classInd_path = os.path.join(self.root, 'ucfTrainTestlist', 'classInd.txt')
            if os.path.exists(classInd_path):
                with open(classInd_path, 'r') as f:
                    lines = f.readlines()
                    class_names = [line.strip().split(' ')[1] for line in lines]
                return class_names
            else:
                # Fallback to default class indexes
                return [str(i) for i in range(101)]
        except Exception as e:
            print(f"Error loading class names: {e}")
            return [str(i) for i in range(101)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (video, target) where target is class index of the target class.
        """
        if isinstance(self.data, list):
            video, target = self.data[index], self.targets[index]
        else:
            # Handle numpy array
            video, target = self.data[index], self.targets[index]

        # Apply transformations to each frame
        if isinstance(video, np.ndarray):
            # Convert to tensor if necessary
            video = torch.from_numpy(video)

        # Apply transforms to each frame
        transformed_frames = []
        for t in range(video.shape[0]):  # Iterate over frames
            frame = video[t].permute(1, 2, 0).numpy()  # Convert to HWC for PIL
            frame = Image.fromarray(frame.astype('uint8'))

            if self.transform is not None:
                frame = self.transform(frame)

            transformed_frames.append(frame)

        # Stack frames back together
        video = torch.stack(transformed_frames)

        # Convert video from [T, C, H, W] to [C, T, H, W] for model input
        video = video.permute(1, 0, 2, 3)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return video, target

    def __len__(self):
        return len(self.data)