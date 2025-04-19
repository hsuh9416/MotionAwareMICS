# MIT License
#
# Copyright (c) 2023 Solang Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This function is based on the code from https://github.com/solangii/MICS
# with modifications to adapt it to the Motion-Aware MICS implementation

# Import necessary libraries
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

from config import Config
from data.dataloader.data_utils import set_up_datasets, get_dataloader

# Argument set-up for dataloader configuration
class Args:
    def __init__(self, config):
        self.dataset = config.dataset
        self.dataroot = config.dataroot
        self.batch_size_base = config.batch_size
        self.batch_size_new = getattr(config, 'batch_size_new', 0)
        self.test_batch_size = getattr(config, 'test_batch_size', config.batch_size)
        self.num_workers = config.num_workers
        self.drop_last = getattr(config, 'drop_last', False)
        self.train = getattr(config, 'train', 'vanilla')
        self.is_autoaug = getattr(config, 'is_autoaug', False)

def load_cifar100(config):
    """
    Load and preprocess CIFAR-100 dataset for Few-Shot Class Incremental Learning (FSCIL).
    This function divides the dataset into base session and incremental sessions based on the given parameters.

    Original source: https://github.com/solangii/MICS/blob/main/dataloader.py
    MIT License, Copyright (c) 2023 Solang Kim

    Modifications:
    - Adapted to work with the Config class parameters
    - Reorganized for compatibility with Motion-Aware implementation
    - Added documentation

    Args:
        base_classes (int): Number of classes for the base session
        novel_classes_per_session (int): Number of new classes per incremental session
        num_sessions (int): Number of incremental sessions

    Returns:
        tuple: (sessions_train_data, sessions_test_data) where each element is a list of datasets for each session
    """

    # Load Config
    arg = Args(config)

    # Load datasets
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transform_test)

    # Create indices by class
    train_indices = {i: [] for i in range(100)}
    for idx, (_, label) in enumerate(trainset):
        train_indices[label].append(idx)

    test_indices = {i: [] for i in range(100)}
    for idx, (_, label) in enumerate(testset):
        test_indices[label].append(idx)

    # Prepare datasets for each session
    sessions_train_data = []
    sessions_test_data = []

    # Base session (with base_classes)
    base_train_indices = []
    for cls in range(base_classes):
        base_train_indices.extend(train_indices[cls])

    base_test_indices = []
    for cls in range(base_classes):
        base_test_indices.extend(test_indices[cls])

    sessions_train_data.append(Subset(trainset, base_train_indices))
    sessions_test_data.append(Subset(testset, base_test_indices))

    # Incremental sessions (with novel classes)
    novel_start_class = base_classes
    for session in range(num_sessions):
        novel_end_class = novel_start_class + novel_classes_per_session

        # Select K-shot samples for each class in incremental session
        inc_train_indices = []
        for cls in range(novel_start_class, novel_end_class):
            # Select only shots_per_class samples from each class
            selected_indices = train_indices[cls][:Config.shots_per_class]
            inc_train_indices.extend(selected_indices)

        # Use all samples for testing
        inc_test_indices = []
        for cls in range(novel_start_class, novel_end_class):
            inc_test_indices.extend(test_indices[cls])

        sessions_train_data.append(Subset(trainset, inc_train_indices))
        sessions_test_data.append(Subset(testset, inc_test_indices))

        novel_start_class = novel_end_class

    return sessions_train_data, sessions_test_data

def load_ucf101(base_classes, novel_classes_per_session, num_sessions, shots_per_class, max_classes=30):
    """
    Load and preprocess UCF101 dataset for Few-Shot Class Incremental Learning (FSCIL) with motion awareness.
    This function divides the dataset into base session and incremental sessions based on the given parameters.

    Args:
        base_classes (int): Number of classes for the base session
        novel_classes_per_session (int): Number of new classes per incremental session
        num_sessions (int): Number of incremental sessions
        shots_per_class (int): Number of samples per class for incremental sessions
        max_classes (int): Maximum number of classes to use

    Returns:
        tuple: (sessions_train_data, sessions_test_data) where each element is a list of datasets for each session
    """
    # Data augmentation and preprocessing for training
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocessing for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load UCF101 dataset
    trainset = torchvision.datasets.UCF101(root='/content/drive/MyDrive/MotionAwareMICS/data/UCF101',
                                           annotation_path='/content/drive/MyDrive/MotionAwareMICS/data/ucfTrainTestlist',
                                           frames_per_clip=8,
                                           step_between_clips=16,
                                           fold=1,
                                           train=True,
                                           transform=transform_train)
    testset = torchvision.datasets.UCF101(root='/content/drive/MyDrive/MotionAwareMICS/data/UCF101',
                                          annotation_path='/content/drive/MyDrive/MotionAwareMICS/data/ucfTrainTestlist',
                                          frames_per_clip=8,
                                          step_between_clips=16,
                                          fold=1,
                                          train=False,
                                          transform=transform_test)

    # Create indices by class
    train_indices = {i: [] for i in range(max_classes)}  # Total: 101, Test only: 30
    for idx, (_, label) in enumerate(trainset):
        train_indices[label].append(idx)

    test_indices = {i: [] for i in range(max_classes)}
    for idx, (_, label) in enumerate(testset):
        test_indices[label].append(idx)

    # Prepare datasets for each session
    sessions_train_data = []
    sessions_test_data = []

    # Base session (with base_classes)
    base_train_indices = []
    for cls in range(base_classes):
        base_train_indices.extend(train_indices[cls])

    base_test_indices = []
    for cls in range(base_classes):
        base_test_indices.extend(test_indices[cls])

    sessions_train_data.append(Subset(trainset, base_train_indices))
    sessions_test_data.append(Subset(testset, base_test_indices))

    # Incremental sessions (with novel classes)
    novel_start_class = base_classes
    for session in range(num_sessions):
        novel_end_class = novel_start_class + novel_classes_per_session

        # Ensure we don't exceed the total number of classes
        novel_end_class = min(novel_end_class, 101)

        # Select K-shot samples for each class in incremental session
        inc_train_indices = []
        for cls in range(novel_start_class, novel_end_class):
            # Select only shots_per_class samples from each class
            if cls in train_indices and len(train_indices[cls]) >= shots_per_class:
                selected_indices = train_indices[cls][:shots_per_class]
                inc_train_indices.extend(selected_indices)
            elif cls in train_indices:
                # If fewer than shots_per_class samples are available, use all of them
                inc_train_indices.extend(train_indices[cls])

        # Use all samples for testing
        inc_test_indices = []
        for cls in range(novel_start_class, novel_end_class):
            if cls in test_indices:
                inc_test_indices.extend(test_indices[cls])

        # Only append non-empty datasets
        if inc_train_indices:
            sessions_train_data.append(Subset(trainset, inc_train_indices))
        if inc_test_indices:
            sessions_test_data.append(Subset(testset, inc_test_indices))

        novel_start_class = novel_end_class

        # Break if we've used all available classes
        if novel_end_class >= 101:
            break

    return sessions_train_data, sessions_test_data