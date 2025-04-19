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
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset
from config import Config

def load_cifar100(base_classes, novel_classes_per_session, num_sessions):
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
    # Data augmentation and preprocessing for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Preprocessing for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

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

# UCF101 데이터셋 로드 (TorchVision 내장 기능 사용)
def load_ucf101(base_classes, novel_classes_per_session, num_sessions, shots_per_class):
    # 데이터 증강 및 전처리
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # UCF101 데이터셋 로드 (TorchVision 내장 기능 사용)
    # 참고: 실제 구현에서는 아래와 같이 로드할 수 있습니다
    # trainset = torchvision.datasets.UCF101(root='./data', annotation_path='ucfTrainTestlist',
    #                                       frames_per_clip=16, step_between_clips=8,
    #                                       fold=1, train=True, transform=transform_train, download=True)
    # testset = torchvision.datasets.UCF101(root='./data', annotation_path='ucfTrainTestlist',
    #                                      frames_per_clip=16, step_between_clips=8,
    #                                      fold=1, train=False, transform=transform_test, download=True)

    # 간략화를 위해 더미 데이터셋 생성
    trainset = KineticsSubset(root_dir='./data', subset='train', transform=transform_train,
                             num_classes=101, num_clips=50)
    testset = KineticsSubset(root_dir='./data', subset='test', transform=transform_test,
                            num_classes=101, num_clips=20)

    # 세션별 데이터셋 준비 (CIFAR-100과 유사한 방식)
    # 실제 구현에서는 클래스별 인덱스를 구성하고 세션별로 나누는 로직이 필요합니다

    # 여기서는 개념적인 설명만 제공합니다
    sessions_train_data = []
    sessions_test_data = []

    # 기본 세션 및 증분 세션 구성 (실제 구현 필요)
    # ...

    return sessions_train_data, sessions_test_data


