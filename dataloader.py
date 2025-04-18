# Import neccesary libraries
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from torch.utils.data import Subset, Dataset
from config import Config

# CIFAR-100 데이터셋 로드 및 전처리 함수
def load_cifar100(base_classes, novel_classes_per_session, num_sessions):
    # 데이터 증강 및 전처리
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # 데이터셋 로드
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)

    # 클래스별 인덱스 생성
    train_indices = {i: [] for i in range(100)}
    for idx, (_, label) in enumerate(trainset):
        train_indices[label].append(idx)

    test_indices = {i: [] for i in range(100)}
    for idx, (_, label) in enumerate(testset):
        test_indices[label].append(idx)

    # 세션별 데이터셋 준비
    sessions_train_data = []
    sessions_test_data = []

    # 기본 세션 (base classes)
    base_train_indices = []
    for cls in range(base_classes):
        base_train_indices.extend(train_indices[cls])

    base_test_indices = []
    for cls in range(base_classes):
        base_test_indices.extend(test_indices[cls])

    sessions_train_data.append(Subset(trainset, base_train_indices))
    sessions_test_data.append(Subset(testset, base_test_indices))

    # 증분 세션 (novel classes)
    novel_start_class = base_classes
    for session in range(num_sessions):
        novel_end_class = novel_start_class + novel_classes_per_session

        # 각 클래스에서 K-shot 샘플 선택
        inc_train_indices = []
        for cls in range(novel_start_class, novel_end_class):
            # 각 클래스에서 shots_per_class 개의 샘플만 선택
            selected_indices = train_indices[cls][:Config.shots_per_class]
            inc_train_indices.extend(selected_indices)

        # 테스트 셋은 모든 샘플 사용
        inc_test_indices = []
        for cls in range(novel_start_class, novel_end_class):
            inc_test_indices.extend(test_indices[cls])

        sessions_train_data.append(Subset(trainset, inc_train_indices))
        sessions_test_data.append(Subset(testset, inc_test_indices))

        novel_start_class = novel_end_class

    return sessions_train_data, sessions_test_data


# Kinetics-400 데이터셋의 서브셋을 사용 (모션 인식 실험용)
# 실제로는 TorchVision의 내장 데이터셋을 사용하는 것이 좋지만,
# 여기서는 간략화를 위해 샘플 코드만 제공
class KineticsSubset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None, num_classes=51, num_clips=5):
        """
        실제 구현에서는 torchvision.datasets.Kinetics400을 사용하는 것이 좋습니다.
        여기서는 개념적인 구현만 제공합니다.
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.samples = self._load_samples()

    def _load_samples(self):
        # 실제 구현에서는 데이터셋 파일 경로를 로드합니다
        # 여기서는 더미 데이터를 생성합니다
        samples = []
        for cls in range(self.num_classes):
            for clip in range(self.num_clips):
                samples.append((f"dummy_path_{cls}_{clip}.mp4", cls))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 실제 구현에서는 비디오를 로드하고 프레임을 추출합니다
        # 여기서는 더미 텐서를 생성합니다 (3채널, 16프레임, 112x112 해상도)
        video = torch.rand(3, 16, 112, 112)

        if self.transform:
            # 각 프레임에 transform 적용
            frames = [self.transform(video[:, i]) for i in range(video.shape[1])]
            video = torch.stack(frames, dim=1)

        return video, label


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


