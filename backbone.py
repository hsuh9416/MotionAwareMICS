# Import neccesary libraries
import torch
import torch.nn as nn
import torchvision
from sklearn.decomposition import PCA
from torchvision.models import resnet18, ResNet18_Weights
import cv2

# ResNet18 백본 모델 (CIFAR-100용)
class ResNet18Backbone(nn.Module):
    def __init__(self, feature_dim=512):
        super(ResNet18Backbone, self).__init__()
        # CIFAR-100용 ResNet18 (32x32 입력)
        model = torchvision.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # CIFAR에는 maxpool이 필요 없음

        # 특성 추출기: 마지막 FC 레이어 제외
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

# ResNet18 백본 모델 (UCF101/모션 데이터셋용)
class MotionResNet18Backbone(nn.Module):
    def __init__(self, feature_dim=512):
        super(MotionResNet18Backbone, self).__init__()
        # ImageNet 사전 학습 가중치로 초기화
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 특성 추출기: 마지막 FC 레이어 제외
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = feature_dim

    def forward(self, x):
        # 입력: [B, C, T, H, W] - 배치, 채널, 시간, 높이, 너비
        B, C, T, H, W = x.shape

        # 각 프레임에 개별적으로 ResNet 적용
        features = []
        for t in range(T):
            frame = x[:, :, t]  # [B, C, H, W]
            frame_feat = self.features(frame)  # [B, feat_dim, 1, 1]
            features.append(frame_feat)

        # 모든 프레임 특성의 평균
        x = torch.stack(features, dim=2)  # [B, feat_dim, T, 1, 1]
        x = torch.mean(x, dim=2)  # 시간 차원에 대해 평균
        x = torch.flatten(x, 1)

        return x