# Import neccesary libraries
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

from backbone import *
from mics_ext import *

# MICS 알고리즘의 핵심 구현
class MICS(nn.Module):
    def __init__(self, config, num_classes):
        super(MICS, self).__init__()
        self.config = config
        self.device = config.device

        # 백본 모델 선택
        if config.dataset == 'cifar100':
            self.backbone = ResNet18Backbone(config.feature_dim).to(self.device)
        else:  # 모션 데이터셋
            self.backbone = MotionResNet18Backbone(config.feature_dim).to(self.device)

        # 클래스 분류기
        self.classifiers = nn.Parameter(torch.randn(num_classes, config.feature_dim).to(self.device))

        # 모션 인식 모듈
        self.motion_mixup = MotionAwareMixup(config).to(self.device)

        # 가상 클래스 인덱스 추적
        self.virtual_class_indices = {}

    def forward(self, x, return_features=False):
        features = self.backbone(x)

        if return_features:
            return features

        # 코사인 유사도 기반 분류
        logits = F.linear(F.normalize(features, p=2, dim=1),
                         F.normalize(self.classifiers, p=2, dim=1))
        return logits / 0.1  # 온도 스케일링

    def compute_midpoint_classifier(self, class1, class2):
        """중간점 분류기 계산"""
        midpoint = (self.classifiers[class1] + self.classifiers[class2]) / 2
        return midpoint

    def compute_soft_label(self, lam, gamma):
        """MICS 소프트 라벨 정책"""
        # 원본 클래스 확률
        prob_class1 = max((1 - gamma - lam) / (1 - gamma), 0)
        prob_class2 = max((lam - gamma) / (1 - gamma), 0)

        # 가상 클래스 확률
        prob_virtual = 1 - prob_class1 - prob_class2

        return prob_class1, prob_virtual, prob_class2

    def manifold_mixup(self, x1, x2, y1, y2, session_idx):
        """
        Manifold Mixup 수행
        x1, x2: 입력 이미지 샘플
        y1, y2: 클래스 레이블
        """
        # 베타 분포에서 혼합 비율 샘플링
        alpha = self.config.alpha
        lam = np.random.beta(alpha, alpha)

        # 히든 레이어의 특성 추출
        # 실제 구현에서는 백본 모델을 두 부분으로 나누어야 함
        # 여기서는 간략화를 위해 입력 레벨 믹스업으로 대체

        # 모션 인식 적용 여부에 따라 처리
        if self.config.use_motion and len(x1.shape) == 5:  # 비디오 데이터
            mixed_x, adjusted_lam = self.motion_mixup(x1, x2, lam)
            lam = adjusted_lam
        else:
            # 기본 이미지 믹스업
            mixed_x = lam * x1 + (1 - lam) * x2

        # 소프트 라벨 계산
        gamma = self.config.gamma
        prob_1, prob_v, prob_2 = self.compute_soft_label(lam, gamma)

        # 클래스 쌍에 대한 가상 클래스 인덱스 생성/검색
        class_pair = (int(y1.item()), int(y2.item()))
        if class_pair not in self.virtual_class_indices:
            # 새 가상 클래스 인덱스 생성
            virtual_idx = len(self.classifiers)
            self.virtual_class_indices[class_pair] = virtual_idx

            # 중간점 분류기 계산
            midpoint = self.compute_midpoint_classifier(y1, y2)

            # 분류기 확장
            new_classifiers = nn.Parameter(
                torch.cat([self.classifiers.data, midpoint.unsqueeze(0)], dim=0)
            ).to(self.device)

            self.classifiers = new_classifiers

        virtual_idx = self.virtual_class_indices[class_pair]

        # 소프트 라벨 생성
        soft_label = torch.zeros(len(self.classifiers), device=self.device)
        soft_label[y1] = prob_1
        soft_label[y2] = prob_2
        soft_label[virtual_idx] = prob_v

        return mixed_x, soft_label

    def cleanup_virtual_classifiers(self, num_real_classes):
        """증분 학습 후 가상 분류기 제거"""
        # 실제 클래스에 대한 분류기만 유지
        self.classifiers = nn.Parameter(self.classifiers[:num_real_classes].clone())
        self.virtual_class_indices = {}  # 가상 클래스 인덱스 초기화