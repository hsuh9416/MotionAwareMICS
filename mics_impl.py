# Import necessary libraries
import torch.nn.functional as F
import numpy as np
import cv2

from feature_extractor import *

# Implemented MICS Model
class MICS(nn.Module):
    def __init__(self, config, num_classes):
        super(MICS, self).__init__()
        self.config = config
        self.device = config.device

        # Set the Feature Extractor
        if config.dataset == 'cifar100':
            self.backbone = ResNet20Backbone(config.feature_dim).to(self.device)
        else:  # Motion dataset 'ucf101'
            self.backbone = ResNet18Backbone(config.feature_dim).to(self.device)

        # class classifier
        self.classifiers = nn.Parameter(torch.randn(num_classes, config.feature_dim).to(self.device))

        # motion recognition module
        self.motion_mixup = MotionAwareMixup(config).to(self.device)

        # Virtual class index tracking
        self.virtual_class_indices = {}

    def forward(self, x, return_features=False):
        features = self.backbone(x)

        if return_features:
            return features

        # Cosine similarity based classification
        logits = F.linear(F.normalize(features, p=2, dim=1),
                         F.normalize(self.classifiers, p=2, dim=1))
        return logits / 0.1  # temperature scaling

    def compute_midpoint_classifier(self, class1, class2):
        # Compute midpoint
        midpoint = (self.classifiers[class1] + self.classifiers[class2]) / 2
        return midpoint

    def compute_soft_label(self, lam, gamma):
        # soft label policy
        # Original class probability
        prob_class1 = max((1 - gamma - lam) / (1 - gamma), 0)
        prob_class2 = max((lam - gamma) / (1 - gamma), 0)

        # Virtual class probability
        prob_virtual = 1 - prob_class1 - prob_class2

        return prob_class1, prob_virtual, prob_class2

    def manifold_mixup(self, x1, x2, y1, y2, session_idx):
        """
        Manifold Mixup 수행 - 모델의 중간 레이어에서 특성을 혼합
        x1, x2: 입력 이미지 샘플
        y1, y2: 클래스 레이블
        """
        # 베타 분포에서 혼합 비율 샘플링
        alpha = self.config.alpha
        lam = np.random.beta(alpha, alpha)

        # 중간 레이어 선택 (ResNet의 경우 layer1, layer2, layer3, layer4 중 하나)
        # ResNet20 또는 ResNet18의 내부 구조에 따라 접근 방식이 달라짐
        if isinstance(self.backbone, ResNet20Backbone):
            # ResNet20의 경우 내부 구조에 접근
            # features는 Sequential이므로 layer별로 분리
            layers = list(self.backbone.features.children())

            # 첫 번째 부분 (입력 ~ 선택한 레이어까지)
            layer_idx = np.random.randint(1, len(layers) - 1)  # 첫 레이어와 마지막 레이어 제외
            first_part = nn.Sequential(*layers[:layer_idx])

            # 두 번째 부분 (선택한 레이어 이후 ~ 출력)
            second_part = nn.Sequential(*layers[layer_idx:])

            # 첫 번째 부분을 통과하여 중간 레이어 특성 추출
            h1 = first_part(x1)
            h2 = first_part(x2)

            # 중간 레이어에서 특성 혼합
            h_mixed = lam * h1 + (1 - lam) * h2

            # 두 번째 부분을 통과시켜 최종 특성 추출
            mixed_features = second_part(h_mixed)
            mixed_features = torch.flatten(mixed_features, 1)

        elif isinstance(self.backbone, ResNet18Backbone) and len(x1.shape) == 5:
            # 비디오 데이터의 경우 (ResNet18 with motion data)
            B, C, T, H, W = x1.shape

            # 각 프레임에 대해 별도로 처리
            mixed_frame_features = []

            # ResNet18의 레이어 분리
            resnet_layers = list(self.backbone.features.children())
            layer_idx = np.random.randint(1, len(resnet_layers) - 1)
            first_part = nn.Sequential(*resnet_layers[:layer_idx])
            second_part = nn.Sequential(*resnet_layers[layer_idx:])

            for t in range(T):
                # 각 프레임 추출
                frame1 = x1[:, :, t]  # [B, C, H, W]
                frame2 = x2[:, :, t]  # [B, C, H, W]

                # 첫 번째 부분 통과
                h1 = first_part(frame1)
                h2 = first_part(frame2)

                # 모션 인식이 활성화된 경우 광학 흐름 정보를 활용하여 혼합 비율 조정
                if self.config.use_motion and t > 0:
                    # t와 t-1 프레임 간의 광학 흐름 근사
                    # 간단히 하기 위해 현재 프레임 차이를 사용
                    frame_diff1 = torch.mean(torch.abs(x1[:, :, t] - x1[:, :, t - 1]))
                    frame_diff2 = torch.mean(torch.abs(x2[:, :, t] - x2[:, :, t - 1]))

                    # 모션의 정도에 따라 혼합 비율 조정
                    motion_factor = torch.sigmoid(frame_diff1 - frame_diff2) * self.config.flow_alpha
                    adjusted_lam = lam + motion_factor * (0.5 - lam)
                    adjusted_lam = torch.clamp(adjusted_lam, 0.0, 1.0).item()
                else:
                    adjusted_lam = lam

                # 중간 레이어에서 특성 혼합
                h_mixed = adjusted_lam * h1 + (1 - adjusted_lam) * h2

                # 두 번째 부분 통과
                frame_features = second_part(h_mixed)
                mixed_frame_features.append(frame_features)

            # 모든 프레임 특성을 스택
            stacked_features = torch.stack(mixed_frame_features, dim=2)  # [B, feat_dim, T, 1, 1]

            # 시간 차원에 대해 평균
            mixed_features = torch.mean(stacked_features, dim=2)  # [B, feat_dim, 1, 1]
            mixed_features = torch.flatten(mixed_features, 1)  # [B, feat_dim]

        else:
            # 일반 이미지 데이터 (ResNet18)
            resnet_layers = list(self.backbone.features.children())
            layer_idx = np.random.randint(1, len(resnet_layers) - 1)
            first_part = nn.Sequential(*resnet_layers[:layer_idx])
            second_part = nn.Sequential(*resnet_layers[layer_idx:])

            # 첫 번째 부분 통과
            h1 = first_part(x1)
            h2 = first_part(x2)

            # 중간 레이어에서 특성 혼합
            h_mixed = lam * h1 + (1 - lam) * h2

            # 두 번째 부분 통과
            mixed_features = second_part(h_mixed)
            mixed_features = torch.flatten(mixed_features, 1)

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

        return mixed_features, soft_label

    def cleanup_virtual_classifiers(self, num_real_classes):
        """증분 학습 후 가상 분류기 제거"""
        # 실제 클래스에 대한 분류기만 유지
        self.classifiers = nn.Parameter(self.classifiers[:num_real_classes].clone())
        self.virtual_class_indices = {}  # 가상 클래스 인덱스 초기화
    
# Optical flow function
def compute_optical_flow(frames):
    # Input
    B, C, T, H, W = frames.shape
    flows = torch.zeros(B, 2, T-1, H, W, device=frames.device)

    # Process each item in the batch
    for b in range(B):
        for t in range(T-1):
            # Convert to NumPy array
            prev_frame = frames[b, :, t].permute(1, 2, 0).cpu().numpy()
            next_frame = frames[b, :, t+1].permute(1, 2, 0).cpu().numpy()

            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

            # Compute optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,0.5, 3, 15, 3, 5, 1.2, 0)

            # Convert x, y direction flow to tensor
            flows[b, 0, t] = torch.from_numpy(flow[:, :, 0]).to(frames.device)
            flows[b, 1, t] = torch.from_numpy(flow[:, :, 1]).to(frames.device)

    return flows

# Mix-up for Motion awareness
class MotionAwareMixup(nn.Module):
    def __init__(self, config):
        super(MotionAwareMixup, self).__init__()
        self.config = config
        self.flow_alpha = config.flow_alpha

    def compute_motion_consistency(self, flow1, flow2, lam):
        # Computing optical flow coherence of two samples
        # Calculating cosine similarity of two streams
        flow1_flat = flow1.reshape(2, -1)
        flow2_flat = flow2.reshape(2, -1)

        norm1 = torch.norm(flow1_flat, dim=1, keepdim=True)
        norm2 = torch.norm(flow2_flat, dim=1, keepdim=True)

        # Prevent division by zero
        epsilon = 1e-8
        cos_sim = torch.sum(flow1_flat * flow2_flat, dim=1) / (norm1 * norm2 + epsilon)

        # Consistency score: The higher the similarity, the closer the lam is to 0.5
        consistency = torch.mean(cos_sim)
        adjusted_lam = lam + self.flow_alpha * (0.5 - lam) * consistency

        # Clipping to the range [0, 1]
        adjusted_lam = torch.clamp(adjusted_lam, 0.0, 1.0)

        return adjusted_lam.item()

    def forward(self, frames1, frames2, lam):
        # Conduct motion aware mix-up
        if not self.config.use_motion:
            # Perform basic mix-up when motion detection is disabled
            mixed_frames = lam * frames1 + (1 - lam) * frames2
            return mixed_frames, lam

        # Optical flow calculation
        flow1 = compute_optical_flow(frames1)
        flow2 = compute_optical_flow(frames2)

        # Motion consistency based blend ratio adjustment
        adjusted_lam = self.compute_motion_consistency(flow1[0], flow2[0], lam)

        # Mix up with adjusted ratio
        mixed_frames = adjusted_lam * frames1 + (1 - adjusted_lam) * frames2

        return mixed_frames, adjusted_lam