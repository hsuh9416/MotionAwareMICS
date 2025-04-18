# Import neccesary libraries
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import cv2

# 광학 흐름 계산 함수
def compute_optical_flow(frames):
    """
    비디오 프레임에서 광학 흐름 계산
    입력: [B, C, T, H, W] 텐서
    출력: [B, 2, T-1, H, W] 광학 흐름 텐서 (x, y 방향)
    """
    B, C, T, H, W = frames.shape
    flows = torch.zeros(B, 2, T-1, H, W, device=frames.device)

    # 배치의 각 항목에 대해 처리
    for b in range(B):
        for t in range(T-1):
            # NumPy 배열로 변환
            prev_frame = frames[b, :, t].permute(1, 2, 0).cpu().numpy()
            next_frame = frames[b, :, t+1].permute(1, 2, 0).cpu().numpy()

            # 그레이스케일로 변환
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

            # 광학 흐름 계산 (Farneback 방법)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                               0.5, 3, 15, 3, 5, 1.2, 0)

            # x, y 방향 흐름을 텐서로 변환
            flows[b, 0, t] = torch.from_numpy(flow[:, :, 0]).to(frames.device)
            flows[b, 1, t] = torch.from_numpy(flow[:, :, 1]).to(frames.device)

    return flows


# 모션 인식 모듈
class MotionAwareMixup(nn.Module):
    def __init__(self, config):
        super(MotionAwareMixup, self).__init__()
        self.config = config
        self.flow_alpha = config.flow_alpha

    def compute_motion_consistency(self, flow1, flow2, lam):
        """
        두 샘플의 광학 흐름 일관성 계산
        flow1, flow2: [2, T-1, H, W] 광학 흐름 텐서
        lam: 혼합 비율
        """
        # 두 흐름의 코사인 유사도 계산
        flow1_flat = flow1.reshape(2, -1)
        flow2_flat = flow2.reshape(2, -1)

        norm1 = torch.norm(flow1_flat, dim=1, keepdim=True)
        norm2 = torch.norm(flow2_flat, dim=1, keepdim=True)

        # 제로 나누기 방지
        epsilon = 1e-8
        cos_sim = torch.sum(flow1_flat * flow2_flat, dim=1) / (norm1 * norm2 + epsilon)

        # 일관성 점수: 유사도가 높을수록 lam을 0.5에 가깝게 조정
        consistency = torch.mean(cos_sim)
        adjusted_lam = lam + self.flow_alpha * (0.5 - lam) * consistency

        # [0, 1] 범위로 클리핑
        adjusted_lam = torch.clamp(adjusted_lam, 0.0, 1.0)

        return adjusted_lam.item()

    def forward(self, frames1, frames2, lam):
        """
        모션 인식 믹스업 수행
        frames1, frames2: [B, C, T, H, W] 비디오 프레임
        lam: 기본 혼합 비율
        """
        if not self.config.use_motion:
            # 모션 인식 비활성화 시 기본 믹스업 수행
            mixed_frames = lam * frames1 + (1 - lam) * frames2
            return mixed_frames, lam

        # 광학 흐름 계산
        flow1 = compute_optical_flow(frames1)
        flow2 = compute_optical_flow(frames2)

        # 모션 일관성 기반 혼합 비율 조정
        adjusted_lam = self.compute_motion_consistency(flow1[0], flow2[0], lam)

        # 조정된 비율로 믹스업
        mixed_frames = adjusted_lam * frames1 + (1 - adjusted_lam) * frames2

        return mixed_frames, adjusted_lam

