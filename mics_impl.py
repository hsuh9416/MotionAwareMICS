# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from feature_extractor import ResNet20Backbone, ResNet18Backbone


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

        # Normalize classifiers at initialization
        with torch.no_grad():
            self.classifiers.data = F.normalize(self.classifiers.data, p=2, dim=1)

        # No need for virtual class index tracking in the simplified approach
        self.num_real_classes = num_classes

        # Motion-aware mixup module
        self.motion_mixup = MotionAwareMixup(config).to(self.device)

    def forward(self, x, return_features=False):
        # Handle video data (UCF101)
        if len(x.shape) == 5:  # [B, C, T, H, W]
            B, C, T, H, W = x.shape

            # Process each frame
            frame_features = []
            for t in range(T):
                frame = x[:, :, t]  # [B, C, H, W]
                frame_feat = self.backbone.features(frame)
                frame_features.append(frame_feat)

            # Average features across time
            features = torch.stack(frame_features, dim=2)
            features = torch.mean(features, dim=2)
            features = torch.flatten(features, 1)
        else:
            # Standard image processing
            features = self.backbone(x)

        if return_features:
            return features

        # Cosine similarity based classification
        logits = F.linear(F.normalize(features, p=2, dim=1),
                          F.normalize(self.classifiers, p=2, dim=1))
        return logits / self.config.temperature  # temperature scaling

    def expand_classifier(self, novel_classes):
        """Properly expand classifier for new classes"""
        # Initialize new classifiers with proper normalization
        new_class_vectors = torch.randn(novel_classes, self.config.feature_dim).to(self.device)
        new_class_vectors = F.normalize(new_class_vectors, p=2, dim=1)

        # Concatenate with existing classifiers
        expanded_classifiers = nn.Parameter(
            torch.cat([self.classifiers.data, new_class_vectors], dim=0)
        )

        self.classifiers = expanded_classifiers
        self.num_real_classes += novel_classes

    def apply_mixup(self, x1, x2, lam):
        """Apply basic mixup to input data"""
        mixed = lam * x1 + (1 - lam) * x2
        return mixed

    def compute_soft_label(self, lam, y1, y2, num_classes):
        """Create soft labels for mixup"""
        soft_label = torch.zeros(num_classes, device=self.device)
        soft_label[y1] = lam
        soft_label[y2] = 1 - lam
        return soft_label


# Motion-aware mixup implementation
class MotionAwareMixup(nn.Module):
    def __init__(self, config):
        super(MotionAwareMixup, self).__init__()
        self.config = config
        self.flow_alpha = config.flow_alpha

    def compute_motion_consistency(self, flow1, flow2):
        """Compute motion consistency between two optical flows"""
        # Reshape flows for computation
        flow1_flat = flow1.reshape(2, -1)
        flow2_flat = flow2.reshape(2, -1)

        # Calculate magnitudes
        norm1 = torch.norm(flow1_flat, dim=1, keepdim=True)
        norm2 = torch.norm(flow2_flat, dim=1, keepdim=True)

        # Prevent division by zero
        epsilon = 1e-8

        # Cosine similarity
        cos_sim = torch.sum(flow1_flat * flow2_flat, dim=1) / (norm1 * norm2 + epsilon)

        # Average similarity
        consistency = torch.mean(cos_sim)
        return consistency.item()

    def forward(self, frames1, frames2, lam):
        """Mix two video sequences with motion awareness"""
        # Skip motion awareness if disabled
        if not self.config.use_motion:
            mixed_frames = lam * frames1 + (1 - lam) * frames2
            return mixed_frames, lam

        try:
            # Compute optical flow
            flow1 = compute_optical_flow(frames1)
            flow2 = compute_optical_flow(frames2)

            # Compute motion consistency
            consistency = self.compute_motion_consistency(flow1[0], flow2[0])

            # Adjust lambda based on motion consistency
            # Higher consistency -> closer to 0.5 (balanced mixing)
            adjusted_lam = lam + self.flow_alpha * (0.5 - lam) * consistency

            # Clamp to valid range
            adjusted_lam = max(0.0, min(1.0, adjusted_lam))
        except Exception as e:
            print(f"Error in motion processing: {e}")
            adjusted_lam = lam  # Fallback to original lambda

        # Mix frames with adjusted lambda
        mixed_frames = adjusted_lam * frames1 + (1 - adjusted_lam) * frames2

        return mixed_frames, adjusted_lam


# Optical flow computation function
def compute_optical_flow(frames):
    """Compute optical flow between consecutive frames"""
    B, C, T, H, W = frames.shape
    flows = torch.zeros(B, 2, T - 1, H, W, device=frames.device)

    # Process each batch and time step
    for b in range(B):
        for t in range(T - 1):
            try:
                # Convert to NumPy for OpenCV
                prev_frame = frames[b, :, t].permute(1, 2, 0).detach().cpu().numpy()
                next_frame = frames[b, :, t + 1].permute(1, 2, 0).detach().cpu().numpy()

                # Ensure valid range for OpenCV
                prev_frame = np.clip(prev_frame, 0, 1)
                next_frame = np.clip(next_frame, 0, 1)

                # Convert to uint8 format
                prev_frame = (prev_frame * 255).astype(np.uint8)
                next_frame = (next_frame * 255).astype(np.uint8)

                # Convert to grayscale
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

                # Compute flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, flow=None, pyr_scale=0.5, levels=3,
                                                    winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

                # Store flow
                flows[b, 0, t] = torch.from_numpy(flow[:, :, 0]).to(frames.device)
                flows[b, 1, t] = torch.from_numpy(flow[:, :, 1]).to(frames.device)

            except Exception as e:
                print(f"Error computing optical flow: {e}")

    return flows