# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from resnet import resnet18, resnet20
from mix_up import to_one_hot, middle_label_mix_process

class MICS(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        # Feature Extractor
        if args.dataset == 'cifar100':
            self.encoder = resnet20(num_classes=args.num_classes)
        else:  # Motion dataset 'ucf101'
            self.encoder = resnet18(num_classes=args.num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Init classifier
        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.args.num_features, self.pre_allocate, bias=False)
        nn.init.orthogonal_(self.fc.weight)

    def encode(self, x):
        """ Encoding: x(Tensor): [B, C, H, W] -> [B, C', H', W'] -> [B, C', 1, 1] -> [B, C']"""
        x = self.encoder(x)  # Apply feature extractor to extract a feature map
        x = F.adaptive_avg_pool2d(x, 1)  # Reduced to (1,1) spatial size by averaging each channel
        x = x.squeeze(-1).squeeze(-1)  # 4D -> 2D
        return x  # Embedded feature vector

    def get_logits(self, x, fc):
        """A function that calculates the raw score (logit) to be used as a classification result."""
        # Cosine similarity computation
        x = F.linear(F.normalize(x, p=2, dim=1), F.normalize(fc, p=2, dim=1))

        # Temperature scaling is a hyperparameter applied to the softmax function and is used to adjust the output
        # distribution of the model. The lower the temperature, the higher the confidence of the classification. T =
        # 0.1, MICS has the concept of Boundary Thickness, so it has a relatively small value.
        x = x * self.args.temperature

        return x

    def calculate_middle_classifier(self, mix_label_mask):
        """Calculate weights for virtual classifiers based on mix-up labels"""
        first = self.fc.weight[mix_label_mask[0]]
        second = self.fc.weight[mix_label_mask[1]]

        # Paper 4.2 Establishment of Midpoint Classifiers - Formula (4)
        mid = (first + second) / 2  # e.g. ([1, 0, 0] + [0, 1, 0]) / 2 -> [0.5, 0.5, 0]
        return mid

    def forward(self, x):
        x = self.encode(x)
        if self.mode != 'encoder':  # classification
            # Cosine similarity
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            # Scale temperature
            x = self.args.temperature * x
        return x

    def forward_mix_up(self, args, x, labels=None):
        """Forward propagation with Mix-Up"""
        cur_num_class = int(max(labels)) + 1  # Biggest class index + 1

        # Handle video data (UCF101)
        if len(x.shape) == 5:  # [B, C, T, H, W]
            B, C, T, H, W = x.shape

            # Process each frame
            frame_features = []
            for t in range(T):
                frame = x[:, :, t]  # [B, C, H, W]
                frame_feat = self.encoder.features(frame)
                frame_features.append(frame_feat)

            # Average features across time
            features = torch.stack(frame_features, dim=2)
            features = torch.mean(features, dim=2)
            features = torch.flatten(features, 1)

        # Feature extraction and Mix-up processing via encoder
        x, new_labels, mix_label_mask = self.encoder(x,
                                                     labels=labels,
                                                     mix_type=args.train,
                                                     mixup_alpha=args.alpha,
                                                     num_base_classes=cur_num_class,
                                                     gamma=args.gamma)

        # Embedding
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)

        # Calculate middle classifier
        classifier = self.calculate_middle_classifier(
            mix_label_mask)  # e.g. (dog, bird) : ([1, 0, 0], [0, 1, 0]) => (virtual) : [0.5, 0.5, 0]

        # Cosine similarity
        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(classifier, p=2, dim=-1))

        # Temperature scaling is a hyperparameter applied to the softmax function and is used to adjust the output
        # distribution of the model. The lower the temperature, the higher the confidence of the classification. T =
        # 0.1, MICS has the concept of Boundary Thickness, so it has a relatively small value.
        x = args.temperature * x

        return x, new_labels  # images, re-targeted labels


# TODO should fix below functions!

# Motion-aware mixup implementation
class MotionAwareMixup(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.flow_alpha = args.flow_alpha

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
        if not self.args.use_motion:
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
            adjusted_lam = lam  # Fallback to the original lambda

        # Mix frames with adjusted lambda
        mixed_frames = adjusted_lam * frames1 + (1 - adjusted_lam) * frames2

        return mixed_frames, adjusted_lam


    def forward_mix_up_motion(self, args, x, labels=None):
        """Forward propagation with Motion-Aware Mix-Up for video data"""
        cur_num_class = int(max(labels)) + 1

        # Motion-aware processing for video data
        if len(x.shape) == 5:  # [B, C, T, H, W] - Video data
            # Initialize motion-aware mixup module
            motion_mixup = MotionAwareMixup(args)

            # Compute lambda = beta distribution with mixup_alpha
            lamb = np.random.beta(args.alpha, args.alpha) if args.alpha > 0 else 1
            lam_tensor = torch.from_numpy(np.array([lamb]).astype('float32')).cuda()

            # Get batch indices for mixing
            indices = np.random.permutation(x.size(0))

            # Create one-hot labels
            target_reweighted = to_one_hot(labels, self.num_classes)

            # Apply motion-aware mixup
            x_mixed, adjusted_lam = motion_mixup(x, x[indices], lam_tensor)

            # Extract features
            x_mixed = self.encoder(x_mixed)

            # Create midpoint labels and virtual classifiers
            retarget, mix_label_mask = middle_label_mix_process(
                target_reweighted, target_reweighted[indices],
                cur_num_class, adjusted_lam, args.gamma
            )

            # Calculate middle classifier
            classifier = self.calculate_middle_classifier(mix_label_mask)

            # Compute logits
            x_mixed = F.adaptive_avg_pool2d(x_mixed, 1)
            x_mixed = x_mixed.squeeze(-1).squeeze(-1)
            x_mixed = F.linear(F.normalize(x_mixed, p=2, dim=-1), F.normalize(classifier, p=2, dim=-1))

            # Apply temperature scaling
            output = args.temperature * x_mixed

            return output, retarget
        else:
            # Fall back to standard mixup for image data
            return self.forward_mix_up(args, x, labels)

# Optical flow computation function
def compute_optical_flow(frames):
    """
    Compute optical flow between consecutive frames
    Args:
        frames: tensor of shape [B, C, T, H, W]
    Returns:
        flows: tensor of shape [B, 2, T-1, H, W]
    """
    B, C, T, H, W = frames.shape
    flows = torch.zeros(B, 2, T - 1, H, W, device=frames.device)

    # Process each batch and time step
    for b in range(B):
        prev_frame = None

        for t in range(T - 1):
            # Convert current frames to numpy for OpenCV
            if prev_frame is None:
                prev_frame = frames[b, :, t].permute(1, 2, 0).detach().cpu().numpy()
                prev_frame = (prev_frame * 255).astype(np.uint8)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

            curr_frame = frames[b, :, t + 1].permute(1, 2, 0).detach().cpu().numpy()
            curr_frame = (curr_frame * 255).astype(np.uint8)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Store flow in tensor
            flows[b, 0, t] = torch.from_numpy(flow[:, :, 0]).to(frames.device)
            flows[b, 1, t] = torch.from_numpy(flow[:, :, 1]).to(frames.device)

            # Update previous frame
            prev_frame = curr_frame
            prev_gray = curr_gray

    return flows
