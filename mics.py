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
        elif args.dataset == 'hmdb51':
            self.encoder = resnet18(num_classes=args.num_classes)

        # Initialize motion-aware mixup module if needed
        if args.use_motion and args.dataset == 'hmdb51':
            self.motion_mixup = MotionAwareMixup(args)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Init classifier
        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.args.num_features, self.pre_allocate, bias=False)
        nn.init.orthogonal_(self.fc.weight)

    def encode(self, x):
        """Extract features from input data"""
        # Handle video data(HMDB51)
        if len(x.shape) == 5:  # [B, C, T, H, W]
            B, C, T, H, W = x.shape

            # Process each frame separately and average
            frame_features = []
            for t in range(T):
                frame = x[:, :, t]  # [B, C, H, W]
                frame_features.append(self.encoder(frame))

            # Average features across frames
            x = torch.stack(frame_features, dim=2)
            x = torch.mean(x, dim=2)
        else:
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
        # Handle video data
        if len(x.shape) == 5:  # [B, C, T, H, W]
            B, C, T, H, W = x.shape

            # Process each frame separately and average
            frame_features = []
            for t in range(T):
                frame = x[:, :, t]  # [B, C, H, W]
                frame_features.append(self.encode(frame))

            x = torch.stack(frame_features, dim=1).mean(dim=1)
        else:
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

    def forward_with_mixup(self, args, x, labels=None):
        """Main forward function that handles both image and video data with mixup"""
        # For video data with motion awareness enabled
        if len(x.shape) == 5 and args.use_motion and args.dataset == 'hmdb51':
            return self.motion_mixup.forward_motion_mixup(self, x, labels, args)
        else:
            # Regular image mixup
            return self.forward_mix_up(args, x, labels)


# Motion-aware mixup implementation
def compute_motion_consistency(flow1, flow2):
    """Compute motion consistency between two optical flows"""
    # Reshape flows for computation
    flow1_flat = flow1.reshape(2, -1)
    flow2_flat = flow2.reshape(2, -1)

    # Calculate magnitudes
    norm1 = torch.norm(flow1_flat, dim=1, keepdim=True)
    norm2 = torch.norm(flow2_flat, dim=1, keepdim=True)

    # Prevent division by zero
    epsilon = 1e-8

    # Compute cosine similarity between flow vectors
    cos_sim = torch.sum(flow1_flat * flow2_flat, dim=1) / (norm1 * norm2 + epsilon)

    # Return average similarity across all pixels
    consistency = torch.mean(cos_sim)
    return consistency.item()


class MotionAwareMixup(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.flow_alpha = args.flow_alpha  # Optical flow weighting factor

    def compute_optical_flow(self, frames):
        """
        Compute optical flow between consecutive frames using Farneback method

        """
        if len(frames) < 2:
            # Return zero flow if not enough frames
            H, W = frames[0].shape[:2]
            return torch.zeros(2, 1, H, W)

        flows = []
        # Process each consecutive pair of frames
        for t in range(len(frames) - 1):
            # Convert to grayscale for optical flow
            prev_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[t + 1], cv2.COLOR_RGB2GRAY)

            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            flows.append(flow)

        # Convert to tensor [2, T-1, H, W]
        flow_tensor = torch.tensor(np.array(flows).transpose(1, 0, 2, 3), dtype=torch.float32)
        return flow_tensor

    def forward(self, x1, x2, lam):
        """
        Mix two video sequences with motion awareness
        """
        # Skip motion awareness if disabled
        if not self.args.use_motion:
            mixed_x = lam * x1 + (1 - lam) * x2
            return mixed_x, lam

        try:
            # Extract frames for optical flow computation
            frames1 = self.process_video_data(x1)
            frames2 = self.process_video_data(x2)

            # Compute batch-wise motion consistency
            batch_size = x1.size(0)
            consistencies = []

            for b in range(batch_size):
                # Compute optical flow for each video
                flow1 = self.compute_optical_flow(frames1[b])
                flow2 = self.compute_optical_flow(frames2[b])

                # Compute motion consistency
                if flow1.shape[1] > 0 and flow2.shape[1] > 0:
                    consistency = compute_motion_consistency(flow1[:, 0], flow2[:, 0])
                    consistencies.append(consistency)
                else:
                    consistencies.append(0.0)

            # Average consistency across batch
            avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0.0

            # Adjust lambda based on motion consistency
            # Higher consistency -> closer to 0.5 (more balanced mixing)
            adjusted_lam = lam + self.flow_alpha * (0.5 - lam) * avg_consistency

            # Ensure lambda stays in valid range
            adjusted_lam = max(0.0, min(1.0, adjusted_lam))

            # Create batch-specific lambdas if needed
            if isinstance(adjusted_lam, float):
                adjusted_lam = torch.tensor([adjusted_lam], device=x1.device)

        except Exception as e:
            print(f"Error in motion processing: {e}")
            # Fall back to original lambda on error
            adjusted_lam = lam

        # Mix videos with adjusted lambda
        mixed_x = adjusted_lam * x1 + (1 - adjusted_lam) * x2
        return mixed_x, adjusted_lam

    def forward_motion_mixup(self, model, x, labels, args):
        """
        Applies motion-aware mixup to video data and computes the model output
        """
        # Only apply motion-aware mixup to video data
        if len(x.shape) != 5:  # Not video data
            return model.forward_mix_up(args, x, labels)

        # Get current number of classes for label processing
        cur_num_class = int(max(labels)) + 1

        # Sample lambda from beta distribution
        lamb = np.random.beta(args.alpha, args.alpha) if args.alpha > 0 else 1
        lam_tensor = torch.tensor([lamb], dtype=torch.float32).cuda()

        # Generate permutation indices for mixing
        indices = np.random.permutation(x.size(0))

        # Create one-hot encoded labels
        target_reweighted = to_one_hot(labels, model.args.num_classes)

        # Apply motion-aware mixup to input videos
        mixed_x, adjusted_lam = self.forward(x, x[indices], lam_tensor)

        # Process mixed videos through the feature extractor
        features = model.encoder(mixed_x)

        # Create midpoint labels between original and mixed classes
        retarget, mix_label_mask = middle_label_mix_process(
            target_reweighted,
            target_reweighted[indices],
            cur_num_class,
            adjusted_lam,
            args.gamma
        )

        # Apply global average pooling to features
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.squeeze(-1).squeeze(-1)

        # Calculate midpoint classifier
        classifier = model.calculate_middle_classifier(mix_label_mask)

        # Compute cosine similarity with normalized features and classifiers
        outputs = F.linear(
            F.normalize(features, p=2, dim=-1),
            F.normalize(classifier, p=2, dim=-1)
        )

        # Apply temperature scaling
        outputs = args.temperature * outputs

        return outputs, retarget
