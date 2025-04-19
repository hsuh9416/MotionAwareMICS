# Updated MICS Model implementation
# The main issue is in how the model handles classifiers in incremental sessions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class MICS(nn.Module):
    def __init__(self, config, num_classes):
        super(MICS, self).__init__()
        self.config = config
        self.device = config.device

        # Set the Feature Extractor
        if config.dataset == 'cifar100':
            from feature_extractor import ResNet20Backbone
            self.backbone = ResNet20Backbone(config.feature_dim).to(self.device)
        else:  # Motion dataset 'ucf101'
            from feature_extractor import ResNet18Backbone
            self.backbone = ResNet18Backbone(config.feature_dim).to(self.device)

        # class classifier
        self.classifiers = nn.Parameter(torch.randn(num_classes, config.feature_dim).to(self.device))

        # Normalize classifiers at initialization
        with torch.no_grad():
            self.classifiers.data = F.normalize(self.classifiers.data, p=2, dim=1)

        # motion recognition module
        self.motion_mixup = MotionAwareMixup(config).to(self.device)

        # Virtual class index tracking
        self.virtual_class_indices = {}

        # Keep track of real classes vs virtual classes
        self.num_real_classes = num_classes

    def forward(self, x, return_features=False):
        features = self.backbone(x)

        if return_features:
            return features

        # Cosine similarity based classification
        logits = F.linear(F.normalize(features, p=2, dim=1),
                          F.normalize(self.classifiers, p=2, dim=1))
        return logits / self.config.temperature  # temperature scaling

    def compute_midpoint_classifier(self, class1, class2):
        """Calculate midpoint classifier"""
        # Convert tensor inputs to integers if needed
        if isinstance(class1, torch.Tensor):
            class1 = class1.item()
        if isinstance(class2, torch.Tensor):
            class2 = class2.item()

        # Calculate midpoint
        midpoint = (self.classifiers[class1] + self.classifiers[class2]) / 2

        # Normalize the midpoint
        midpoint = F.normalize(midpoint.unsqueeze(0), p=2, dim=1).squeeze(0)

        # Ensure proper dimensionality
        if len(midpoint.shape) > 1:
            if midpoint.shape[0] != 1:
                midpoint = midpoint.unsqueeze(0)
        else:
            midpoint = midpoint.unsqueeze(0)

        return midpoint

    def compute_soft_label(self, lam, gamma):
        # Soft label policy
        # Original class probability
        prob_class1 = max((1 - gamma - lam) / (1 - gamma), 0)
        prob_class2 = max((lam - gamma) / (1 - gamma), 0)

        # Virtual class probability
        prob_virtual = 1 - prob_class1 - prob_class2

        return prob_class1, prob_virtual, prob_class2

    def manifold_mixup(self, x1, x2, y1, y2, session_idx, total_classes=None):
        """
        Perform Manifold Mixup - mix features in the middle layers
        x1, x2: input image samples
        y1, y2: class labels
        total_classes: fixed size for soft labels (to ensure consistent tensor sizes)
        """
        # Sample mixing ratio from beta distribution
        alpha = self.config.alpha
        lam = np.random.beta(alpha, alpha)

        # Get integer class indices
        y1_int = int(y1.item())
        y2_int = int(y2.item())

        # Class pair (ordered for consistency)
        class_pair = (min(y1_int, y2_int), max(y1_int, y2_int))

        # Create virtual class if needed
        if class_pair not in self.virtual_class_indices:
            # New virtual class index
            virtual_idx = len(self.classifiers)
            self.virtual_class_indices[class_pair] = virtual_idx

            # Calculate midpoint classifier
            midpoint = self.compute_midpoint_classifier(y1_int, y2_int)

            # Expand classifiers
            new_classifiers = nn.Parameter(
                torch.cat([self.classifiers.data, midpoint], dim=0)
            ).to(self.device)

            self.classifiers = new_classifiers

        virtual_idx = self.virtual_class_indices[class_pair]

        # Apply mixup based on model type
        if hasattr(self.backbone, 'features'):
            # For ResNet20/ResNet18 with features attribute
            layers = list(self.backbone.features.children())
            layer_idx = np.random.randint(1, len(layers) - 1)

            # Forward to middle layer
            h1 = x1
            h2 = x2
            for j in range(layer_idx):
                h1 = layers[j](h1)
                h2 = layers[j](h2)

            # Apply motion-aware mixup if enabled (for video data)
            if self.config.use_motion and len(x1.shape) == 5:
                # Video-specific processing
                adjusted_lam = lam
                # Motion awareness code would go here
            else:
                adjusted_lam = lam

            # Mix features
            h_mixed = adjusted_lam * h1 + (1 - adjusted_lam) * h2

            # Continue forward pass
            for j in range(layer_idx, len(layers)):
                h_mixed = layers[j](h_mixed)

            mixed_features = torch.flatten(h_mixed, 1)
        else:
            # Fallback to standard feature extraction
            with torch.no_grad():
                feat1 = self.backbone(x1)
                feat2 = self.backbone(x2)

            mixed_features = lam * feat1 + (1 - lam) * feat2

        # Calculate soft labels
        gamma = self.config.gamma
        prob_1, prob_v, prob_2 = self.compute_soft_label(lam, gamma)

        # Create soft label tensor with fixed size if specified
        if total_classes is None:
            total_classes = len(self.classifiers)

        soft_label = torch.zeros(total_classes, device=self.device)

        # Make sure indices are within bounds
        if y1_int < total_classes and y2_int < total_classes and virtual_idx < total_classes:
            soft_label[y1_int] = prob_1
            soft_label[y2_int] = prob_2
            soft_label[virtual_idx] = prob_v
        else:
            # Handle out-of-bounds case
            valid_indices = torch.ones(total_classes, device=self.device) / total_classes
            soft_label = valid_indices  # Uniform distribution as fallback

        return mixed_features, soft_label

    def _apply_mixup(self, x1, x2, y1, y2, lam):
        """Helper method to apply mixup based on backbone type"""
        # Get integer class indices
        y1_int = int(y1.item())
        y2_int = int(y2.item())

        # Class pair (ordered for consistency)
        class_pair = (min(y1_int, y2_int), max(y1_int, y2_int))

        # Create virtual class if needed
        if class_pair not in self.virtual_class_indices:
            # New virtual class index
            virtual_idx = len(self.classifiers)
            self.virtual_class_indices[class_pair] = virtual_idx

            # Calculate midpoint classifier
            midpoint = self.compute_midpoint_classifier(y1_int, y2_int)

            # Expand classifiers
            new_classifiers = nn.Parameter(
                torch.cat([self.classifiers.data, midpoint], dim=0)
            ).to(self.device)

            self.classifiers = new_classifiers

        virtual_idx = self.virtual_class_indices[class_pair]

        # Apply motion-aware mixup if enabled
        if self.config.use_motion and isinstance(self.backbone, torch.nn.Module) and len(x1.shape) == 5:
            # Video data processing
            mixed_frames, adjusted_lam = self.motion_mixup(x1, x2, lam)

            # Extract features from mixed frames
            B, C, T, H, W = mixed_frames.shape
            frame_features = []

            for t in range(T):
                frame = mixed_frames[:, :, t]
                frame_feat = self.backbone.features(frame)
                frame_features.append(frame_feat)

            # Average over time dimension
            stacked_features = torch.stack(frame_features, dim=2)
            avg_features = torch.mean(stacked_features, dim=2)
            mixed_features = torch.flatten(avg_features, 1)

            lam = adjusted_lam
        else:
            # Standard image processing
            # Select a random layer for mixup
            if hasattr(self.backbone, 'features'):
                layers = list(self.backbone.features.children())
                layer_idx = np.random.randint(1, len(layers) - 1)

                # Forward to middle layer
                h1 = x1
                h2 = x2
                for j in range(layer_idx):
                    h1 = layers[j](h1)
                    h2 = layers[j](h2)

                # Mix features
                h_mixed = lam * h1 + (1 - lam) * h2

                # Continue forward pass
                for j in range(layer_idx, len(layers)):
                    h_mixed = layers[j](h_mixed)

                mixed_features = torch.flatten(h_mixed, 1)
            else:
                # Fallback to standard feature extraction
                with torch.no_grad():
                    feat1 = self.backbone(x1)
                    feat2 = self.backbone(x2)

                mixed_features = lam * feat1 + (1 - lam) * feat2

        # Calculate soft labels
        gamma = self.config.gamma
        prob_1, prob_v, prob_2 = self.compute_soft_label(lam, gamma)

        # Create soft label tensor
        soft_label = torch.zeros(len(self.classifiers), device=self.device)
        soft_label[y1_int] = prob_1
        soft_label[y2_int] = prob_2
        soft_label[virtual_idx] = prob_v

        return mixed_features, soft_label

    def cleanup_virtual_classifiers(self, num_real_classes):
        """Remove virtual classifiers after training"""
        # Store the updated number of real classes
        self.num_real_classes = num_real_classes

        # Keep only real class classifiers (not virtual ones)
        self.classifiers = nn.Parameter(self.classifiers[:num_real_classes].clone())

        # Reset virtual class indices
        self.virtual_class_indices = {}

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


# Motion-aware mixup implementation
class MotionAwareMixup(nn.Module):
    def __init__(self, config):
        super(MotionAwareMixup, self).__init__()
        self.config = config
        self.flow_alpha = config.flow_alpha

    def compute_motion_consistency(self, flow1, flow2, lam):
        # Computing optical flow coherence between two samples
        flow1_flat = flow1.reshape(2, -1)
        flow2_flat = flow2.reshape(2, -1)

        # Calculate magnitudes
        norm1 = torch.norm(flow1_flat, dim=1, keepdim=True)
        norm2 = torch.norm(flow2_flat, dim=1, keepdim=True)

        # Prevent division by zero
        epsilon = 1e-8
        cos_sim = torch.sum(flow1_flat * flow2_flat, dim=1) / (norm1 * norm2 + epsilon)

        # Consistency score: higher similarity -> closer to 0.5 (balanced mixing)
        consistency = torch.mean(cos_sim)
        adjusted_lam = lam + self.flow_alpha * (0.5 - lam) * consistency

        # Clamp to valid range
        adjusted_lam = torch.clamp(adjusted_lam, 0.0, 1.0)

        return adjusted_lam.item()

    def forward(self, frames1, frames2, lam):
        # Skip motion awareness if disabled
        if not self.config.use_motion:
            mixed_frames = lam * frames1 + (1 - lam) * frames2
            return mixed_frames, lam

        # Compute optical flow
        flow1 = compute_optical_flow(frames1)
        flow2 = compute_optical_flow(frames2)

        # Adjust mixing ratio based on motion consistency
        adjusted_lam = self.compute_motion_consistency(flow1[0], flow2[0], lam)

        # Apply adjusted mixing
        mixed_frames = adjusted_lam * frames1 + (1 - adjusted_lam) * frames2

        return mixed_frames, adjusted_lam


# Optical flow computation function
def compute_optical_flow(frames):
    # Input processing
    B, C, T, H, W = frames.shape
    flows = torch.zeros(B, 2, T - 1, H, W, device=frames.device)

    # Process each batch item
    for b in range(B):
        for t in range(T - 1):
            # Convert to NumPy for OpenCV processing
            prev_frame = frames[b, :, t].permute(1, 2, 0).detach().cpu().numpy()
            next_frame = frames[b, :, t + 1].permute(1, 2, 0).detach().cpu().numpy()

            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

            # Compute flow (with error handling)
            try:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Convert to tensor
                flows[b, 0, t] = torch.from_numpy(flow[:, :, 0]).to(frames.device)
                flows[b, 1, t] = torch.from_numpy(flow[:, :, 1]).to(frames.device)
            except Exception as e:
                print(f"Error in optical flow calculation: {e}")
                # Provide zeros as fallback
                flows[b, :, t] = 0

    return flows