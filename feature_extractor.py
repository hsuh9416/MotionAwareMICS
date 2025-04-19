# Import necessary libraries
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Implementation of ResNet-20 (for CIFAR-100)
class ResNet20Backbone(nn.Module):
    """
     Pre-trained model source:
      Repository: https://github.com/chenyaofo/pytorch-cifar-models
      Model: cifar100_resnet20
      Accessed via torch.hub.load (pretrained=True)
    License: MIT License
    Note: The classification head is removed to use the network as a feature extractor.
    """
    def __init__(self, feature_dim=64):
        super(ResNet20Backbone, self).__init__()
        # Load the pre-trained ResNet-20 model
        pretrained_model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                                          "cifar100_resnet20", pretrained=True)

        # Remove classification layer (This is for feature extraction only!)
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.feature_dim = feature_dim

    def forward(self, x):
        # Use a pre-trained model
        x = self.features(x)
        x = torch.flatten(x, 1)

        return x

# ResNet18 backbone model (for UCF101/motion dataset)
class ResNet18Backbone(nn.Module):
    def __init__(self, feature_dim=512):
        super(ResNet18Backbone, self).__init__()
        # Initialize with ImageNet pre-trained weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Feature extractor: Excluding the last FC layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = feature_dim

    def forward(self, x):
        # Input: [B, C, T, H, W] - batch, channel, time, height, width
        B, C, T, H, W = x.shape

        # Apply ResNet to each frame individually
        features = []
        for t in range(T):
            frame = x[:, :, t]  # [B, C, H, W]
            frame_feat = self.features(frame)  # [B, feat_dim, 1, 1]
            features.append(frame_feat)

        # Average of all frame features
        x = torch.stack(features, dim=2)  # [B, feat_dim, T, 1, 1]
        x = torch.mean(x, dim=2)  # Average over time dimension
        x = torch.flatten(x, 1)

        return x