import torch.nn as nn
from model.mix_up import *

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last # Investigates whether this is the last block or not
        self.expansion = 1

    def forward(self, x):
        residual = x

        # First block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second block
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsampling (Stride > 1)
        if self.downsample:
            residual = self.downsample(x)

        # Residual concatenation
        out += residual

        # Output activation function
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # make layers
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2, is_last=True)

    def make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        # If stride is greater than 1 or the channel is larger than the input, downsampling layer is required.
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = [] # Layer blocks.

        # Add the first block of the layer.(Downsampling applied)
        first_layer = block(self.in_planes, planes, stride, downsample)
        layers.append(first_layer)

        self.in_planes = planes * block.expansion # Increase the number of input channels

        # Add the remaining blocks of the layer.
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))

        # Note for the final layer, is_last is always True.
        last_layer = block(self.inplanes, planes, is_last = True if is_last else False)
        layers.append(last_layer)

        return nn.Sequential(*layers)

    def forward(self, x, labels=None, mixup_alpha=0.5, num_base_classes=-1, piecewise_linear_h1=0.5, piecewise_linear_h2=0.):
        """ forward function for ResNet-20 with Maniford mix-up."""
        # Input layer
        out = x

        """
            # Maniford Mix-up: Mix-up technique used in the paper
            Reference: Manifold Mixup [Souvik Mandal, Medium]
            https://medium.com/@mandalsouvik/manifold-mixup-learning-better-representations-by-interpolating-hidden-states-8a2c949d5b5b
            - lambda = beta distribution with mixup_alpha
        """
        # Pick the layer to conduct mix-up
        layer_mix = 2 # 0: Input, 1: First layer, 2: Second layer

        # Compute lambda

        lamb = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1
        lamb = torch.from_numpy(np.array([lamb]).astype('float32')).cuda()

        # Re-weighting labels for mix-up
        new_labels = to_one_hot(labels, self.num_classes)

        # Layer mix after Input
        if layer_mix == 0:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes,lamb,
                                                                   piecewise_linear_h1=piecewise_linear_h1,
                                                                   piecewise_linear_h2=piecewise_linear_h2)

        # First layer
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)

        # Layer mix after First layer
        if layer_mix == 1:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb,
                                                                   piecewise_linear_h1=piecewise_linear_h1,
                                                                   piecewise_linear_h2=piecewise_linear_h2)

        # Second layer
        out = self.layer2(out)

        # Layer mix after Second layer
        if layer_mix == 1:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb,
                                                                   piecewise_linear_h1=piecewise_linear_h1,
                                                                   piecewise_linear_h2=piecewise_linear_h2)

        # Third layer
        out = self.layer3(out)

        return out

def resnet20(**kwargs):
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # Get pretrained model

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
