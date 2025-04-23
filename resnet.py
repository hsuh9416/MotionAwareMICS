import hashlib
import random
import re
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

import torch.nn as nn

from mix_up import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last  # Investigates whether this is the last block or not
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation

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


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=100, downsample=None, groups=1,
                 width_per_group=64, dilation=1, replace_stride_with_dilation=None):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.dilation = dilation
        self.downsample = downsample
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=1, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def make_layer(self, block, planes, blocks, stride=1, dilate=False):
        expansion = block.expansion if block.expansion else 1
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion))

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def _process_frame(self, frame, labels, mix_type, mixup_alpha, num_base_classes, gamma):
        """Process a single frame through the network with mixup"""
        # Pick the layer to conduct mix-up
        if "mixup_hidden" in mix_type:
            layer_mix = random.randint(0, 3)  # 0: Input, 1: First layer, 2: Second layer, 3: Third layer
        else:
            layer_mix = None

        out = frame

        # Compute lambda
        lamb = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1
        lamb = torch.from_numpy(np.array([lamb]).astype('float32')).cuda()

        # Re-weighting labels for mix-up
        if labels is not None:
            target_reweighted = to_one_hot(labels, self.num_classes)
        else:
            target_reweighted = None

        # Rest of your original frame processing code...

        return out, target_reweighted, mix_label_mask

    def forward(self, x, labels=None, mix_type="vanilla", mixup_alpha=0.5, num_base_classes=-1, gamma=0.5):
        """ forward function for ResNet-18 with Maniford mix-up."""
        # Initialize mix_label_mask
        mix_label_mask = None

        # Pick the layer to conduct mix-up
        if "mixup_hidden" in mix_type:
            layer_mix = random.randint(0, 3)  # 0: Input, 1: First layer, 2: Second layer, 3: Third layer
        else:
            layer_mix = None

        new_labels = None
        out = x

        # Compute lambda = beta distribution with mixup_alpha
        lamb = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1
        lamb = torch.from_numpy(np.array([lamb]).astype('float32')).cuda()

        # Re-weighting labels for mix-up
        new_labels = to_one_hot(labels, self.num_classes)

        # Layer mix after Input
        if layer_mix == 0:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb, gamma=gamma)

        # First layer
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)

        # Layer mix after First layer
        if layer_mix == 1:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb, gamma=gamma)

        # Second layer
        out = self.layer2(out)

        # Layer mix after Second layer
        if layer_mix == 2:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb, gamma=gamma)

        # Third layer
        out = self.layer3(out)

        # Layer mix after Third layer
        if layer_mix == 3:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb, gamma=gamma)

        # Forth layer
        out = self.layer4(out)

        return out, new_labels, mix_label_mask


class ResNet20(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet20, self).__init__()
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
        expansion = block.expansion if block.expansion else 1
        downsample = None
        # If stride is greater than 1 or the channel is larger than the input, a downsampling layer is required.
        if stride != 1 or self.in_planes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion))

        layers = []  # Layer blocks.

        # Add the first block of the layer.(Downsampling applied)
        first_layer = block(self.in_planes, planes, stride, downsample)
        layers.append(first_layer)

        self.in_planes = planes * expansion  # Increase the number of input channels

        # Add the remaining blocks of the layer.
        for i in range(1, blocks - 1):
            layers.append(block(self.in_planes, planes))

        # Note for the final layer, is_last is always True.
        last_layer = block(self.in_planes, planes, is_last=True if is_last else False)
        layers.append(last_layer)

        return nn.Sequential(*layers)

    def forward(self, x, labels=None, mix_type="vanilla", mixup_alpha=0.5, num_base_classes=-1, gamma=0.5):
        """ forward function for ResNet-20 with Maniford mix-up."""
        # Initialize mix_label_mask
        mix_label_mask = None
        # Input layer
        out = x

        """
            # Maniford Mix-up: Mix-up technique used in the paper
            Reference: Manifold Mixup [Souvik Mandal, Medium]
            https://medium.com/@mandalsouvik/manifold-mixup-learning-better-representations-by-interpolating-hidden-states-8a2c949d5b5b
        """
        # Pick the layer to conduct mix-up
        if "mixup_hidden" in mix_type:
            layer_mix = random.randint(0, 2) # 0: Input, 1: First layer, 2: Second layer
        else:
            layer_mix = None

        new_labels = None

        # Compute lambda = beta distribution with mixup_alpha
        lamb = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1
        lamb = torch.from_numpy(np.array([lamb]).astype('float32')).cuda()

        # Re-weighting labels for mix-up
        if labels is not None:
            new_labels = to_one_hot(labels, self.num_classes)

        # Layer mix after Input
        if layer_mix == 0:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb, gamma=gamma)

        # First layer
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)

        # Layer mix after First layer
        if layer_mix == 1:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb, gamma=gamma)

        # Second layer
        out = self.layer2(out)

        # Layer mix after Second layer
        if layer_mix == 2:
            out, new_labels, mix_label_mask = middle_mixup_process(out, new_labels, num_base_classes, lamb, gamma=gamma)

        # Third layer
        out = self.layer3(out)

        if labels is not None:
            return out, new_labels, mix_label_mask
        else:
            return out

def resnet18(**kwargs):
    """
     ResNet-18 model from
     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    model = ResNet18(BasicBlock, [2, 2, 2, 2], **kwargs)
    # Get and updated from pretrained model
    model_dict = model.state_dict()
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
    state_dict = {k: v for k, v in state_dict.items() if k not in ['fc.weight', 'fc.bias']}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    return model


def resnet20(**kwargs):
    model = ResNet20(BasicBlock, [3, 3, 3], **kwargs)
    return model


def _get_torch_home():
    ENV_TORCH_HOME = 'TORCH_HOME'
    ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
    DEFAULT_CACHE_DIR = '~/.cache'

    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home


def load_state_dict_from_url(url):
    r"""Loads the Torch serialized object at the given URL.

    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        hash_prefix = HASH_REGEX.search(filename).group(1)
        _download_url_to_file(url, cached_file, hash_prefix, progress=False)
    return torch.load(cached_file, map_location=None)


def _download_url_to_file(url, dst, hash_prefix, progress):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overriden by a broken download.
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        from tqdm import tqdm
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
