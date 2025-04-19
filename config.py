# Import necessary libraries
import torch
import multiprocessing
import data.dataloader.cifar100.cifar as Dataset
# Config class
class Config:
    # Default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Increase dataloader workers based on virtual CPU count
    num_workers = min(multiprocessing.cpu_count() * 2, 12)

    # Dataset(Default: CIFAR-100)
    dataset = 'cifar100'  # Plain: 'cifar100' Motion-aware: 'ucf101'
    dataroot = '/content/drive/MyDrive/MotionAwareMICS/data/' # Colab
    base_class = 60     # Follows standard FSCIL protocol. base: increment = 60: 40
    num_classes = 100 # Total number of classes
    way = 5
    shots = 5
    sessions = 8  # 40 / 5 = 8 sessions
    Dataset = Dataset # Pre-defined dataset class

    # Feature Extractor
    backbone = 'resnet20' # Feature Extractor
    feature_dim = 64     # Feature Vector Dimension for resnet20

    # Training - Same as the paper
    batch_size = 128 # High capacity fits with A100 GPU
    base_epochs = 100 # Base session epoch number
    inc_epochs = 10 # Incremental session epoch number
    learning_rate = 0.1 # Initial learning rate
    momentum = 0.9 # SGD momentum value
    weight_decay = 5e-4 # Weight decay (normalization)
    temperature = 0.1 # Cosine similarity temperature scaling (mentioned in the paper)

    # MICS settings - based on Table 4 (Section 4.5) of the paper
    # CIFAR-100(Best practice)
    alpha = 0.5  # Parameters of the beta distribution
    gamma = 0.5  # Parameters of soft labeling policy
    epsilon = 0.01  # The percentage of parameters to update in incremental steps

    # Motion recognition settings
    use_motion = False  # Whether motion recognition function is enabled
    flow_alpha = 0.5    # optical flow weighting