# Import necessary library
import torch

# Config class
class Config:
    # Default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 # Parallel workers

    # Dataset
    dataset = 'cifar100'  # Plain: 'cifar100' Motion-aware: 'ucf101'
    base_classes = 60     # Follows standard FSCIL protocol. base: increment = 60: 40
    novel_classes_per_session = 5  # Number of new classes per session
    num_sessions = 8      # 40/5 = 8 sessions
    shots_per_class = 5   # 5-shot

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