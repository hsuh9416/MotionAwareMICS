# Import necessary libraries
import torch
import multiprocessing


# Common config class
class Config:
    def __init__(self):
        # Base
        self.device = torch.device("cuda")
        self.phase = 'inc'
        self.project = 'mics'
        self.dataset = 'cifar100' # or 'ucf101'
        self.dataroot = '/content/drive/MyDrive/MotionAwareMICS/data/'  # Colab
        self.gpu = 0
        self.num_workers = 8
        self.seed = 1
        self.memo = ''

        # Training - Same as the paper
        self.base_mode = 'ft_cos'  # Cosine classifier
        self.new_mode = 'avg_cos'  # Average data embedding / Cosine classifier
        self.schedule = 'Step'  # Schedule function
        self.step_size = 40  # Check schedule every 40 epochs
        self.batch_size = 128  # High-capacity fits with A100 GPU
        self.base_epochs = 100  # Base session epoch number
        self.inc_epochs = 100  # Incremental session epoch number
        self.learning_rate = 0.1  # Initial learning rate
        self.inc_learning_rate = 0.1  # Initial learning rate
        self.momentum = 0.9  # SGD momentum value
        self.weight_decay = 5e-4  # Weight decay (normalization)
        self.temperature = 16  # Cosine similarity temperature scaling (mentioned in the paper)
        self.st_ratio = 0.01  # session trainable parameter ratio

        # MICS settings - based on Table 4 (Section 4.5) of the paper
        # CIFAR-100(Best practice)
        self.alpha = 0.5  # Parameters of the beta distribution
        self.gamma = 0.1  # Parameters of soft labeling policy
        self.epsilon = 0.01  # The percentage of parameters to update in incremental steps

        # Motion recognition settings
        self.use_motion = False  # Whether motion recognition function is enabled
        self.flow_alpha = 0.5  # optical flow weighting
