# Import necessary libraries
import torch
import multiprocessing

# Common config class
class Config:
    def __init__(self):
        # Default
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Increase dataloader workers based on virtual CPU count
        self.num_workers = min(multiprocessing.cpu_count() * 2, 12)

        self.dataroot = '/content/drive/MyDrive/MotionAwareMICS/data/' # Colab

        # Training - Same as the paper
        self.batch_size = 128 # High-capacity fits with A100 GPU
        self.base_epochs = 100 # Base session epoch number
        self.inc_epochs = 10 # Incremental session epoch number
        self.learning_rate = 0.1 # Initial learning rate
        self.momentum = 0.9 # SGD momentum value
        self.weight_decay = 5e-4 # Weight decay (normalization)
        self.temperature = 0.1 # Cosine similarity temperature scaling (mentioned in the paper)

        # MICS settings - based on Table 4 (Section 4.5) of the paper
        # CIFAR-100(Best practice)
        self.alpha = 0.5  # Parameters of the beta distribution
        self.gamma = 0.5  # Parameters of soft labeling policy
        self.epsilon = 0.01  # The percentage of parameters to update in incremental steps

        # Motion recognition settings
        self.use_motion = False  # Whether motion recognition function is enabled
        self.flow_alpha = 0.5    # optical flow weighting