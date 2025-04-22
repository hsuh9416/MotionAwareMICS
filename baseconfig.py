# Import necessary libraries
import torch
import os


# Common config class
class BaseConfig:
    def __init__(self):
        # Base
        self.device = torch.device("cuda")
        self.phase = 'inc'
        self.project = 'mics'
        self.dataset = 'cifar100'  # or 'ucf101'
        self.dataroot = '/content/drive/MyDrive/MotionAwareMICS/data/'  # Colab
        if not os.path.exists(self.dataroot):
            os.makedirs(self.dataroot, exist_ok=True)

        self.gpu = 0
        self.num_workers = 8
        self.seed = 1
        self.memo = ''

        # Training - Same as the paper
        self.epochs_base = 10  # epochs_base - Base session epoch number, In paper 600
        self.inc_epochs = 10  # epochs_new - Incremental session epoch number
        self.epochs_new = 10  # In Paper
        self.learning_rate = 0.1  # lr_base - Initial learning rate
        self.inc_learning_rate = 0.0005  # lr_new - Initial learning rate
        self.schedule = 'Step'  # Schedule function
        # self.milestones = [60, 70]
        self.step_size = 40  # Step = Check schedule every 40 epochs
        self.weight_decay = 5e-4  # decay - Weight decay (normalization)
        self.momentum = 0.9  # SGD momentum value
        self.gamma = 0.1  # Parameters of soft labeling policy
        self.temperature = 16  # Cosine similarity temperature scaling (mentioned in the paper)
        self.batch_size = 128  # High-capacity fits with A100 GPU, In paper 256
        # self.batch_size_new = 0

        self.base_mode = 'ft_cos'  # Cosine classifier
        self.new_mode = 'avg_cos'  # Average data embedding / Cosine classifier
        # self.start_session = 0
        self.model_dir = '/content/drive/MyDrive/MotionAwareMICS/models/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        # MICS settings - based on Table 4 (Section 4.5) of the paper
        self.st_ratio = 0.01  # session trainable parameter ratio
        self.train = 'mixup_hidden'
        self.alpha = 0.5  # mixup_alpha - Parameters of the beta distribution
        # self.use_hard_positive_aug = False
        # self.hpa_type = 'none'
        # self.add_noise_level = 0.
        # self.mult_noise_level = 0.
        # self.minimum_lambda = 0.5
        # self.label_sharpening = True # Only can be used when use_hard_positive_aug is True
        self.label_mix = 'steep_dummy'
        self.epsilon = 0.01  # label_mix_threshold - The percentage of parameters to update in incremental steps
        self.label_mix_threshold = 0.5
        # self.gaussian_h1 = 0.2
        # self.piecewise_linear_h1 = 0.5
        # self.piecewise_linear_h2 = 0.
        self.num_similar_class = 3
        self.num_pre_allocate = 40
        # self.normalized_middle_classifier = False
        # self.exp_coef = 1. # Only used with exponential dummy
        self.drop_last = True
        # self.use_resnet_alice = False
        self.use_mixup = True
        self.use_softlabel = True
        self.use_midpoint = True

        # cifar100 pre-trained
        self.checkpoint = os.path.join(self.model_dir, 'cifar100.pth')

        # Motion recognition settings
        self.use_motion = False  # Whether motion recognition function is enabled
        self.flow_alpha = 0.5  # optical flow weighting
