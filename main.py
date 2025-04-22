# Import necessary libraries
import os
import shutil
import random
import numpy as np
import torch

# Import user-defined modules
from baseconfig import BaseConfig
from data.dataloader.data_utils import set_up_datasets
from mics import MICS  # MICS model with motion-aware features
from trainer import MICSTrainer

# Seed setting for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to execute the entire MICS algorithm
def run_process(config):
    # Initialize model with base classes
    model = MICS(config).cuda()

    # Initialize trainer
    trainer = MICSTrainer(model, config)

    # Train and Evaluate the model
    trainer.train()

    # Evaluation results
    results = trainer.results

    return results

# Main function
def main():
    # Set seed for reproducibility
    set_seed(1)

    # Call pre-set config
    config = BaseConfig()

    # Setup directory for result saving
    results_dir = '/content/drive/MyDrive/MotionAwareMICS/results'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    # Plain MICS (CIFAR-100)
    print("=" * 50)
    print("Running Plain MICS algorithm...")
    print("=" * 50)

    # Ensure proper configuration for plain MICS
    config.dataset = 'cifar100'
    config = set_up_datasets(config) # Setup Arguments
    config.use_motion = False

    # Run MICS algorithm and save checkpoints
    plain_results = run_process(config)

    #TODO: Visualizations for comparison
    print(plain_results)

    # Motion-aware MICS (UCF101)
    print("\n" + "=" * 50)
    print("Running Motion-Aware MICS algorithm...")
    print("=" * 50)

    # Reconfigure for motion-aware MICS
    config.use_motion = True
    config.dataset = 'ucf101'
    config = set_up_datasets(config)  # Setup Arguments

    # Run Extended MICS algorithm and save checkpoints
    motion_results = run_process(config)

    #TODO: Visualizations for comparison
    print(motion_results)