# Import necessary libraries
import os
import shutil
import random
import numpy as np
import torch

# Import user-defined modules
from baseconfig import BaseConfig
from data.dataloader.data_utils import set_up_datasets
from evaluate import visualize_performance_comparison, visualize_pca, visualize_nVar
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
    config = set_up_datasets(config)  # Setup Arguments
    config.use_motion = False

    # Run MICS algorithm and save checkpoints
    plain_results = run_process(config)

    # # Motion-aware MICS (HMDB51)
    # print("\n" + "=" * 50)
    # print("Running Motion-Aware MICS algorithm...")
    # print("=" * 50)
    #
    # # Reconfigure for motion-aware MICS
    # config.use_motion = True
    # config.flow_alpha = 0.5  # Optical flow weighting factor
    # config.dataset = 'hmdb51'
    # config = set_up_datasets(config)  # Setup Arguments
    #
    # # Run Motion-Aware MICS algorithm
    # motion_results = run_process(config)
    #
    # # Visualize and compare results
    # visualize_acc(plain_results['acc'], motion_results['acc'], config)
    # visualize_nVar(plain_results['train_nVAR'], motion_results['train_nVAR'], config)
    #
    # # Save detailed comparison
    # print("Performance Summary:")
    # print(
    #     f"Plain MICS - Final Accuracy: {plain_results['acc'][-1]:.2f}%, PD: {plain_results['acc'][0] - plain_results['acc'][-1]:.2f}%")
    # print(
    #     f"Motion-Aware MICS - Final Accuracy: {motion_results['acc'][-1]:.2f}%, PD: {motion_results['acc'][0] - motion_results['acc'][-1]:.2f}%")
