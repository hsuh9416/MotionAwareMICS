# Import necessary libraries
import os
import shutil
import random
import numpy as np
import torch

# Import user-defined modules
from baseconfig import BaseConfig
from data.dataloader.data_utils import set_up_datasets
from evaluate import (
    visualize_pca,
    visualize_nvar_progression,
    visualize_accuracy_progression,
    compute_nVar
)
from mics import MICS
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
    """Run the MICS training process and return results"""
    print(f"\nInitializing MICS model for {config.dataset}...")

    # Initialize model with base classes
    model = MICS(config).cuda()

    # Initialize trainer
    trainer = MICSTrainer(model, config)

    # Train and Evaluate the model
    trainer.train()

    # Evaluation results
    results = trainer.results

    # Create visualizations for this dataset
    create_visualizations(model, trainer, results, config)

    return results


def create_visualizations(model, trainer, results, config):
    """Create and save visualizations for the current dataset"""
    print(f"\nGenerating visualizations for {config.dataset}...")

    # Ensure the visualization directory exists
    viz_dir = os.path.join(config.save_path, 'visualizations', config.dataset)
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Visualize accuracy progression
    print("- Creating accuracy progression visualization...")
    visualize_accuracy_progression(results['acc'], config)

    # 2. Visualize nVAR progression
    print("- Creating nVAR progression visualization...")
    visualize_nvar_progression(results['train_nVAR'], config)

    # 3. Visualize PCA for the final session
    print("- Creating PCA visualization for feature space...")
    # Get data from the last session's test loader
    final_session = len(results['acc']) - 1
    _, _, test_loader = trainer.get_test_data(final_session)

    # Calculate current classes based on session
    current_classes = config.base_class + final_session * config.way

    # Create PCA visualization
    visualize_pca(model, test_loader, current_classes, final_session, config)

    print(f"Visualizations saved to: {viz_dir}")


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

    # Dictionary to store results
    all_results = {}

    # Run CIFAR-100 experiment
    print("\n" + "=" * 50)
    print("Running MICS on CIFAR-100 dataset...")
    print("=" * 50)

    # Configure for CIFAR-100
    config.dataset = 'cifar100'
    config = set_up_datasets(config)
    config.use_motion = False  # Disable motion awareness for CIFAR-100

    # Run experiment
    cifar_results = run_process(config)

    print(f"\nCIFAR-100 Results:")
    print(f"  Final Accuracy: {cifar_results['acc'][-1]:.2f}%")
    print(f"  Performance Degradation: {cifar_results['acc'][0] - cifar_results['acc'][-1]:.2f}%")

    # Run HMDB51 experiment with motion awareness
    print("\n" + "=" * 50)
    print("Running Motion-Aware MICS on HMDB51 dataset...")
    print("=" * 50)

    # Configure for HMDB51 with motion awareness
    config.dataset = 'hmdb51'
    config = set_up_datasets(config)
    config.use_motion = True
    config.flow_alpha = 0.5  # Optical flow weighting factor

    # Run motion-aware experiment
    hmdb_results = run_process(config)

    print(f"\nHMDB51 Motion-Aware MICS Results:")
    print(f"  Final Accuracy: {hmdb_results['acc'][-1]:.2f}%")
    print(f"  Performance Degradation: {hmdb_results['acc'][0] - hmdb_results['acc'][-1]:.2f}%")

    print("\n" + "=" * 50)
    print("All experiments completed successfully!")
    print("=" * 50)