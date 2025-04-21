# Import necessary libraries
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn

# Import user-defined modules
from config import Config
from data.dataloader.data_utils import set_up_datasets, get_dataloader
from model.mics import MICS  # Using our fixed implementation
from evaluate import evaluate, compute_nVar, visualize_pca, visualize_nVar, visualize_acc
from train import train_base, train_inc  # Using our fixed training functions
from train.trainer import MICSTrainer


# Seed setting for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to execute the entire MICS algorithm
def run_mics(config):
    # Initialize model with base classes
    model = MICS(config).to(config.device)

    # Initialize trainer
    trainer = MICSTrainer(model, config)

    # Base session Evaluation
    current_classes = config.base_classes
    acc_history = []
    nVar_history = []

    # Base session Training
    print("Training base session...")

    # Load base dataset
    trainset, trainloader, testloader = get_dataloader(config, 0)
    model = train_base(model, trainloader, config)

    # Evaluate base session
    acc_per_session = evaluate(model, [testloader], config, 0)
    acc_history.extend(acc_per_session)

    nVar = compute_nVar(model, [testloader], current_classes, config)
    nVar_history.append(nVar)

    # Visualize base session
    visualize_pca(model, [testloader], current_classes, config, 0)

    # Process each incremental session
    for session_idx in range(1, config.num_sessions + 1):
        print(f"\nTraining incremental session {session_idx}...")

        # load incremental dataset
        trainset, trainloader, testloader = get_dataloader(config, session_idx)

        # Create list of all test loaders up to the current session for evaluation
        test_loaders = []
        for s in range(session_idx + 1):
            _, _, testloader_s = get_dataloader(config, s)
            test_loaders.append(testloader_s)

        # Expand classifier for new classes
        novel_classes = config.way  # Number of new classes per session
        expanded_classifiers = nn.Parameter(
            torch.cat([
                model.classifiers.data,
                torch.randn(novel_classes, config.feature_dim).to(config.device)
            ], dim=0)
        )
        model.classifiers = expanded_classifiers
        current_classes += novel_classes

        # Incremental session training
        model = train_inc(model, trainloader, session_idx, current_classes, config)

        # Evaluate on all classes seen so far
        acc_per_session = evaluate(model, test_loaders, config, session_idx)
        acc_history.append(acc_per_session[-1])  # Store accuracy for current session

        nVar = compute_nVar(model, test_loaders, current_classes, config)
        nVar_history.append(nVar)

        # Visualize feature space
        visualize_pca(model, test_loaders, current_classes, config, session_idx)

    return model, nVar_history, acc_history


# Main function
def main():
    # Set seed for reproducibility
    set_seed(1)

    # Call pre-set config
    config = Config()

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
    model_plain, nVar_plain, acc_plain = run_mics(config)

    # Save checkpoints
    torch.save({
        'model_state_dict': model_plain.state_dict(),
        'nVar': nVar_plain,
        'accuracy': acc_plain
    }, os.path.join(results_dir, 'plain_mics_checkpoint.pth'))

    # Motion-aware MICS (UCF101)
    print("\n" + "=" * 50)
    print("Running Motion-Aware MICS algorithm...")
    print("=" * 50)

    # Reconfigure for motion-aware MICS
    config.use_motion = True
    config.dataset = 'ucf101'
    config = set_up_datasets(config)  # Setup Arguments

    # Run Extended MICS algorithm and save checkpoints
    model_motion, nVar_motion, acc_motion = run_mics(config)

    # Save checkpoints
    torch.save({
        'model_state_dict': model_motion.state_dict(),
        'nVar': nVar_motion,
        'accuracy': acc_motion
    }, os.path.join(results_dir, 'motion_mics_checkpoint.pth'))

    # Visualizations for comparison
    visualize_nVar(nVar_plain, nVar_motion, config)
    visualize_acc(acc_plain, acc_motion, config)

    print("\nTraining and evaluation completed!")
    print(f"Plain MICS final accuracy: {acc_plain[-1]:.2f}%")
    print(f"Motion-Aware MICS final accuracy: {acc_motion[-1]:.2f}%")

    # Calculate performance dropping rate
    pd_plain = acc_plain[0] - acc_plain[-1]
    pd_motion = acc_motion[0] - acc_motion[-1]

    print(f"Plain MICS performance dropping rate: {pd_plain:.2f}%")
    print(f"Motion-Aware MICS performance dropping rate: {pd_motion:.2f}%")

    return model_plain, model_motion