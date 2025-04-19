# Import necessary libraries
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import user-defined modules
from config import Config
from test_config import TestConfig
from mics_impl import MICS  # Using our fixed implementation
from dataloader import load_cifar100, load_ucf101
from evaluate import evaluate, compute_nVar, visualize_pca, visualize_nVar, visualize_acc
from train import train_base, train_inc  # Using our fixed training functions


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
    # Load appropriate dataset
    if config.dataset == 'cifar100':  # CIFAR-100
        sessions_train_data, sessions_test_data = load_cifar100(
            config.base_classes,
            config.novel_classes_per_session,
            config.num_sessions
        )
    else:  # UCF101
        sessions_train_data, sessions_test_data = load_ucf101(
            config.base_classes,
            config.novel_classes_per_session,
            config.num_sessions,
            config.shots_per_class
        )

    # Create dataloaders with the configured batch size and worker count
    train_loaders = [
        DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                   num_workers=config.num_workers, pin_memory=True)
        for dataset in sessions_train_data
    ]

    test_loaders = [
        DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                   num_workers=config.num_workers, pin_memory=True)
        for dataset in sessions_test_data
    ]

    # Initialize model with base classes
    model = MICS(config, config.base_classes).to(config.device)

    # Base session Training
    print("Training base session...")
    model = train_base(model, train_loaders[0], config)

    # Base session Evaluation
    current_classes = config.base_classes
    acc_history = []
    nVar_history = []

    # Evaluate base session
    acc_per_session = evaluate(model, [test_loaders[0]], config)
    acc_history.extend(acc_per_session)

    nVar = compute_nVar(model, [test_loaders[0]], current_classes, config)
    nVar_history.append(nVar)

    # Visualize base session
    visualize_pca(model, [test_loaders[0]], current_classes, config, 0)

    # Process each incremental session
    for session_idx in range(1, config.num_sessions + 1):
        if session_idx < len(train_loaders):
            print(f"\nTraining incremental session {session_idx}...")

            # Expand classifier for new classes
            novel_classes = config.novel_classes_per_session
            model.expand_classifier(novel_classes)
            current_classes += novel_classes

            # Incremental Session Training
            model = train_inc(
                model, train_loaders[session_idx], session_idx, current_classes, config
            )

            # Evaluation of all sessions up to current
            acc_per_session = evaluate(model, test_loaders[:session_idx + 1], config)
            acc_history.append(acc_per_session[-1])  # Store final session accuracy

            nVar = compute_nVar(model, test_loaders[:session_idx + 1], current_classes, config)
            nVar_history.append(nVar)

            # Visualization
            visualize_pca(model, test_loaders[:session_idx + 1], current_classes, config, session_idx)
        else:
            print(f"Skipping session {session_idx} - no data available")

    return model, nVar_history, acc_history


# Main function
def main(isTest):
    # Set seed for reproducibility
    set_seed(1)

    # Call pre-set config
    config = TestConfig() if isTest else Config()

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
    config.backbone = 'resnet20'
    config.feature_dim = 64
    config.use_motion = False

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
    config.backbone = 'resnet18'
    config.feature_dim = 512

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