# Import necessary libraries
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# User-defined
from config import Config
from test_config import TestConfig
from mics_impl import MICS
from dataloader import load_cifar100, load_ucf101
from evaluate import evaluate, compute_nVar, visualize_pca, visualize_nVar, visualize_acc
from train import train_base, train_inc

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

    # Init model
    model = MICS(config, config.base_classes).to(config.device)

    # Base session Training
    print("Training base session...")
    model = train_base(model, train_loaders[0], config)

    # Base session Evaluation
    current_classes = config.base_classes
    acc_per_session = evaluate(model, [test_loaders[0]], config)
    nVar = compute_nVar(model, [test_loaders[0]], current_classes, config)
    visualize_pca(model, [test_loaders[0]], current_classes, config, 0)

    # Process each incremental session
    for session_idx in range(1, config.num_sessions + 1):
        print(f"\nTraining incremental session {session_idx}...")

        # Classifier extension
        novel_classes = config.novel_classes_per_session
        expanded_classifiers = nn.Parameter(
            torch.cat([
                model.classifiers.data,
                torch.randn(novel_classes, config.feature_dim).to(config.device)
            ], dim=0)
        )
        model.classifiers = expanded_classifiers
        current_classes += novel_classes

        # Incremental Session Training
        model = train_inc(
            model, train_loaders[session_idx], session_idx, current_classes, config
        )

        # Evaluation
        acc_per_session = evaluate(
            model, test_loaders[:session_idx+1], config)
        nVar = compute_nVar(
            model, test_loaders[:session_idx+1], current_classes, config
        )
        visualize_pca(
            model, test_loaders[:session_idx+1], current_classes, config, session_idx
        )

    return model, nVar, acc_per_session


# Main function
def main(isTest):
    # Set seed
    set_seed(1)

    # Call pre-set config
    config = TestConfig() if isTest else Config()

    # Setup directory for result saving
    if os.path.exists('results'):
        shutil.rmtree('results')
    os.makedirs('results')

    # Plain MICS
    print("=" * 50)
    print("Running Plain MICS algorithm...")
    print("=" * 50)
    model_plain, nVar_plain, acc_plain = run_mics(config)

    # Save checkpoints (back up intermediate results)
    torch.save({
        'model_state_dict': model_plain.state_dict(),
        'nVar': nVar_plain,
        'accuracy': acc_plain
    }, 'results/plain_mics_checkpoint.pth')

    print("\n" + "=" * 50)
    print("Running Motion-Aware MICS algorithm...")
    print("=" * 50)

    # Motion aware MICS
    config.use_motion = True
    config.dataset = 'ucf101' # Dataset for motion detection
    config.backbone = 'resnet18'
    config.feature_dim = 512

    model_motion, nVar_motion, acc_motion = run_mics(config)

    # Save checkpoints (back up intermediate results)
    torch.save({
        'model_state_dict': model_motion.state_dict(),
        'nVar': nVar_motion,
        'accuracy': acc_motion
    }, 'results/motion_mics_checkpoint.pth')

    # Visualize nVar
    visualize_nVar(nVar_plain, nVar_motion, config)

    # Visualize session-by-session accuracy and performance degradation
    visualize_acc(acc_plain, acc_motion, config)

