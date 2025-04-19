# Import necessary libraries
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# User-defined
from config import Config
from mics_impl import MICS
from dataloader import load_cifar100, load_ucf101
from evaluate import evaluate, compute_nVar, visualize_pca
from train import train_base_session, train_incremental_session

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
    # load dataset
    if config.dataset == 'cifar100': # CIFAR-100
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

    # Create dataloader
    train_loaders = [
        DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                  num_workers=config.num_workers)
        for dataset in sessions_train_data
    ]

    test_loaders = [
        DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                  num_workers=config.num_workers)
        for dataset in sessions_test_data
    ]

    # Init model
    model = MICS(config, config.base_classes).to(config.device)

    # Base session Training
    print("Training base session...")
    model = train_base_session(model, train_loaders[0], config)

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
        model = train_incremental_session(
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
def main():
    # Set seed
    set_seed(42)

    # Call pre-set config
    config = Config()

    # Plain MICS
    print("Running Plain MICS algorithm...")
    model_plain, nVar_plain, history_plain = run_mics(config)

    # Motion aware MICS
    config.use_motion = True
    config.dataset = 'ucf101' # Dataset for motion detection
    config.backbone = 'resnet18'
    config.feature_dim = 512
    model_motion, nVar_motion, history_motion = run_mics(config)

    # Visualize the results(Acc, Loss)
    #TODO to be updated

    print("Experiment completed!")