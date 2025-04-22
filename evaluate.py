# Import necessary libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Calculating the normalized variance (nVAR)
def compute_nVar(model, dataloader, num_classes):
    """ Compute the normalized variance (nVAR) for each class."""
    model.eval()  # Evaluation mode

    # Compute prototype
    class_prototypes = model.fc.weight.data[:num_classes].clone()

    # feature vector by class
    features_by_class = {i: [] for i in range(num_classes)}
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda() for _ in batch]
            features = model.encode(data)

            # classification by class
            for j in range(len(label)):
                class_idx = label[j].item()
                if class_idx < num_classes:  # Only consider current included classes
                    features_by_class[class_idx].append(features[j].detach().cpu())

    # Find the closest centroid of each class
    nearest_prototypes = {}
    for i in range(num_classes):
        dists = []
        for j in range(num_classes):
            if i != j:
                dist = torch.norm(class_prototypes[i] - class_prototypes[j], p=2)
                dists.append((j, dist.item()))

        if dists:  # If there are other classes
            nearest_idx, nearest_dist = min(dists, key=lambda x: x[1])
            nearest_prototypes[i] = (nearest_idx, nearest_dist)

    # Calculating within-class variance and normalized variance
    nvar_values = []
    for class_idx in range(num_classes):
        if features_by_class[class_idx] and class_idx in nearest_prototypes:
            features = torch.stack(features_by_class[class_idx])
            prototype = class_prototypes[class_idx].cpu()

            # Paper 3.1. Compact and Separable Representations: nVAR - Formula (1)
            # inter-class seperability (Mean-squared distance from class centroid)
            intra_var = torch.mean(torch.norm(features - prototype, dim=1) ** 2).item()

            # intra-class compactness (Mean squared distance from nearest centroid)
            _, nearest_dist = nearest_prototypes[class_idx]

            # nVAR for this class
            nvar = intra_var / (nearest_dist ** 2)
            nvar_values.append(nvar)

    # average nVAR for all classes
    avg_nvar = sum(nvar_values) / len(nvar_values) if nvar_values else float('inf')
    return avg_nvar


# PCA visualization function (Per session)
def visualize_pca(model, test_loaders, current_classes, session_idx):
    model.eval()
    all_features = []
    all_labels = []

    # Collect characteristics
    with torch.no_grad():
        for loader_idx, test_loader in enumerate(test_loaders[:session_idx + 1]):
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                features = model(inputs, return_features=True)

                all_features.append(features.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

    # Combining features and labels
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # Conduct PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # Visualization
    plt.figure(figsize=(10, 8))

    # Display in different colors for each class
    cmap = plt.cm.get_cmap('tab20', current_classes)
    for i in range(current_classes):
        mask = (labels == i)
        if np.any(mask):
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1],
                        c=[cmap(i)], label=f'Class {i}', alpha=0.7, s=20)

    # Show class center point
    for i in range(current_classes):
        mask = (labels == i)
        if np.any(mask):
            centroid = np.mean(features_pca[mask], axis=0)
            plt.scatter(centroid[0], centroid[1], c='black', marker='*', s=150,
                        edgecolor='w', linewidth=1.5)

    plt.title(f'PCA Visualization after Session {session_idx}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=2)
    plt.tight_layout()
    plt.savefig(f'/content/drive/MyDrive/MotionAwareMICS/results/pca_session_{session_idx}.png', dpi=300)
    plt.close()


# nVAR visualization function (X: session index, Y: nVAR)
def visualize_nVar(nVar_plain, nVar_motion, config):
    sessions = list(range(config.num_sessions + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(sessions, nVar_plain, marker='o', label='Plain MICS nVAR')
    plt.plot(sessions, nVar_motion, marker='s', label='Motion-Aware MICS nVAR')
    plt.title('nVAR Comparison')
    plt.xlabel('Session')
    plt.ylabel('Normalized Variance (nVAR)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('/content/drive/MyDrive/MotionAwareMICS/results/nvar_comparison.png', dpi=300)


def visualize_performance_comparison(plain_results, motion_results, config):
    """Create comprehensive comparison visualizations of MICS vs Motion-Aware MICS"""
    # Directory for saving results
    save_dir = os.path.join(config.save_path, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    # Extract performance metrics
    plain_acc = plain_results['acc']
    motion_acc = motion_results['acc']

    plain_nvar = plain_results['train_nVAR']
    motion_nvar = motion_results['train_nVAR']

    # Session ranges
    sessions = list(range(config.sessions))

    # 1. Accuracy comparison
    plt.figure(figsize=(12, 8))
    plt.plot(sessions, plain_acc, marker='o', linestyle='-', linewidth=2, label='Plain MICS')
    plt.plot(sessions, motion_acc, marker='s', linestyle='-', linewidth=2, label='Motion-Aware MICS')

    # Compute PD
    plain_pd = plain_acc[0] - plain_acc[-1]
    motion_pd = motion_acc[0] - motion_acc[-1]

    # Annotate PD values
    plt.annotate(f'PD: {plain_pd:.2f}%',
                 xy=(sessions[-1], plain_acc[-1]),
                 xytext=(sessions[-1] - 1, plain_acc[-1] - 5),
                 arrowprops=dict(arrowstyle='->'))

    plt.annotate(f'PD: {motion_pd:.2f}%',
                 xy=(sessions[-1], motion_acc[-1]),
                 xytext=(sessions[-1] - 1, motion_acc[-1] + 5),
                 arrowprops=dict(arrowstyle='->'))

    plt.title('Performance Comparison: MICS vs Motion-Aware MICS', fontsize=16)
    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300)

    # 2. nVAR comparison (compactness and separability)
    plt.figure(figsize=(12, 8))
    plt.plot(sessions, plain_nvar, marker='o', linestyle='-', linewidth=2, label='Plain MICS nVAR')
    plt.plot(sessions, motion_nvar, marker='s', linestyle='-', linewidth=2, label='Motion-Aware MICS nVAR')
    plt.title('Feature Compactness and Separability Comparison', fontsize=16)
    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Normalized Variance (nVAR)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(os.path.join(save_dir, 'nvar_comparison.png'), dpi=300)

    # 3. Bar chart for final performance
    plt.figure(figsize=(10, 6))
    methods = ['Plain MICS', 'Motion-Aware MICS']
    final_acc = [plain_acc[-1], motion_acc[-1]]
    pd_values = [plain_pd, motion_pd]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width / 2, final_acc, width, label='Final Accuracy')
    ax.bar(x + width / 2, pd_values, width, label='Performance Degradation')

    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_title('Final Performance Metrics', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.savefig(os.path.join(save_dir, 'final_performance_comparison.png'), dpi=300)

    # Return statistics summary
    return {
        'plain': {
            'final_acc': plain_acc[-1],
            'pd': plain_pd,
            'avg_nvar': np.mean(plain_nvar)
        },
        'motion': {
            'final_acc': motion_acc[-1],
            'pd': motion_pd,
            'avg_nvar': np.mean(motion_nvar)
        }
    }
