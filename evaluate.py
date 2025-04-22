# Import necessary libraries
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
def visualize_pca(model, test_loaders, current_classes, config, session_idx):
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


# Accuracy/Loss visualization function (X: epochs, Y: nVAR)
def visualize_acc(history_plain, history_motion, config):
    # Visualize accuracy by session
    plt.figure(figsize=(12, 8))

    # Dataset and session information
    sessions = list(range(config.num_sessions + 1))

    # Plain MICS
    plt.plot(sessions, history_plain, marker='o', linestyle='-',
             linewidth=2, markersize=8, label='Plain MICS')

    # Motion-Aware MICS
    plt.plot(sessions, history_motion, marker='s', linestyle='-',
             linewidth=2, markersize=8, label='Motion-Aware MICS')

    # Graph setting
    plt.title('Session-wise Accuracy Comparison', fontsize=18)
    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sessions)

    # Compute performance degradation
    pd_plain = history_plain[0] - history_plain[-1]
    pd_motion = history_motion[0] - history_motion[-1]

    # Annotation
    plt.annotate(f'PD: {pd_plain:.2f}%',
                 xy=(sessions[-1], history_plain[-1]),
                 xytext=(sessions[-1] - 1, history_plain[-1] - 5),
                 arrowprops=dict(arrowstyle='->'))

    plt.annotate(f'PD: {pd_motion:.2f}%',
                 xy=(sessions[-1], history_motion[-1]),
                 xytext=(sessions[-1] - 1, history_motion[-1] + 5),
                 arrowprops=dict(arrowstyle='->'))

    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    # Save results
    plt.savefig('results/mics_comparison_results.png', dpi=300)

    # Performance degradation rate comparison graph
    plt.figure(figsize=(8, 6))

    methods = ['Plain MICS', 'Motion-Aware MICS']
    pd_values = [pd_plain, pd_motion]

    # Create a bar graph
    bars = plt.bar(methods, pd_values, color=['#3498db', '#e74c3c'])

    # Show value above bar
    for bar, val in zip(bars, pd_values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f'{val:.2f}%',
                 ha='center', va='bottom', fontsize=12)

    plt.title('Performance Dropping Rate Comparison', fontsize=16)
    plt.ylabel('Performance Dropping Rate (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save results
    plt.savefig('/content/drive/MyDrive/MotionAwareMICS/results/performance_dropping_rate.png', dpi=300)

    # Compare final accuracy
    plt.figure(figsize=(8, 6))

    final_acc = [history_plain[-1], history_motion[-1]]

    # Create a bar graph
    bars = plt.bar(methods, final_acc, color=['#3498db', '#e74c3c'])

    # Show value above bar
    for bar, val in zip(bars, final_acc):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f'{val:.2f}%',
                 ha='center', va='bottom', fontsize=12)

    plt.title('Final Session Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save results
    plt.savefig('/content/drive/MyDrive/MotionAwareMICS/results/final_accuracy_comparison.png', dpi=300)
