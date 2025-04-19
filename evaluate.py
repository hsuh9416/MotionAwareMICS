# Import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

# Evaluation function
def evaluate(model, test_loaders, config, session):
    model.eval()
    acc_per_session = []

    if session == 0: # First session
        print("Base session evaluation:")

    with torch.no_grad():
        # Evaluate all sessions
        for session_idx, test_loader in enumerate(test_loaders):
            correct = 0
            total = 0

            for inputs, targets in tqdm(test_loader):
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = model(inputs)

                # NCM classification (processed similarly to the existing MICS classification method)
                _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            session_acc = 100. * correct / total
            acc_per_session.append(session_acc)
            print(f'Session {session_idx} Accuracy: {session_acc:.2f}%')

    # Calculating the Performance Dropping Rate
    pd = acc_per_session[0] - acc_per_session[-1]
    print(f'Base Accuracy: {acc_per_session[0]:.2f}%, Final Accuracy: {acc_per_session[-1]:.2f}%')
    print(f'Performance Dropping Rate: {pd:.2f}%')

    return acc_per_session

# Calculating the normalized variance (nVAR)
def compute_nVar(model, test_loaders, current_classes, config):
    model.eval()
    class_features = {i: [] for i in range(current_classes)}

    # Collecting class-specific characteristics
    with torch.no_grad():
        for session_idx, test_loader in enumerate(test_loaders):
            for inputs, targets in tqdm(test_loader):
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                features = model(inputs, return_features=True)

                for i in range(inputs.size(0)):
                    class_idx = targets[i].item()
                    if class_idx < current_classes:  # Consider only the classes learned so far
                        class_features[class_idx].append(features[i].cpu().numpy())

    # class-centric calculation
    class_centroids = {}
    for class_idx in range(current_classes):
        if class_features[class_idx]:
            class_centroids[class_idx] = np.mean(np.stack(class_features[class_idx]), axis=0)

    # Find the nearest center of interference
    nearest_interfering_centroids = {}
    for class_idx in range(current_classes):
        if class_idx not in class_centroids:
            continue

        min_dist = float('inf')
        nearest_idx = None

        for other_idx in range(current_classes):
            if other_idx != class_idx and other_idx in class_centroids:
                dist = np.sum((class_centroids[class_idx] - class_centroids[other_idx]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = other_idx

        if nearest_idx is not None:
            nearest_interfering_centroids[class_idx] = class_centroids[nearest_idx]

    # Compute nVAR
    nVar_values = []
    for class_idx in range(current_classes):
        if class_idx not in class_centroids or class_idx not in nearest_interfering_centroids:
            continue

        centroid = class_centroids[class_idx]
        interfering_centroid = nearest_interfering_centroids[class_idx]

        # Denominator: Square of the distance between the centers
        denominator = np.sum((centroid - interfering_centroid) ** 2)

        # Molecule: Variance of features within a class
        variance_sum = 0
        for feat in class_features[class_idx]:
            variance_sum += np.sum((feat - centroid) ** 2)

        if len(class_features[class_idx]) > 0:
            numerator = variance_sum / len(class_features[class_idx])
            nVar_values.append(numerator / denominator)

    # Average nVAR
    avg_nVar = np.mean(nVar_values)
    return avg_nVar

# PCA visualization function
def visualize_pca(model, test_loaders, current_classes, config, session_idx):
    model.eval()
    all_features = []
    all_labels = []

    # Collect characteristics
    with torch.no_grad():
        for loader_idx, test_loader in enumerate(test_loaders[:session_idx+1]):
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
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

# nVAR visualization function
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

# Accuracy visualization function
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

    # Create bar graph
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

    # Create bar graph
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