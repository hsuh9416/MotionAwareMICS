# Import necessary libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.ticker import MaxNLocator


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
def visualize_pca(model, dataloader, current_classes, session_idx, config):
    """Generate 2D PCA visualization of feature space and class distributions"""
    # Directory for saving results
    save_dir = os.path.join(config.save_path, config.dataset)
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_features = []
    all_labels = []

    # Collect features
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()

            # For video data (HMDB51)
            if len(inputs.shape) == 5:
                # Process video data - extract per-frame features and average
                B, C, T, H, W = inputs.shape
                features_list = []

                for t in range(T):
                    frame_features = model(inputs[:, :, t], return_features=True)
                    features_list.append(frame_features)

                features = torch.stack(features_list, dim=1).mean(dim=1)
            else:
                # Regular image data
                features = model(inputs, return_features=True)

            all_features.append(features.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    # Combining features and labels
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # Conduct PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # Visualization with enhanced styling
    plt.figure(figsize=(12, 10))

    # Use a color palette appropriate for the dataset
    if config.dataset == 'cifar100':
        cmap = plt.cm.get_cmap('tab20', current_classes)
        marker_size = 30
        alpha = 0.7
    else:  # HMDB51
        cmap = plt.cm.get_cmap('viridis', current_classes)
        marker_size = 60
        alpha = 0.8

    # Plot each class with different colors
    for i in range(current_classes):
        mask = (labels == i)
        if np.any(mask):
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1],
                        c=[cmap(i)], label=f'Class {i}',
                        alpha=alpha, s=marker_size, edgecolors='w', linewidth=0.5)

    # Plot class centroids
    for i in range(current_classes):
        mask = (labels == i)
        if np.any(mask):
            centroid = np.mean(features_pca[mask], axis=0)
            plt.scatter(centroid[0], centroid[1], c='black', marker='X', s=200,
                        edgecolor='w', linewidth=1.5)

    # Add title and labels with dataset-specific details
    if config.dataset == 'cifar100':
        dataset_name = "CIFAR-100"
    else:
        dataset_name = "HMDB51"

    plt.title(f'Feature Space Visualization ({dataset_name}) - Session {session_idx}',
              fontsize=16, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)

    # Adjust legend based on number of classes
    if current_classes > 15:
        legend_ncol = 3
        legend_loc = 'upper center'
        legend_bbox = (0.5, -0.15)
    else:
        legend_ncol = 2
        legend_loc = 'best'
        legend_bbox = None

    plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox,
               ncol=legend_ncol, fontsize=10, framealpha=0.7)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save the visualization
    filename = f'{config.dataset}_pca_session_{session_idx}.png'
    plt.savefig(os.path.join(config.visual_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


# Visualize nVAR progression across sessions
def visualize_nvar_progression(nvar_values, config):
    """Visualize how nVAR changes across training sessions"""
    # Directory for saving results
    save_dir = os.path.join(config.save_path, config.dataset)
    os.makedirs(save_dir, exist_ok=True)

    sessions = list(range(len(nvar_values)))

    plt.figure(figsize=(10, 6))

    # Set style based on dataset
    if config.dataset == 'cifar100':
        color = 'tab:blue'
        title = 'CIFAR-100: Feature Space Compactness'
    else:  # HMDB51
        color = 'tab:orange'
        title = 'HMDB51: Feature Space Compactness'

    # Plot nVAR progression
    plt.plot(sessions, nvar_values, marker='o', linestyle='-',
             linewidth=2, color=color, markersize=8)

    # Add session markers
    for i, nvar in enumerate(nvar_values):
        if i == 0:
            label = "Base Session"
        else:
            label = f"Incremental Session {i}"

        plt.annotate(label, (i, nvar),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8)

    # Add trendline
    z = np.polyfit(sessions, nvar_values, 1)
    p = np.poly1d(z)
    plt.plot(sessions, p(sessions), "r--", alpha=0.7,
             label=f"Trend (slope: {z[0]:.4f})")

    # Format plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Normalized Variance (nVAR)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Integer x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the visualization
    filename = f'{config.dataset}_nvar_progression.png'
    plt.savefig(os.path.join(config.visual_dir, filename), dpi=300)
    plt.close()


# Visualize class separation in feature space
def visualize_class_separation(nvar_class_values, session_idx, config):
    """Visualize the distribution of nVAR values across classes"""
    if not nvar_class_values:
        return

    # Directory for saving results
    save_dir = os.path.join(config.save_path, config.dataset)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Set style based on dataset
    if config.dataset == 'cifar100':
        color = 'tab:blue'
        title = f'CIFAR-100: Class Separation Distribution (Session {session_idx})'
    else:  # HMDB51
        color = 'tab:orange'
        title = f'HMDB51: Class Separation Distribution (Session {session_idx})'

    # Create histogram of nVAR values
    sns.histplot(nvar_class_values, kde=True, color=color)

    # Add mean and median markers
    mean_val = np.mean(nvar_class_values)
    median_val = np.median(nvar_class_values)

    plt.axvline(mean_val, color='red', linestyle='--',
                label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='green', linestyle='-',
                label=f'Median: {median_val:.4f}')

    # Format plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('nVAR Value (lower is better)', fontsize=14)
    plt.ylabel('Number of Classes', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    # Save the visualization
    filename = f'{config.dataset}_class_separation_session_{session_idx}.png'
    plt.savefig(os.path.join(config.visual_dir, filename), dpi=300)
    plt.close()


# Visualize accuracy per session
def visualize_accuracy_progression(accuracy_values, config):
    """Visualize how accuracy changes across training sessions"""
    # Directory for saving results
    save_dir = os.path.join(config.save_path, config.dataset)
    os.makedirs(save_dir, exist_ok=True)

    sessions = list(range(len(accuracy_values)))

    plt.figure(figsize=(10, 6))

    # Set style based on dataset
    if config.dataset == 'cifar100':
        color = 'tab:blue'
        title = 'CIFAR-100: Performance Progression'
    else:  # HMDB51
        color = 'tab:orange'
        title = 'HMDB51: Performance Progression'

    # Plot accuracy progression
    plt.plot(sessions, accuracy_values, marker='o', linestyle='-',
             linewidth=2, color=color, markersize=8)

    # Calculate performance degradation (PD)
    pd = accuracy_values[0] - accuracy_values[-1]

    # Annotate first and last points
    plt.annotate(f'Base: {accuracy_values[0]:.2f}%',
                 (0, accuracy_values[0]),
                 textcoords="offset points",
                 xytext=(10, 0),
                 ha='left',
                 fontsize=12)

    plt.annotate(f'Final: {accuracy_values[-1]:.2f}%\nPD: {pd:.2f}%',
                 (len(accuracy_values) - 1, accuracy_values[-1]),
                 textcoords="offset points",
                 xytext=(-10, -20),
                 ha='right',
                 fontsize=12)

    # Format plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Session', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Integer x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the visualization
    filename = f'{config.dataset}_accuracy_progression.png'
    plt.savefig(os.path.join(config.visual_dir, filename), dpi=300)
    plt.close()


# Comprehensive analysis of FSCIL performance
def analyze_fscil_performance(results, config):
    """Create a comprehensive analysis dashboard for FSCIL performance"""
    # Directory for saving results
    save_dir = os.path.join(config.save_path, config.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # Extract metrics from results
    acc = results['acc']
    acc_base = results['acc_base']
    acc_novel = results['acc_novel']
    train_nvar = results['train_nVAR']

    sessions = list(range(len(acc)))

    # Create subplots for comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Dataset-specific styling
    if config.dataset == 'cifar100':
        title = "CIFAR-100: Few-Shot Class-Incremental Learning Analysis"
        color_main = 'tab:blue'
        color_base = 'tab:green'
        color_novel = 'tab:red'
        color_nvar = 'tab:purple'
    else:  # HMDB51
        title = "HMDB51: Few-Shot Class-Incremental Learning Analysis"
        color_main = 'tab:orange'
        color_base = 'tab:olive'
        color_novel = 'tab:cyan'
        color_nvar = 'tab:brown'

    # 1. Overall Accuracy
    axes[0, 0].plot(sessions, acc, marker='o', linestyle='-',
                    linewidth=3, color=color_main, markersize=10)

    # Calculate PD and annotate
    pd = acc[0] - acc[-1]
    axes[0, 0].annotate(f'PD: {pd:.2f}%',
                        (sessions[-1], acc[-1]),
                        textcoords="offset points",
                        xytext=(-20, -30),
                        ha='center',
                        fontsize=12,
                        arrowprops=dict(arrowstyle="->", color='black'))

    axes[0, 0].set_title('Overall Accuracy', fontsize=16)
    axes[0, 0].set_xlabel('Session', fontsize=14)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=14)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. Base vs Novel Classes
    axes[0, 1].plot(sessions[1:], acc_base[1:], marker='s', linestyle='-',
                    linewidth=2, color=color_base, markersize=8, label='Base Classes')
    axes[0, 1].plot(sessions[1:], acc_novel[1:], marker='^', linestyle='-',
                    linewidth=2, color=color_novel, markersize=8, label='Novel Classes')

    axes[0, 1].set_title('Base vs. Novel Class Performance', fontsize=16)
    axes[0, 1].set_xlabel('Session', fontsize=14)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=14)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. nVAR Progression
    axes[1, 0].plot(sessions, train_nvar, marker='o', linestyle='-',
                    linewidth=2, color=color_nvar, markersize=8)

    # Add trendline for nVAR
    z = np.polyfit(sessions, train_nvar, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(sessions, p(sessions), "r--", alpha=0.7,
                    label=f"Trend (slope: {z[0]:.4f})")

    axes[1, 0].set_title('Feature Space Compactness (nVAR)', fontsize=16)
    axes[1, 0].set_xlabel('Session', fontsize=14)
    axes[1, 0].set_ylabel('Normalized Variance', fontsize=14)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. Correlation between nVAR and Accuracy
    axes[1, 1].scatter(train_nvar, acc, s=150, c=sessions, cmap='viridis',
                       alpha=0.8, edgecolors='w', linewidth=1)

    # Add labels for each point
    for i, (x, y) in enumerate(zip(train_nvar, acc)):
        axes[1, 1].annotate(f"S{i}", (x, y),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=12)

    # Calculate and display correlation
    correlation = np.corrcoef(train_nvar, acc)[0, 1]
    axes[1, 1].text(0.05, 0.95, f"Correlation: {correlation:.2f}",
                    transform=axes[1, 1].transAxes,
                    fontsize=14,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    axes[1, 1].set_title('nVAR vs. Accuracy Correlation', fontsize=16)
    axes[1, 1].set_xlabel('Normalized Variance (nVAR)', fontsize=14)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=14)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)

    # Add colorbar for session reference
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(sessions) - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1, 1])
    cbar.set_label('Session', fontsize=12)

    # Main title for entire figure
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save comprehensive analysis
    filename = f'{config.dataset}_comprehensive_analysis.png'
    plt.savefig(os.path.join(config.visual_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    # Return summary statistics
    summary = {
        'final_acc': acc[-1],
        'pd': pd,
        'base_final_acc': acc_base[-1] if len(acc_base) > 0 else None,
        'novel_final_acc': acc_novel[-1] if len(acc_novel) > 0 else None,
        'avg_nvar': np.mean(train_nvar),
        'nvar_trend': z[0],
        'acc_nvar_correlation': correlation
    }

    return summary


# Compare motion-aware vs standard approach for HMDB51
def compare_motion_vs_standard(motion_results, standard_results, config):
    """Compare motion-aware MICS with standard MICS for HMDB51 dataset"""
    if config.dataset != 'hmdb51':
        return None

    # Directory for saving results
    save_dir = os.path.join(config.save_path, config.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # Extract performance metrics
    motion_acc = motion_results['acc']
    standard_acc = standard_results['acc']

    motion_nvar = motion_results['train_nVAR']
    standard_nvar = standard_results['train_nVAR']

    sessions = list(range(len(motion_acc)))

    # Create comparison visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    # 1. Accuracy Comparison
    axes[0].plot(sessions, motion_acc, marker='o', linestyle='-',
                 linewidth=3, color='tab:blue', markersize=10,
                 label='Motion-Aware MICS')
    axes[0].plot(sessions, standard_acc, marker='s', linestyle='--',
                 linewidth=3, color='tab:orange', markersize=10,
                 label='Standard MICS')

    # Calculate PD values
    motion_pd = motion_acc[0] - motion_acc[-1]
    standard_pd = standard_acc[0] - standard_acc[-1]

    # Annotate PD values
    axes[0].annotate(f'PD: {motion_pd:.2f}%',
                     xy=(sessions[-1], motion_acc[-1]),
                     xytext=(sessions[-1] - 1, motion_acc[-1] - 5),
                     arrowprops=dict(arrowstyle='->'))

    axes[0].annotate(f'PD: {standard_pd:.2f}%',
                     xy=(sessions[-1], standard_acc[-1]),
                     xytext=(sessions[-1] - 1, standard_acc[-1] + 5),
                     arrowprops=dict(arrowstyle='->'))

    axes[0].set_title('Motion-Aware vs Standard MICS: Accuracy Comparison', fontsize=16)
    axes[0].set_xlabel('Session', fontsize=14)
    axes[0].set_ylabel('Accuracy (%)', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=12)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. nVAR Comparison
    axes[1].plot(sessions, motion_nvar, marker='o', linestyle='-',
                 linewidth=3, color='tab:blue', markersize=10,
                 label='Motion-Aware MICS')
    axes[1].plot(sessions, standard_nvar, marker='s', linestyle='--',
                 linewidth=3, color='tab:orange', markersize=10,
                 label='Standard MICS')

    axes[1].set_title('Motion-Aware vs Standard MICS: nVAR Comparison', fontsize=16)
    axes[1].set_xlabel('Session', fontsize=14)
    axes[1].set_ylabel('Normalized Variance (nVAR)', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(fontsize=12)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    # Save comparison
    filename = 'hmdb51_motion_vs_standard_comparison.png'
    plt.savefig(os.path.join(config.visual_dir, filename), dpi=300)
    plt.close()

    # Calculate improvement stats
    final_acc_improvement = motion_acc[-1] - standard_acc[-1]
    pd_improvement = standard_pd - motion_pd
    nvar_improvement = standard_nvar[-1] - motion_nvar[-1]

    # Return comparison stats
    comparison = {
        'final_acc_improvement': final_acc_improvement,
        'pd_improvement': pd_improvement,
        'nvar_improvement': nvar_improvement,
        'motion_final_acc': motion_acc[-1],
        'standard_final_acc': standard_acc[-1],
        'motion_pd': motion_pd,
        'standard_pd': standard_pd
    }

    return comparison
