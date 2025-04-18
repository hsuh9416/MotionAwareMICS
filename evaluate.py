# Import neccesary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

# 평가 함수
def evaluate(model, test_loaders, current_classes, config):
    model.eval()
    acc_per_session = []

    with torch.no_grad():
        # 지금까지의 모든 세션에 대해 평가
        for session_idx, test_loader in enumerate(test_loaders):
            correct = 0
            total = 0

            for inputs, targets in tqdm(test_loader):
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = model(inputs)

                # NCM 분류 (기존 MICS의 분류 방식과 유사하게 처리)
                _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            session_acc = 100. * correct / total
            acc_per_session.append(session_acc)
            print(f'Session {session_idx} Accuracy: {session_acc:.2f}%')

    # 성능 감소율 (Performance Dropping Rate) 계산
    pd = acc_per_session[0] - acc_per_session[-1]
    print(f'Base Accuracy: {acc_per_session[0]:.2f}%, Final Accuracy: {acc_per_session[-1]:.2f}%')
    print(f'Performance Dropping Rate: {pd:.2f}%')

    return acc_per_session


# 정규화된 분산 (nVAR) 계산
def compute_nvar(model, test_loaders, current_classes, config):
    model.eval()
    class_features = {i: [] for i in range(current_classes)}

    # 클래스별 특성 수집
    with torch.no_grad():
        for session_idx, test_loader in enumerate(test_loaders):
            for inputs, targets in tqdm(test_loader):
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                features = model(inputs, return_features=True)

                for i in range(inputs.size(0)):
                    class_idx = targets[i].item()
                    if class_idx < current_classes:  # 현재까지 학습한 클래스만 고려
                        class_features[class_idx].append(features[i].cpu().numpy())

    # 클래스 중심 계산
    class_centroids = {}
    for class_idx in range(current_classes):
        if class_features[class_idx]:
            class_centroids[class_idx] = np.mean(np.stack(class_features[class_idx]), axis=0)

    # 가장 가까운 방해 중심점 찾기
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

    # nVAR 계산
    nvar_values = []
    for class_idx in range(current_classes):
        if class_idx not in class_centroids or class_idx not in nearest_interfering_centroids:
            continue

        centroid = class_centroids[class_idx]
        interfering_centroid = nearest_interfering_centroids[class_idx]

        # 분모: 중심점 간 거리의 제곱
        denominator = np.sum((centroid - interfering_centroid) ** 2)

        # 분자: 클래스 내 특성의 분산
        variance_sum = 0
        for feat in class_features[class_idx]:
            variance_sum += np.sum((feat - centroid) ** 2)

        if len(class_features[class_idx]) > 0:
            numerator = variance_sum / len(class_features[class_idx])
            nvar_values.append(numerator / denominator)

    # 평균 nVAR
    if nvar_values:
        avg_nvar = np.mean(nvar_values)
        print(f'Normalized Variance (nVAR): {avg_nvar:.4f}')
        return avg_nvar
    else:
        print('Unable to compute nVAR due to insufficient data')
        return None


# PCA 시각화 함수
def visualize_features_pca(model, test_loaders, current_classes, config, session_idx):
    model.eval()
    all_features = []
    all_labels = []

    # 특성 수집
    with torch.no_grad():
        for loader_idx, test_loader in enumerate(test_loaders[:session_idx+1]):
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                features = model(inputs, return_features=True)

                all_features.append(features.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

    # 특성 및 레이블 결합
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # PCA 수행
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # 시각화
    plt.figure(figsize=(10, 8))

    # 각 클래스에 대해 다른 색상으로 표시
    cmap = plt.cm.get_cmap('tab20', current_classes)
    for i in range(current_classes):
        mask = (labels == i)
        if np.any(mask):
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1],
                       c=[cmap(i)], label=f'Class {i}', alpha=0.7, s=20)

    # 클래스 중심점 표시
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
    plt.savefig(f'pca_session_{session_idx}.png', dpi=300)
    plt.close()
