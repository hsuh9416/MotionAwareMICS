# Import neccesary libraries
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from mics import *
from dataloader import *
from evaluate import *
from train import *

#전체 MICS 알고리즘 실행 함수
def run_mics(config):
    # 데이터셋 로드
    if config.dataset == 'cifar100':
        sessions_train_data, sessions_test_data = load_cifar100(
            config.base_classes,
            config.novel_classes_per_session,
            config.num_sessions
        )
    else:  # UCF101 또는 다른 모션 데이터셋
        sessions_train_data, sessions_test_data = load_ucf101(
            config.base_classes,
            config.novel_classes_per_session,
            config.num_sessions,
            config.shots_per_class
        )

    # 데이터 로더 생성
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

    # 모델 초기화
    model = MICS(config, config.base_classes).to(config.device)

    # 기본 세션 훈련
    print("Training base session...")
    model = train_base_session(model, train_loaders[0], config)

    # 기본 세션 평가
    current_classes = config.base_classes
    acc_per_session = evaluate(model, [test_loaders[0]], current_classes, config)
    nvar = compute_nvar(model, [test_loaders[0]], current_classes, config)
    visualize_features_pca(model, [test_loaders[0]], current_classes, config, 0)

    # 각 증분 세션 처리
    for session_idx in range(1, config.num_sessions + 1):
        print(f"\nTraining incremental session {session_idx}...")

        # 분류기 확장
        novel_classes = config.novel_classes_per_session
        expanded_classifiers = nn.Parameter(
            torch.cat([
                model.classifiers.data,
                torch.randn(novel_classes, config.feature_dim).to(config.device)
            ], dim=0)
        )
        model.classifiers = expanded_classifiers
        current_classes += novel_classes

        # 증분 세션 훈련
        model = train_incremental_session(
            model, train_loaders[session_idx], session_idx, current_classes, config
        )

        # 평가
        acc_per_session = evaluate(
            model, test_loaders[:session_idx+1], current_classes, config
        )
        nvar = compute_nvar(
            model, test_loaders[:session_idx+1], current_classes, config
        )
        visualize_features_pca(
            model, test_loaders[:session_idx+1], current_classes, config, session_idx
        )

    return model, acc_per_session


# 메인 실행 코드
def main():
    # 설정
    config = Config()

    # MICS 알고리즘 실행
    print("Running MICS algorithm...")
    model, acc_history = run_mics(config)

    # 모션 인식 활성화 후 다시 실행 (선택적)
    if not config.use_motion:
        print("\nRunning MICS with motion-aware feature adaptation...")
        config.use_motion = True
        model_motion, acc_history_motion = run_mics(config)

        # 결과 비교 시각화
        plt.figure(figsize=(10, 6))
        x = list(range(len(acc_history)))
        plt.plot(x, acc_history, 'o-', label='Standard MICS')
        plt.plot(x, acc_history_motion, 's-', label='Motion-Aware MICS')
        plt.xlabel('Session')
        plt.ylabel('Accuracy (%)')
        plt.title('Comparison of Standard MICS and Motion-Aware MICS')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('mics_comparison.png', dpi=300)
        plt.close()

    print("Experiment completed!")