import torch

# 설정 클래스
class Config:
    # 기본 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # 데이터셋 설정
    dataset = 'cifar100'  # 'cifar100' 또는 'kinetics'
    base_classes = 60     # 기본 클래스 수
    novel_classes_per_session = 5  # 세션당 새로운 클래스 수
    num_sessions = 8      # 증분 세션 수
    shots_per_class = 5   # 각 증분 클래스당 샘플 수

    # 모델 설정
    backbone = 'resnet18'
    feature_dim = 512     # ResNet18의 특성 벡터 차원

    # 훈련 설정
    batch_size = 128
    base_epochs = 100
    inc_epochs = 10
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # MICS 설정
    alpha = 1.0  # 베타 분포의 파라미터
    gamma = 0.4  # 소프트 라벨링 정책의 파라미터
    epsilon = 0.3  # 증분 단계에서 업데이트할 파라미터의 비율

    # 모션 인식 설정
    use_motion = False  # 모션 인식 기능 활성화 여부
    flow_alpha = 0.5    # 광학 흐름 가중치
