# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from feature_extractor import ResNet20Backbone, ResNet18Backbone

# Train - Base session
def train_base(model, train_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.base_epochs)

    for epoch in range(config.base_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # 샘플 쌍 생성
            indices = torch.randperm(inputs.size(0)).to(config.device)
            inputs_a, targets_a = inputs, targets
            inputs_b, targets_b = inputs[indices], targets[indices]

            # 1. 배치 내에서 사용할 클래스 쌍 미리 파악
            candidate_pairs = []
            mixup_indices = []

            for i in range(inputs.size(0)):
                if np.random.random() < 0.5 and targets_a[i] != targets_b[i]:
                    class_pair = (int(targets_a[i].item()), int(targets_b[i].item()))
                    # 순서를 정규화하여 (작은 값, 큰 값) 형태로 저장
                    class_pair = (min(class_pair), max(class_pair))
                    candidate_pairs.append(class_pair)
                    mixup_indices.append(i)

            # 중복 제거
            unique_pairs = []
            for pair in candidate_pairs:
                if pair not in unique_pairs and pair not in model.virtual_class_indices:
                    unique_pairs.append(pair)

            # 2. 미리 모든 가상 클래스 생성 (배치 전처리)
            for class_pair in unique_pairs:
                if class_pair not in model.virtual_class_indices:
                    y1, y2 = class_pair
                    # 새 가상 클래스 인덱스 생성
                    virtual_idx = len(model.classifiers)
                    model.virtual_class_indices[class_pair] = virtual_idx

                    # 중간점 분류기 계산
                    midpoint = model.compute_midpoint_classifier(y1, y2)

                    # 차원 맞추기
                    if len(midpoint.shape) != 2:
                        midpoint = midpoint.view(1, -1)
                    elif len(midpoint.shape) == 2 and midpoint.shape[0] != 1:
                        midpoint = midpoint.unsqueeze(0)

                    # 분류기 확장
                    new_classifiers = nn.Parameter(
                        torch.cat([model.classifiers.data, midpoint], dim=0)
                    ).to(config.device)

                    model.classifiers = new_classifiers

            # 3. 현재 총 클래스 수 (실제 + 가상)
            total_classes = len(model.classifiers)

            # 4. 믹스업 적용
            mixed_samples = []
            mixed_labels = []

            for i in mixup_indices:
                # 클래스 쌍 및 가상 클래스 인덱스
                y1, y2 = int(targets_a[i].item()), int(targets_b[i].item())
                # 순서 정규화
                class_pair = (min(y1, y2), max(y1, y2))

                # 베타 분포에서 혼합 비율 샘플링
                alpha = config.alpha
                lam = np.random.beta(alpha, alpha)

                # Manifold Mixup 수행
                if isinstance(model.backbone, ResNet20Backbone):
                    # ResNet20의 내부 구조에 접근
                    layers = list(model.backbone.features.children())

                    # 첫 번째/두 번째 부분 분리
                    layer_idx = np.random.randint(1, len(layers) - 1)
                    first_part = nn.Sequential(*layers[:layer_idx])
                    second_part = nn.Sequential(*layers[layer_idx:])

                    # 중간 레이어 특성 추출
                    h1 = first_part(inputs_a[i:i + 1])
                    h2 = first_part(inputs_b[i:i + 1])

                    # 특성 혼합
                    h_mixed = lam * h1 + (1 - lam) * h2

                    # 최종 특성 추출
                    mixed_x = second_part(h_mixed)
                    mixed_x = torch.flatten(mixed_x, 1)
                else:
                    # 다른 백본 모델에 대한 처리
                    # (여기서는 간략화를 위해 생략)
                    continue

                # 소프트 라벨 계산
                gamma = config.gamma
                prob_1, prob_v, prob_2 = model.compute_soft_label(lam, gamma)

                # 가상 클래스 인덱스 가져오기
                virtual_idx = model.virtual_class_indices[class_pair]

                # 고정된 크기의 소프트 라벨 생성
                soft_label = torch.zeros(total_classes, device=config.device)
                soft_label[y1] = prob_1
                soft_label[y2] = prob_2
                soft_label[virtual_idx] = prob_v

                mixed_samples.append(mixed_x)
                mixed_labels.append(soft_label)

            # 5. 혼합 샘플이 있는 경우 학습
            if mixed_samples:
                mixed_inputs = torch.cat(mixed_samples, dim=0)
                # 이제 모든 텐서가 동일한 크기를 가짐
                mixed_targets = torch.stack(mixed_labels, dim=0)

                optimizer.zero_grad()
                mixed_outputs = model(mixed_inputs)
                loss = -torch.sum(F.log_softmax(mixed_outputs, dim=1) * mixed_targets, dim=1).mean()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 6. 원본 샘플에 대한 일반 훈련도 수행
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        # 진행 상황 출력
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # 7. 가상 분류기 제거 및 실제 클래스 프로토타입 계산
    num_real_classes = config.base_classes
    model.cleanup_virtual_classifiers(num_real_classes)

    return model


# Train - Incremental session
def train_inc(model, train_loader, session_idx, current_classes, config):
    criterion = nn.CrossEntropyLoss()

    # 1. 업데이트할 파라미터 선택 (절대값이 작은 파라미터)
    backbone_params = []
    for name, param in model.backbone.named_parameters():
        backbone_params.append((name, param))

    # 절대값 기준 정렬
    backbone_params.sort(key=lambda x: x[1].abs().mean().item())

    # epsilon 비율의 파라미터만 학습 가능하도록 설정
    num_trainable = int(len(backbone_params) * config.epsilon)
    for i, (name, param) in enumerate(backbone_params):
        if i < num_trainable:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 분류기는 항상 학습 가능
    model.classifiers.requires_grad = True

    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=config.learning_rate / 10,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.inc_epochs)

    for epoch in range(config.inc_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # 2. 샘플 쌍 생성
            indices = torch.randperm(inputs.size(0)).to(config.device)
            inputs_a, targets_a = inputs, targets
            inputs_b, targets_b = inputs[indices], targets[indices]

            # 3. 배치 내에서 사용할 클래스 쌍 미리 파악
            candidate_pairs = []
            mixup_indices = []

            for i in range(inputs.size(0)):
                if np.random.random() < 0.5 and targets_a[i] != targets_b[i]:
                    class_pair = (int(targets_a[i].item()), int(targets_b[i].item()))
                    # 순서를 정규화하여 (작은 값, 큰 값) 형태로 저장
                    class_pair = (min(class_pair), max(class_pair))
                    candidate_pairs.append(class_pair)
                    mixup_indices.append(i)

            # 중복 제거
            unique_pairs = []
            for pair in candidate_pairs:
                if pair not in unique_pairs and pair not in model.virtual_class_indices:
                    unique_pairs.append(pair)

            # 4. 미리 모든 가상 클래스 생성
            for class_pair in unique_pairs:
                if class_pair not in model.virtual_class_indices:
                    y1, y2 = class_pair
                    virtual_idx = len(model.classifiers)
                    model.virtual_class_indices[class_pair] = virtual_idx

                    midpoint = model.compute_midpoint_classifier(y1, y2)

                    if len(midpoint.shape) != 2:
                        midpoint = midpoint.view(1, -1)
                    elif len(midpoint.shape) == 2 and midpoint.shape[0] != 1:
                        midpoint = midpoint.unsqueeze(0)

                    new_classifiers = nn.Parameter(
                        torch.cat([model.classifiers.data, midpoint], dim=0)
                    ).to(config.device)

                    model.classifiers = new_classifiers

            # 5. 현재 총 클래스 수
            total_classes = len(model.classifiers)

            # 6. 믹스업 적용
            mixed_samples = []
            mixed_labels = []

            for i in mixup_indices:
                # 클래스 쌍 및 가상 클래스 인덱스
                y1, y2 = int(targets_a[i].item()), int(targets_b[i].item())
                # 순서 정규화
                class_pair = (min(y1, y2), max(y1, y2))

                # 베타 분포에서 혼합 비율 샘플링
                alpha = config.alpha
                lam = np.random.beta(alpha, alpha)

                # 모델 종류에 따른 Manifold Mixup 처리
                if isinstance(model.backbone, ResNet20Backbone):
                    # ResNet20의 경우
                    layers = list(model.backbone.features.children())
                    layer_idx = np.random.randint(1, len(layers) - 1)
                    first_part = nn.Sequential(*layers[:layer_idx])
                    second_part = nn.Sequential(*layers[layer_idx:])

                    h1 = first_part(inputs_a[i:i + 1])
                    h2 = first_part(inputs_b[i:i + 1])

                    # 모션 인식 활성화 여부에 따라 혼합 비율 조정
                    if config.use_motion and hasattr(model, 'motion_mixup'):
                        # 모션 정보 활용 (간략화된 버전)
                        h_mixed = lam * h1 + (1 - lam) * h2
                    else:
                        h_mixed = lam * h1 + (1 - lam) * h2

                    mixed_x = second_part(h_mixed)
                    mixed_x = torch.flatten(mixed_x, 1)

                elif isinstance(model.backbone, ResNet18Backbone) and len(inputs_a[i:i + 1].shape) == 5:
                    # 비디오 데이터의 경우 (ResNet18)
                    x1, x2 = inputs_a[i:i + 1], inputs_b[i:i + 1]

                    # 모션 인식 활성화 여부
                    if config.use_motion and hasattr(model, 'motion_mixup'):
                        # 모션 인식 기반 믹스업 수행
                        mixed_frames, adjusted_lam = model.motion_mixup(x1, x2, lam)
                        lam = adjusted_lam
                    else:
                        # 기본 프레임 단위 믹스업
                        mixed_frames = lam * x1 + (1 - lam) * x2

                    # 특성 추출 (모델 forward 필요)
                    mixed_x = model.backbone(mixed_frames)

                else:
                    # 다른 모델 처리 (일반 ResNet18 등)
                    # 일반 이미지 데이터
                    resnet_layers = list(model.backbone.features.children())
                    layer_idx = np.random.randint(1, len(resnet_layers) - 1)
                    first_part = nn.Sequential(*resnet_layers[:layer_idx])
                    second_part = nn.Sequential(*resnet_layers[layer_idx:])

                    h1 = first_part(inputs_a[i:i + 1])
                    h2 = first_part(inputs_b[i:i + 1])

                    h_mixed = lam * h1 + (1 - lam) * h2

                    mixed_x = second_part(h_mixed)
                    mixed_x = torch.flatten(mixed_x, 1)

                # 소프트 라벨 계산
                gamma = config.gamma
                prob_1, prob_v, prob_2 = model.compute_soft_label(lam, gamma)

                # 가상 클래스 인덱스 가져오기
                virtual_idx = model.virtual_class_indices[class_pair]

                # 고정된 크기의 소프트 라벨 생성
                soft_label = torch.zeros(total_classes, device=config.device)
                soft_label[y1] = prob_1
                soft_label[y2] = prob_2
                soft_label[virtual_idx] = prob_v

                mixed_samples.append(mixed_x)
                mixed_labels.append(soft_label)

            # 7. 혼합 샘플이 있는 경우 학습
            if mixed_samples:
                mixed_inputs = torch.cat(mixed_samples, dim=0)
                mixed_targets = torch.stack(mixed_labels, dim=0)

                optimizer.zero_grad()
                mixed_outputs = model(mixed_inputs)
                loss = -torch.sum(F.log_softmax(mixed_outputs, dim=1) * mixed_targets, dim=1).mean()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 8. 원본 샘플에 대한 일반 훈련
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # 진행 상황 출력
        train_loss = running_loss / len(train_loader)
        print(f'Incremental Session {session_idx}, Epoch: {epoch}, Train Loss: {train_loss:.4f}')

    # 9. 가상 분류기 제거 및 실제 클래스 프로토타입 계산
    model.cleanup_virtual_classifiers(current_classes)

    # 10. 프로토타입 기반 분류기 재계산
    model.eval()
    prototypes = []

    with torch.no_grad():
        for class_idx in range(current_classes):
            class_samples = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)

                # 현재 클래스 샘플 선택
                mask = (targets == class_idx)
                if mask.sum() > 0:
                    class_features = model(inputs[mask], return_features=True)
                    class_samples.append(class_features)

            if class_samples:
                class_prototype = torch.cat(class_samples, dim=0).mean(0)
            else:
                # 이전 세션에서 계산된 프로토타입 유지
                class_prototype = model.classifiers[class_idx]

            prototypes.append(class_prototype)

    # 분류기 업데이트
    model.classifiers = nn.Parameter(torch.stack(prototypes, dim=0))

    return model