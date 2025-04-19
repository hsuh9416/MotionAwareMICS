# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

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

            # 샘플 쌍 생성 및 Manifold Mixup 적용
            indices = torch.randperm(inputs.size(0)).to(config.device)
            inputs_a, targets_a = inputs, targets
            inputs_b, targets_b = inputs[indices], targets[indices]

            # 각 샘플 쌍에 대해 50% 확률로 mixup 적용
            mixed_samples = []
            mixed_labels = []

            for i in range(inputs.size(0)):
                if np.random.random() < 0.5 and targets_a[i] != targets_b[i]:
                    # Manifold mix-up
                    mixed_x, soft_label = model.manifold_mixup(
                        inputs_a[i:i+1], inputs_b[i:i+1],
                        targets_a[i:i+1], targets_b[i:i+1],
                        session_idx=0
                    )
                    mixed_samples.append(mixed_x)
                    mixed_labels.append(soft_label)

            # In case of mix-up samples
            if mixed_samples:
                mixed_inputs = torch.cat(mixed_samples, dim=0)
                mixed_targets = torch.stack(mixed_labels, dim=0)

                optimizer.zero_grad()
                mixed_outputs = model(mixed_inputs)
                loss = -torch.sum(F.log_softmax(mixed_outputs, dim=1) * mixed_targets, dim=1).mean()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 원본 샘플에 대한 일반 훈련도 수행
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

        # Print training Result
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # 가상 분류기 제거 및 실제 클래스 프로토타입 계산
    num_real_classes = config.base_classes
    model.cleanup_virtual_classifiers(num_real_classes)

    return model

# Train - Incremental session
def train_inc(model, train_loader, session_idx, current_classes, config):
    criterion = nn.CrossEntropyLoss()

    # 업데이트할 파라미터 선택 (절대값이 작은 파라미터)
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

            # 샘플 쌍 생성 및 Manifold Mixup 적용 (기본 세션과 유사)
            indices = torch.randperm(inputs.size(0)).to(config.device)
            inputs_a, targets_a = inputs, targets
            inputs_b, targets_b = inputs[indices], targets[indices]

            mixed_samples = []
            mixed_labels = []

            for i in range(inputs.size(0)):
                if np.random.random() < 0.5 and targets_a[i] != targets_b[i]:
                    mixed_x, soft_label = model.manifold_mixup(
                        inputs_a[i:i+1], inputs_b[i:i+1],
                        targets_a[i:i+1], targets_b[i:i+1],
                        session_idx=session_idx
                    )
                    mixed_samples.append(mixed_x)
                    mixed_labels.append(soft_label)

            if mixed_samples:
                mixed_inputs = torch.cat(mixed_samples, dim=0)
                mixed_targets = torch.stack(mixed_labels, dim=0)

                optimizer.zero_grad()
                mixed_outputs = model(mixed_inputs)
                loss = -torch.sum(F.log_softmax(mixed_outputs, dim=1) * mixed_targets, dim=1).mean()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 원본 샘플에 대한 일반 훈련
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Print training Result
        train_loss = running_loss / len(train_loader)
        print(f'Incremental Session {session_idx}, Epoch: {epoch}, Train Loss: {train_loss:.4f}')

    # 가상 분류기 제거 및 실제 클래스 프로토타입 계산
    model.cleanup_virtual_classifiers(current_classes)

    # 프로토타입 기반 분류기 재계산
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

    # Update classifiers
    model.classifiers = nn.Parameter(torch.stack(prototypes, dim=0))

    return model