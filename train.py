# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler


# Train - Base session
def train_base(model, train_loader, config):
    """
    Training function for the base session with mixed precision training.
    Implements MICS algorithm with midpoint interpolation for feature space.

    Args:
        model: MICS model instance
        train_loader: DataLoader for base session training data
        config: Configuration object containing hyperparameters

    Returns:
        Trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.base_epochs)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Get device type for autocast
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(config.base_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Regular training on original samples
            optimizer.zero_grad()

            with autocast(device_type=device_type):  # Fixed: specify device_type
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics for original samples
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Apply mixup on a separate pass
            if np.random.random() < 0.5:  # 50% chance to apply mixup
                # Create sample pairs
                indices = torch.randperm(inputs.size(0)).to(config.device)
                inputs_b, targets_b = inputs[indices], targets[indices]

                # Find valid pairs (different classes)
                valid_pairs = []
                for i in range(min(16, inputs.size(0))):  # Limit to 16 samples
                    if targets[i] != targets_b[i]:
                        valid_pairs.append(i)

                if len(valid_pairs) > 0:
                    # Process one pair at a time to avoid dimension issues
                    for i in valid_pairs:
                        optimizer.zero_grad()

                        x1, y1 = inputs[i:i + 1], targets[i:i + 1]
                        x2, y2 = inputs_b[i:i + 1], targets_b[i:i + 1]

                        # Simplified mixup in feature space
                        with autocast(device_type=device_type):  # Fixed: specify device_type
                            # Extract features separately
                            features1 = model(x1, return_features=True)
                            features2 = model(x2, return_features=True)

                            # Sample mixing ratio
                            alpha = config.alpha
                            lam = np.random.beta(alpha, alpha)

                            # Mix features
                            mixed_features = lam * features1 + (1 - lam) * features2

                            # Normalize mixed features
                            mixed_features = F.normalize(mixed_features, p=2, dim=1)

                            # Get logits using only real classifiers (no virtual classes)
                            real_classifiers = F.normalize(model.classifiers, p=2, dim=1)
                            logits = F.linear(mixed_features, real_classifiers)

                            # Apply temperature scaling
                            logits = logits / config.temperature

                            # Create soft labels (mix the two classes)
                            soft_labels = torch.zeros_like(logits)
                            soft_labels[0, y1.item()] = lam
                            soft_labels[0, y2.item()] = 1 - lam

                            # KL divergence loss
                            mixup_loss = -torch.sum(F.log_softmax(logits, dim=1) * soft_labels)

                        # Scale and backward
                        scaler.scale(mixup_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        running_loss += mixup_loss.item()

        scheduler.step()

        # Print progress
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    return model


# Train - Incremental session
def train_inc(model, train_loader, session_idx, current_classes, config):
    """
    Training function for incremental sessions with mixed precision.
    Only updates a subset of parameters to prevent catastrophic forgetting.

    Args:
        model: MICS model instance
        train_loader: DataLoader for incremental session training data
        session_idx: Current session index
        current_classes: Total number of classes so far
        config: Configuration object containing hyperparameters

    Returns:
        Trained model with updated classifiers
    """
    criterion = nn.CrossEntropyLoss()

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Get device type for autocast
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Store original classifier data for stability
    old_classes = current_classes - config.novel_classes_per_session
    original_classifier_data = model.classifiers.data[:old_classes].clone()

    # Select parameters to update (those with small absolute values)
    backbone_params = []
    for name, param in model.backbone.named_parameters():
        backbone_params.append((name, param))

    # Sort by absolute value
    backbone_params.sort(key=lambda x: x[1].abs().mean().item())

    # Set only epsilon ratio of parameters as trainable
    num_trainable = int(len(backbone_params) * config.epsilon)
    for i, (name, param) in enumerate(backbone_params):
        if i < num_trainable:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Classifiers are always trainable
    model.classifiers.requires_grad = True

    # Define optimizer (only on trainable parameters)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=config.learning_rate / 10,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.inc_epochs)

    for epoch in range(config.inc_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Regular training
            optimizer.zero_grad()

            with autocast(device_type=device_type):  # Fixed: specify device_type
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Apply mixup for some samples (similar simplified approach as in train_base)
            if np.random.random() < 0.5:
                # Create sample pairs
                indices = torch.randperm(inputs.size(0)).to(config.device)
                inputs_b, targets_b = inputs[indices], targets[indices]

                # Find valid pairs (different classes)
                valid_pairs = []
                for i in range(min(inputs.size(0), 16)):  # Limit to 16 samples
                    if targets[i] != targets_b[i]:
                        valid_pairs.append(i)

                if len(valid_pairs) > 0:
                    # Process one pair at a time
                    for i in valid_pairs:
                        optimizer.zero_grad()

                        x1, y1 = inputs[i:i + 1], targets[i:i + 1]
                        x2, y2 = inputs_b[i:i + 1], targets_b[i:i + 1]

                        with autocast(device_type=device_type):  # Fixed: specify device_type
                            # Extract features
                            features1 = model(x1, return_features=True)
                            features2 = model(x2, return_features=True)

                            # Mix features
                            alpha = config.alpha
                            lam = np.random.beta(alpha, alpha)
                            mixed_features = lam * features1 + (1 - lam) * features2

                            # Normalize
                            mixed_features = F.normalize(mixed_features, p=2, dim=1)

                            # Use only real classifiers
                            real_classifiers = F.normalize(model.classifiers, p=2, dim=1)
                            logits = F.linear(mixed_features, real_classifiers)

                            # Apply temperature
                            logits = logits / config.temperature

                            # Create soft labels
                            soft_labels = torch.zeros_like(logits)
                            soft_labels[0, y1.item()] = lam
                            soft_labels[0, y2.item()] = 1 - lam

                            # KL divergence loss
                            mixup_loss = -torch.sum(F.log_softmax(logits, dim=1) * soft_labels)

                        # Scale and backward
                        scaler.scale(mixup_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        running_loss += mixup_loss.item()

            # Maintain stability of old class prototypes (after each batch)
            with torch.no_grad():
                # Strong regularization to preserve old class knowledge
                reg_strength = 0.9  # 90% old, 10% new updates
                mixed_classifiers = (
                        reg_strength * original_classifier_data +
                        (1 - reg_strength) * model.classifiers.data[:old_classes]
                )
                model.classifiers.data[:old_classes] = mixed_classifiers

        scheduler.step()

        # Print progress
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(
            f'Incremental Session {session_idx}, Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # Update classifier prototypes based on features
    if config.shots_per_class >= 5:  # Only if we have enough samples
        model.eval()
        class_features = {i: [] for i in range(current_classes)}

        # Collect features by class
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                features = model(inputs, return_features=True)

                for i in range(inputs.size(0)):
                    class_idx = targets[i].item()
                    if class_idx < current_classes:  # Ensure valid class index
                        class_features[class_idx].append(features[i])

        # Update new class prototypes only
        for class_idx in range(old_classes, current_classes):
            if class_features[class_idx]:
                # Compute centroid
                class_prototype = torch.stack(class_features[class_idx]).mean(0)
                # Normalize
                class_prototype = F.normalize(class_prototype, p=2, dim=0)
                # Update
                model.classifiers.data[class_idx] = class_prototype

    return model