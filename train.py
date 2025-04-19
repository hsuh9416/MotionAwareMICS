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
    scaler = GradScaler('cuda')

    for epoch in range(config.base_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Create sample pairs for mixup
            indices = torch.randperm(inputs.size(0)).to(config.device)
            inputs_b, targets_b = inputs[indices], targets[indices]

            optimizer.zero_grad()

            # Regular training first
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()

            # Apply mixup for a subset of samples
            if np.random.random() < 0.5:  # 50% chance to use mixup
                # Select valid pairs (different classes)
                valid_pairs = []
                for i in range(min(16, inputs.size(0))):  # Limit to avoid memory issues
                    if targets[i] != targets_b[i]:
                        valid_pairs.append(i)

                if valid_pairs:
                    batch_mixed_features = []
                    batch_mixed_labels = []

                    for i in valid_pairs:
                        x1, y1 = inputs[i:i + 1], targets[i:i + 1]
                        x2, y2 = inputs_b[i:i + 1], targets_b[i:i + 1]

                        # Apply manifold mixup
                        with autocast('cuda'):
                            mixed_features, mixed_targets = model.manifold_mixup(
                                x1, x2, y1, y2, 0
                            )
                            batch_mixed_features.append(mixed_features)
                            batch_mixed_labels.append(mixed_targets)

                    # Process all mixup samples together
                    if batch_mixed_features:
                        with autocast('cuda'):
                            mixed_features_tensor = torch.cat(batch_mixed_features, dim=0)
                            mixed_targets_tensor = torch.stack(batch_mixed_labels, dim=0)

                            # Use features directly with classifier
                            logits = F.linear(
                                F.normalize(mixed_features_tensor, p=2, dim=1),
                                F.normalize(model.classifiers, p=2, dim=1)
                            )
                            mixed_outputs = logits / config.temperature

                            # KL divergence loss for soft labels
                            mixup_loss = -torch.sum(
                                F.log_softmax(mixed_outputs, dim=1) * mixed_targets_tensor,
                                dim=1
                            ).mean()

                            # Scale and accumulate gradient
                            scaler.scale(mixup_loss).backward()

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        # Print progress
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # Remove virtual classifiers after base training
    model.cleanup_virtual_classifiers(config.base_classes)

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
    scaler = GradScaler('cuda')

    # Store original classifier data for stability
    original_classifier_data = model.classifiers.data[:current_classes - config.novel_classes_per_session].clone()

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

    # Training loop
    for epoch in range(config.inc_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Create sample pairs for mixup
            indices = torch.randperm(inputs.size(0)).to(config.device)
            inputs_b, targets_b = inputs[indices], targets[indices]

            optimizer.zero_grad()

            # Regular training
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()

            # Apply mixup for some samples
            if np.random.random() < 0.5:
                valid_pairs = []
                for i in range(min(inputs.size(0), 16)):  # Limit to avoid memory issues
                    if targets[i] != targets_b[i]:
                        valid_pairs.append(i)

                if valid_pairs:
                    batch_mixed_features = []
                    batch_mixed_labels = []

                    for i in valid_pairs:
                        x1, y1 = inputs[i:i + 1], targets[i:i + 1]
                        x2, y2 = inputs_b[i:i + 1], targets_b[i:i + 1]

                        # Apply manifold mixup
                        with autocast('cuda'):
                            mixed_features, mixed_targets = model.manifold_mixup(
                                x1, x2, y1, y2, session_idx
                            )
                            batch_mixed_features.append(mixed_features)
                            batch_mixed_labels.append(mixed_targets)

                    # Process all mixup samples together
                    if batch_mixed_features:
                        with autocast('cuda'):
                            mixed_features_tensor = torch.cat(batch_mixed_features, dim=0)
                            mixed_targets_tensor = torch.stack(batch_mixed_labels, dim=0)

                            # Use features directly with classifier
                            logits = F.linear(
                                F.normalize(mixed_features_tensor, p=2, dim=1),
                                F.normalize(model.classifiers, p=2, dim=1)
                            )
                            mixed_outputs = logits / config.temperature

                            # KL divergence loss for soft labels
                            mixup_loss = -torch.sum(
                                F.log_softmax(mixed_outputs, dim=1) * mixed_targets_tensor,
                                dim=1
                            ).mean()

                            # Scale and accumulate gradient
                            scaler.scale(mixup_loss).backward()

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Maintain stability of old class prototypes
            with torch.no_grad():
                # Apply regularization to prevent old classifiers from changing too much
                old_classes = current_classes - config.novel_classes_per_session
                regularization_strength = 0.9  # High value to preserve old knowledge
                mixed_classifiers = (
                        regularization_strength * original_classifier_data +
                        (1 - regularization_strength) * model.classifiers.data[:old_classes]
                )
                model.classifiers.data[:old_classes] = mixed_classifiers

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        # Print progress
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(
            f'Incremental Session {session_idx}, Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # Remove virtual classifiers after training
    model.cleanup_virtual_classifiers(current_classes)

    # Optional: Update classifier prototypes based on features (if enough samples)
    # Only update new class prototypes, keep old ones stable
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
                    class_features[class_idx].append(features[i])

        # Update new class prototypes
        old_classes = current_classes - config.novel_classes_per_session
        for class_idx in range(old_classes, current_classes):
            if class_features[class_idx]:
                # Calculate class centroid from features
                class_prototype = torch.stack(class_features[class_idx]).mean(0)
                # Normalize the prototype
                class_prototype = F.normalize(class_prototype, p=2, dim=0)
                # Update classifier
                model.classifiers.data[class_idx] = class_prototype

    return model