# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


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

    for epoch in range(config.base_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Create sample pairs
            indices = torch.randperm(inputs.size(0)).to(config.device)
            inputs_a, targets_a = inputs, targets
            inputs_b, targets_b = inputs[indices], targets[indices]

            # Identify class pairs to use in the batch
            candidate_pairs = []
            mixup_indices = []

            for i in range(inputs.size(0)):
                if np.random.random() < 0.5 and targets_a[i] != targets_b[i]:
                    class_pair = (int(targets_a[i].item()), int(targets_b[i].item()))
                    # Normalize order
                    class_pair = (min(class_pair), max(class_pair))
                    candidate_pairs.append(class_pair)
                    mixup_indices.append(i)

            # Remove duplicates
            unique_pairs = []
            for pair in candidate_pairs:
                if pair not in unique_pairs and pair not in model.virtual_class_indices:
                    unique_pairs.append(pair)

            # Pre-create all virtual classes
            for class_pair in unique_pairs:
                if class_pair not in model.virtual_class_indices:
                    y1, y2 = class_pair
                    # Create new virtual class index
                    virtual_idx = len(model.classifiers)
                    model.virtual_class_indices[class_pair] = virtual_idx

                    # Calculate midpoint classifier
                    midpoint = model.compute_midpoint_classifier(y1, y2)

                    # Adjust dimensions
                    if len(midpoint.shape) != 2:
                        midpoint = midpoint.view(1, -1)
                    elif len(midpoint.shape) == 2 and midpoint.shape[0] != 1:
                        midpoint = midpoint.unsqueeze(0)

                    # Expand classifiers
                    new_classifiers = nn.Parameter(
                        torch.cat([model.classifiers.data, midpoint], dim=0)
                    ).to(config.device)

                    model.classifiers = new_classifiers

            # Current total class count
            total_classes = len(model.classifiers)

            # Apply mixup (feature extraction and mixing for base images)
            mixed_features = []  # Final features
            mixed_labels = []  # Soft labels

            for i in mixup_indices:
                # Class pair and virtual class index
                y1, y2 = int(targets_a[i].item()), int(targets_b[i].item())
                # Normalize order
                class_pair = (min(y1, y2), max(y1, y2))

                # Sample mixing ratio from beta distribution
                alpha = config.alpha
                lam = np.random.beta(alpha, alpha)

                # Extract features directly from images
                # Run only to middle layer instead of full feature extraction
                x1 = inputs_a[i:i + 1]  # [1, C, H, W]
                x2 = inputs_b[i:i + 1]  # [1, C, H, W]

                # Utilize ResNet20 structure
                layers = list(model.backbone.features.children())

                # Randomly select middle layer
                layer_idx = np.random.randint(1, len(layers) - 1)

                # Forward propagation to middle layer
                h1 = x1
                h2 = x2
                for j in range(layer_idx):
                    h1 = layers[j](h1)
                    h2 = layers[j](h2)

                # Mix features at middle layer
                h_mixed = lam * h1 + (1 - lam) * h2

                # Pass through remaining layers
                for j in range(layer_idx, len(layers)):
                    h_mixed = layers[j](h_mixed)

                # Flatten final features
                feature = torch.flatten(h_mixed, 1)

                # Calculate soft labels
                gamma = config.gamma
                prob_1, prob_v, prob_2 = model.compute_soft_label(lam, gamma)

                # Get virtual class index
                virtual_idx = model.virtual_class_indices[class_pair]

                # Create fixed-size soft label
                soft_label = torch.zeros(total_classes, device=config.device)
                soft_label[y1] = prob_1
                soft_label[y2] = prob_2
                soft_label[virtual_idx] = prob_v

                mixed_features.append(feature)
                mixed_labels.append(soft_label)

            # Train with mixed samples if available
            if mixed_features:
                # Use already processed features here
                mixed_features_tensor = torch.cat(mixed_features, dim=0)
                mixed_targets = torch.stack(mixed_labels, dim=0)

                optimizer.zero_grad()

                # Use mixed precision for forward pass
                with autocast():
                    # Use features directly with classifier (skip backbone pass)
                    logits = F.linear(
                        F.normalize(mixed_features_tensor, p=2, dim=1),
                        F.normalize(model.classifiers, p=2, dim=1)
                    )
                    mixed_outputs = logits / 0.1  # Temperature scaling

                    loss = -torch.sum(F.log_softmax(mixed_outputs, dim=1) * mixed_targets, dim=1).mean()

                # Scale gradients and perform backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            # Also perform regular training on original samples
            optimizer.zero_grad()

            # Use mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
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

    # Remove virtual classifiers and calculate real class prototypes
    num_real_classes = config.base_classes
    model.cleanup_virtual_classifiers(num_real_classes)

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

    # Set optimizer and scheduler
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=config.learning_rate / 10,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.inc_epochs)

    for epoch in range(config.inc_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Create sample pairs
            indices = torch.randperm(inputs.size(0)).to(config.device)
            inputs_a, targets_a = inputs, targets
            inputs_b, targets_b = inputs[indices], targets[indices]

            # Identify class pairs to use in the batch
            candidate_pairs = []
            mixup_indices = []

            for i in range(inputs.size(0)):
                if np.random.random() < 0.5 and targets_a[i] != targets_b[i]:
                    class_pair = (int(targets_a[i].item()), int(targets_b[i].item()))
                    # Normalize order
                    class_pair = (min(class_pair), max(class_pair))
                    candidate_pairs.append(class_pair)
                    mixup_indices.append(i)

            # Remove duplicates
            unique_pairs = []
            for pair in candidate_pairs:
                if pair not in unique_pairs and pair not in model.virtual_class_indices:
                    unique_pairs.append(pair)

            # Pre-create all virtual classes
            for class_pair in unique_pairs:
                if class_pair not in model.virtual_class_indices:
                    y1, y2 = class_pair
                    # Create new virtual class index
                    virtual_idx = len(model.classifiers)
                    model.virtual_class_indices[class_pair] = virtual_idx

                    # Calculate midpoint classifier
                    midpoint = model.compute_midpoint_classifier(y1, y2)

                    # Adjust dimensions
                    if len(midpoint.shape) != 2:
                        midpoint = midpoint.view(1, -1)
                    elif len(midpoint.shape) == 2 and midpoint.shape[0] != 1:
                        midpoint = midpoint.unsqueeze(0)

                    # Expand classifiers
                    new_classifiers = nn.Parameter(
                        torch.cat([model.classifiers.data, midpoint], dim=0)
                    ).to(config.device)

                    model.classifiers = new_classifiers

            # Current total class count
            total_classes = len(model.classifiers)

            # Apply mixup (based on model type)
            mixed_features = []
            mixed_labels = []

            for i in mixup_indices:
                # Class pair and virtual class index
                y1, y2 = int(targets_a[i].item()), int(targets_b[i].item())
                # Normalize order
                class_pair = (min(y1, y2), max(y1, y2))

                # Sample mixing ratio from beta distribution
                alpha = config.alpha
                lam = np.random.beta(alpha, alpha)

                # Feature extraction and mixing based on model type
                if hasattr(model.backbone, 'features'):
                    # ResNet20 (CIFAR-100)
                    x1 = inputs_a[i:i + 1]
                    x2 = inputs_b[i:i + 1]

                    layers = list(model.backbone.features.children())
                    layer_idx = np.random.randint(1, len(layers) - 1)

                    # Forward propagation to middle layer
                    h1 = x1
                    h2 = x2
                    for j in range(layer_idx):
                        h1 = layers[j](h1)
                        h2 = layers[j](h2)

                    # Adjust mixing ratio based on motion recognition
                    if config.use_motion and hasattr(model, 'motion_mixup'):
                        # Simple approach to approximate motion information
                        # Ideally, use model.motion_mixup, but simplified here
                        adjusted_lam = lam
                    else:
                        adjusted_lam = lam

                    # Mix features at middle layer
                    h_mixed = adjusted_lam * h1 + (1 - adjusted_lam) * h2

                    # Pass through remaining layers
                    for j in range(layer_idx, len(layers)):
                        h_mixed = layers[j](h_mixed)

                    # Flatten final features
                    feature = torch.flatten(h_mixed, 1)

                elif isinstance(model.backbone, nn.Module) and len(inputs_a[i:i + 1].shape) == 5:
                    # Video data (UCF101)
                    x1 = inputs_a[i:i + 1]  # [1, C, T, H, W]
                    x2 = inputs_b[i:i + 1]  # [1, C, T, H, W]

                    # Motion recognition activation
                    if config.use_motion and hasattr(model, 'motion_mixup'):
                        # Perform motion-aware mixup
                        mixed_frames, adjusted_lam = model.motion_mixup(x1, x2, lam)
                        lam = adjusted_lam
                    else:
                        # Basic per-frame mixup
                        mixed_frames = lam * x1 + (1 - lam) * x2

                    # Per-frame feature extraction
                    B, C, T, H, W = mixed_frames.shape
                    frame_features = []

                    for t in range(T):
                        frame = mixed_frames[:, :, t]  # [1, C, H, W]
                        if hasattr(model.backbone, 'features'):
                            frame_feat = model.backbone.features(frame)
                            frame_features.append(frame_feat)

                    # Average over time dimension
                    stacked_features = torch.stack(frame_features, dim=2)
                    avg_features = torch.mean(stacked_features, dim=2)
                    feature = torch.flatten(avg_features, 1)

                else:
                    # Other models (standard ResNet18, etc.)
                    x1 = inputs_a[i:i + 1]
                    x2 = inputs_b[i:i + 1]

                    # Basic feature extraction and mixing
                    with torch.no_grad():
                        feat1 = model.backbone(x1)
                        feat2 = model.backbone(x2)

                    feature = lam * feat1 + (1 - lam) * feat2

                # Calculate soft labels
                gamma = config.gamma
                prob_1, prob_v, prob_2 = model.compute_soft_label(lam, gamma)

                # Get virtual class index
                virtual_idx = model.virtual_class_indices[class_pair]

                # Create fixed-size soft label
                soft_label = torch.zeros(total_classes, device=config.device)
                soft_label[y1] = prob_1
                soft_label[y2] = prob_2
                soft_label[virtual_idx] = prob_v

                mixed_features.append(feature)
                mixed_labels.append(soft_label)

            # Train with mixed samples if available
            if mixed_features:
                mixed_features_tensor = torch.cat(mixed_features, dim=0)
                mixed_targets = torch.stack(mixed_labels, dim=0)

                optimizer.zero_grad()

                # Use mixed precision
                with autocast():
                    # Use features directly with classifier
                    logits = F.linear(
                        F.normalize(mixed_features_tensor, p=2, dim=1),
                        F.normalize(model.classifiers, p=2, dim=1)
                    )
                    mixed_outputs = logits / 0.1  # Temperature scaling

                    loss = -torch.sum(F.log_softmax(mixed_outputs, dim=1) * mixed_targets, dim=1).mean()

                # Scale gradients and perform backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            # Regular training on original samples
            optimizer.zero_grad()

            # Use mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        # Print progress
        train_loss = running_loss / len(train_loader)
        print(f'Incremental Session {session_idx}, Epoch: {epoch}, Train Loss: {train_loss:.4f}')

    # Remove virtual classifiers and calculate real class prototypes
    model.cleanup_virtual_classifiers(current_classes)

    # Recalculate prototype-based classifiers
    model.eval()
    prototypes = []

    with torch.no_grad():
        for class_idx in range(current_classes):
            class_samples = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)

                # Select samples of current class
                mask = (targets == class_idx)
                if mask.sum() > 0:
                    class_features = model(inputs[mask], return_features=True)
                    class_samples.append(class_features)

            if class_samples:
                class_prototype = torch.cat(class_samples, dim=0).mean(0)
            else:
                # Maintain prototype from previous session
                class_prototype = model.classifiers[class_idx]

            prototypes.append(class_prototype)

    # Update classifiers
    model.classifiers = nn.Parameter(torch.stack(prototypes, dim=0))

    return model