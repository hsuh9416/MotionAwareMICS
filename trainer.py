import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data.dataloader.data_utils import get_dataloader
from evaluate import compute_nVar


def accuracy_counting(logits, label):
    """ Accuracy by counting """
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def mix_up_accuracy_counting(logits, label, topk=(1,)):
    """ Accuracy by counting for the mix-up method """
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    acc = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc_value = correct_k.mul_(100.0 / batch_size).item()
        acc.append(acc_value)
    return acc


# Dynamically average the values
class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def val(self):
        return self.v


class MICSTrainer:
    def __init__(self, model, args):
        super().__init__()
        self.args = args
        self.model = model
        self.results = self.set_acc_table()  # Init the accuracy table
        self.save_path = self.args.model_dir  # Save the model here
        self.pre_trained = args.checkpoint if args.dataset == 'cifar100' else None  # Pre-trained wheight for cifar10
        self.best_model_dict = self.set_up_model()  # load pretrained if it existed

    def set_acc_table(self):
        """ Set the accuracy table of the model """
        results = dict()
        results["train_acc"] = [[] for _ in range(self.args.sessions)]
        results["train_loss"] = [[] for _ in range(self.args.sessions)]
        results["train_nVAR"] = np.zeros([self.args.sessions])
        results["test_nVAR"] = np.zeros([self.args.sessions])
        results["acc"] = np.zeros([self.args.sessions])
        results["acc_base"] = np.zeros([self.args.sessions])
        results["acc_novel"] = np.zeros([self.args.sessions])
        results["acc_old"] = np.zeros([self.args.sessions])
        results["acc_new"] = np.zeros([self.args.sessions])
        results["acc_base2"] = np.zeros([self.args.sessions])
        results["acc_novel2"] = np.zeros([self.args.sessions])
        results["acc_old2"] = np.zeros([self.args.sessions])
        results["acc_new2"] = np.zeros([self.args.sessions])

        return results

    def set_up_model(self):
        """ Try load pre_trained model, if not, uses the initial model."""
        if self.pre_trained and os.path.isfile(self.pre_trained):
            print('Loading init parameters from: %s' % self.args.model_dir)
            best_model_dict = dict()
            try:
                temp_model_dict = torch.load(self.pre_trained)['params']
                for key, value in temp_model_dict.items():
                    if 'dummy' not in key:
                        key = key.replace('module.', '')
                        best_model_dict[key] = value
            except:
                temp_model_dict = torch.load(self.args.model_dir)['state_dict']
                for key, value in temp_model_dict.items():
                    if 'backbone' in key:
                        temp_key = 'module.encoder' + key.split('backbone')[1]
                        if 'shortcut' in temp_key:
                            temp_key = temp_key.replace('shortcut', 'downsample')
                        best_model_dict[temp_key] = value
            best_model_dict['fc.weight'] = self.model.fc.weight
            return best_model_dict
        else:
            print("Manually trains base model...")
            return self.model.state_dict()  # Init the best model dict by initial state dict

    def average_embedding(self, trainset, transform):
        """ replace fc.weight with the embedding average of train data """
        model = self.model.eval()  # Evaluation mode
        current_mode = model.mode

        # Dataloader without augmentation
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                   num_workers=8, pin_memory=True, shuffle=False)

        # Replace the transform in the linked data loader with the transform used in the test set.
        train_loader.dataset.transform = transform

        embedding_list = []
        label_list = []

        # Embedding
        with torch.no_grad():  # pure forwarding without gradient
            for i, batch in enumerate(train_loader):
                data, label = [_.cuda() for _ in batch]
                model.mode = 'encoder'  # Encoder mode
                embedding = model(data)  # Feature extraction

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        # Convert(merge) to a single torch.tensor respectively
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        """
            A prototype is a vector representing a particular class, usually the average of the feature vectors of 
            all data points belonging to that class.
            
            p_c = (1/N) * sum(embedding_c)
            
            - p_c: prototype of class c
            - embedding_c: feature vector of class c
            - N: number of data points in class c
        """
        # Calculate prototypes by class
        proto_list = []
        for class_index in range(self.args.base_class):
            data_index = (
                    label_list == class_index).nonzero()  # nonzero: returns all non-zero elements' indices of the input tensor.
            embedding_this = embedding_list[
                data_index.squeeze(-1)]  # [N, 1] -> [N] then embedding for each data point by index
            embedding_mean = embedding_this.mean(0)  # Averaging
            proto_list.append(embedding_mean)

        # Update classifier weights by prototypes
        model.fc.weight.data[:self.args.base_class] = torch.stack(proto_list,
                                                                  dim=0)  # Convert(stack) to a single torch.tensor

        # Recover the mode of the model
        model.mode = current_mode

        return model

    def average_embedding_inc(self, dataloader, class_list):

        data, label = None, None
        model = self.model.eval()

        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = model.encode(data).detach()

        # Update prototype weights
        new_fc = []
        for class_index in class_list:
            data_index = (
                    label == class_index).nonzero()  # nonzero: returns all non-zero elements' indices of the input tensor.
            embedding = data[data_index.squeeze(-1)]  # [N, 1] -> [N] then embedding for each data point by index
            proto = embedding.mean(0)  # Averaging
            new_fc.append(proto)
            model.fc.weight.data[class_index] = proto

        return model

    def get_logits(self, x, fc):
        """A function that calculates the raw score (logit) to be used as a classification result."""
        # Cosine similarity computation
        x = F.linear(F.normalize(x, p=2, dim=1), F.normalize(fc, p=2, dim=1))

        # Temperature scaling is a hyperparameter applied to the softmax function and is used to adjust the output distribution of the model.
        # The lower the temperature, the higher the confidence of the classification.
        # T = 0.1, MICS has the concept of Boundary Thickness, so it has a relatively small value.
        x = x * self.args.temperature

        return x

    def get_optimizer_new(self):
        inc_ir = self.args.inc_learning_rate

        optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': inc_ir},
                                     {'params': self.model.fc.parameters(), 'lr': inc_ir}],
                                    momentum=self.args.momentum, nesterov=True, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_new)
        return optimizer, scheduler

    def get_session_trainable_param_idx(self, model):
        param_dict = dict(model.named_parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_st = math.floor(num_trainable_params * self.args.st_ratio)  # The number of session-trainable parameters

        # Remove not trainable parameter from parameter dictionary
        remove_list = [k for k, v in param_dict.items() if not v.requires_grad]
        for k in remove_list:
            param_dict.pop(k)

        # Pre-processing parameter data
        param_flatten_dict = {}
        param_len_dict = {}
        for k, v in param_dict.items():
            param_dict[k] = param_dict[k].data
            param_flatten_dict[k] = torch.reshape(param_dict[k].detach().data, [-1])
            param_len_dict[k] = param_flatten_dict[k].shape[0]

        # Select the unimportant weight (small absolute value)
        all_weight = np.abs(np.array(sum([list(v.cpu().numpy()) for v in param_flatten_dict.values()], [])))
        sel_weight_idx = all_weight.argsort()[:num_st]

        # Get indices of selected parameters
        count = 0
        param_idx_dict = {}
        for k, v in param_len_dict.items():
            param_flattened_idx_list = [w_idx - count for w_idx in sel_weight_idx
                                        if (w_idx >= count) & (w_idx < count + v)]

            # Multi-dimension indices
            param_idx_dict[k] = [np.unravel_index(flattened_idx, param_dict[k].shape)
                                 for flattened_idx in param_flattened_idx_list]
            count += v

        return param_idx_dict

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # Update
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if self.args.st_ratio < 1:
            # Get the index of the session-trainable parameter
            P_st_idx = self.get_session_trainable_param_idx(model)
            return model, P_st_idx
        else:
            return model, None

    def base_train(self, train_loader, optimizer, scheduler, epoch):
        """ Base session training """
        # Init objects to compute average loss and accuracy
        tot_loss = Averager()
        tot_acc = Averager()

        # Init BCE loss function
        bce_loss = nn.BCEWithLogitsLoss().cuda()  # BCE function
        softmax = nn.Softmax(dim=1).cuda()  # Activation function

        # Set training mode
        model = self.model.train()

        # Init tqdm(progress bar)
        tqdm_gen = tqdm(train_loader, desc='[Base] Epoch 0')

        # For each epoch
        for i, batch in enumerate(tqdm_gen, 1):
            # Init
            loss, acc = 0., 0

            # Midpoint
            data, label = [_.cuda() for _ in batch]
            output, re_label = model.forward_mix_up(self.args, data, label)

            # Synchronize the dimension of a soft label and the ground truth
            if re_label.shape[1] > softmax(output).shape[1]:
                re_label = re_label[:, :softmax(output).shape[1]]
            elif re_label.shape[1] < softmax(output).shape[1]:
                output = output[:, :re_label.shape[1]]

            # Compute BCE loss on raw logits
            # ensure target is float on the same device and clamped to [0,1]
            target = re_label.float().clamp(0, 1).cuda()
            loss += bce_loss(output, target)
            # loss += bce_loss(softmax(output), re_label)

            # Compute accuracy
            acc += mix_up_accuracy_counting(output, label)[0] / 100.0  # percentage

            # Current learning rate
            cur_lr = scheduler.get_last_lr()[0]

            # Process update
            tqdm_gen.set_description(
                'Base Session, epoch {}, lrc={:.4f}, total loss={:.4f} acc={:.4f}'
                .format(epoch + 1, cur_lr, loss.item(), acc)
            )

            # Update average loss and accuracy
            tot_loss.add(loss.item())
            tot_acc.add(acc)

            # Backward propagation and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return tot_acc.val(), tot_loss.val()

    def inc_train(self, model, train_loader, optimizer, scheduler, epoch, args, P_st_idx=None, session=None):
        tot_acc = Averager()
        tot_loss = Averager()
        bce_loss = torch.nn.BCEWithLogitsLoss().cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()

        model = model.train()
        tqdm_gen = tqdm(train_loader)

        for i, batch in enumerate(tqdm_gen, 1):
            loss, acc = 0., 0.
            data, label = [_.cuda() for _ in batch]

            output, re_label = model.forward_mix_up(args, data, label)

            if re_label.shape[1] > softmax(output).shape[1]:
                re_label = re_label[:, :softmax(output).shape[1]]
            elif re_label.shape[1] < softmax(output).shape[1]:
                output = output[:, :re_label.shape[1]]

            # Compute BCE loss on raw logits
            # ensure target is float on the same device and clamped to [0,1]
            target = re_label.float().clamp(0, 1).cuda()
            loss += bce_loss(output, target)
            # loss += bce_loss(softmax(output), re_label)

            # Compute accuracy
            acc += mix_up_accuracy_counting(output, label)[0] / 100.0

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description('Session {}, epoch {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'
                                     .format(session, epoch + 1, lrc, loss.item(), acc))

            tot_acc.add(acc)
            tot_loss.add(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updated_param_dict = deepcopy(model.state_dict())  # parameters after update
            model.load_state_dict(self.best_model_dict)

            if self.args.st_ratio < 1:
                for k, v in dict(model.named_parameters()).items():
                    if v.requires_grad:
                        for idx in P_st_idx[k]:
                            v.data[idx] = updated_param_dict[k].data[idx]

        tot_acc = tot_acc.val()
        tot_loss = tot_loss.val()
        return tot_acc, tot_loss

    def train(self):
        """ Comprehensive training function """
        # 0. Global initialization
        best_acc = 0.0
        best_loss = 0.0

        self.model.load_state_dict(self.best_model_dict)

        # Get dataloader
        train_set, train_loader, test_loader = get_dataloader(self.args, 0)

        if not self.pre_trained:  # Manual training
            ### 1. Base session training ###

            # Optimizer settings - Performance enhancement: Momentum and Nesterov acceleration
            base_optimizer = torch.optim.SGD([
                {'params': self.model.encoder.parameters(), 'lr': self.args.learning_rate},  # Encoder
                {'params': self.model.fc.parameters(), 'lr': self.args.learning_rate}],  # Fully connected layer
                lr=self.args.learning_rate,
                momentum=0.9,  # Speed up the slope and suppress the sedation
                nesterov=True,  # Move in the corrected direction, move in the momentum direction
                weight_decay=0.0005)  # L2 regularization

            # Scheduler settings - Adjusting learning rate: CosineAnnealingLR method
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                base_optimizer,
                T_max=self.args.epochs_base,
                eta_min=0.0)

            for epoch in range(self.args.epochs_base):
                train_acc, train_loss = self.base_train(train_loader, base_optimizer, base_scheduler, epoch)
                print(
                    f'[Train - Base Session: Epoch {epoch}] Accuracy = {round(train_acc, 4)}, Loss = {round(train_loss, 4)}')
                base_scheduler.step()

                self.results["train_acc"][0].append(train_acc)
                self.results["train_loss"][0].append(train_loss)

                # Save the best model
                if train_acc > best_acc:
                    best_acc = train_acc
                    best_loss = train_loss
                    save_model_dir = os.path.join(self.save_path, 'base_session_max_acc.pth')
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    print("Model saved to {}".format(save_model_dir))

            # Compute nVar
            avg_nvar = compute_nVar(self.model, train_loader, self.args.base_class)
            self.results['train_nVAR'][0] = avg_nvar

            print(
                f'[Train - Base session: Final] Accuracy = {round(best_acc, 4)}, Loss = {round(best_loss, 4)}, nVAR = {round(avg_nvar, 4)}')

        # Apply prototype classification to the trained model
        self.model = self.average_embedding(train_set, test_loader.dataset.transform)

        # Evaluation by test
        test_acc, test_loss, test_nvar = self.test(self.model, test_loader, self.args, 0)
        self.results['test_nVAR'][0] = test_nvar
        print(
            f'[Test - Base session] Accuracy = {round(test_acc, 4)}, Loss = {round(test_loss, 4)}, nVAR = {round(test_nvar, 4)}')

        ### 2. Incremental session training ###
        self.model.mode = self.args.new_mode  # avg_cos

        # Optimizer settings - Performance enhancement: Momentum and Nesterov acceleration
        optimizer, scheduler = self.get_optimizer_new()

        for session in range(1, self.args.sessions):
            # Init
            best_acc = 0.0
            best_loss = 0.0

            # Get dataloader
            train_set, train_loader, test_loader = get_dataloader(self.args, session)

            train_transform = deepcopy(train_loader.dataset.transform)  # Copy the transform

            # Replace the transform in the linked data loader with the transform used in the test set
            train_loader.dataset.transform = test_loader.dataset.transform
            self.model = self.average_embedding_inc(train_loader, np.unique(train_set.targets))  # Embedding
            train_loader.dataset.transform = train_transform  # Restore the transform

            # Incremental MICS training
            for epoch in range(self.args.inc_epochs):
                # Update the model parameters
                self.model, P_st_idx = self.update_param(self.model, self.best_model_dict)
                train_acc, train_loss = self.inc_train(self.model, train_loader, optimizer,
                                                       scheduler, epoch, self.args,
                                                       P_st_idx, session)
                print(
                    f'[Train - Increment Session {session}: Epoch {epoch}] Accuracy = {round(train_acc, 4)}, Loss = {round(train_loss, 4)}')

                self.results["train_acc"][session].append(train_acc)
                self.results["train_loss"][session].append(train_loss)

                # Save the best model
                if train_acc > best_acc:
                    best_acc = train_acc
                    best_loss = train_loss
                    save_model_dir = os.path.join(self.save_path, f'inc_session_{session}_max_acc.pth')
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    print("Model saved to {}".format(save_model_dir))

            # Compute nVar
            cur_num_classes = self.args.base_class + session * self.args.way
            avg_nvar = compute_nVar(self.model, train_loader, cur_num_classes)
            self.results['train_nVAR'][session] = avg_nvar

            print(
                f'[Train - Increment session {session}: Final] Accuracy = {round(best_acc, 4)}, Loss = {round(best_loss, 4)}, nVAR = {round(avg_nvar, 4)}')

            # Replace the transform in the linked data loader with the transform used in the test set.
            train_loader.dataset.transform = train_transform
            self.model = self.average_embedding_inc(train_loader, np.unique(train_set.targets))  # Embedding

            # Evaluation by test
            test_acc, test_loss, test_nvar = self.test(self.model, test_loader, self.args, session)
            self.results['test_nVAR'][session] = test_nvar
            print(
                f'[Test - Increment session {session}] Accuracy = {round(test_acc, 4)}, Loss = {round(test_loss, 4)}, nVAR = {round(test_nvar, 4)}')

        # Save the final model
        save_model_dir = os.path.join(self.save_path, f'final_acc.pth')
        torch.save(dict(params=self.model.state_dict()), save_model_dir)

    def test(self, model, test_loader, args, session):
        """ Test session training """
        test_class = args.base_class + session * args.way

        model = model.eval()  # Evaluation mode
        tot_loss = Averager()  # Init average loss

        pred = []
        label = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader, 1):
                data, test_label = [_.cuda() for _ in batch]
                # Compute loss by cross-entropy function
                logits = model(data)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)

                tot_loss.add(loss.item())
                pred.extend(logits)
                label.extend(test_label)

            # Incremental accuracy table
            pred = torch.stack(pred, 0)
            label = torch.stack(label, 0)

            pred_base = pred[:, :args.base_class + (session - 1) * args.way]
            pred_novel = pred[:, args.base_class + (session - 1) * args.way:]
            label_novel = label - (args.base_class + (session - 1) * args.way)

            pred_old = pred[:, :args.base_class]
            pred_new = pred[:, args.base_class:]
            label_new = label - args.base_class

            self.results['acc'][session] = accuracy_counting(pred, label)  # Compute accuracy by counting

            if session == 0:
                top1_base = accuracy_counting(pred, label)
                top1_base2 = top1_base

                top1_novel = 0
                top1_novel2 = 0
            else:
                base_idx = label < args.base_class + (session - 1) * args.way
                novel_idx = label >= args.base_class + (session - 1) * args.way

                top1_base = accuracy_counting(pred[base_idx], label[base_idx])
                top1_base2 = accuracy_counting(pred_base[base_idx], label[base_idx])

                top1_novel = accuracy_counting(pred[novel_idx], label[novel_idx])
                top1_novel2 = accuracy_counting(pred_novel[novel_idx], label_novel[novel_idx])

            self.results['acc_base'][session] = top1_base
            self.results['acc_base2'][session] = top1_base2

            self.results['acc_novel'][session] = top1_novel
            self.results['acc_novel2'][session] = top1_novel2

            old_idx = label < args.base_class
            new_idx = label >= args.base_class

            top1_old = accuracy_counting(pred[old_idx], label[old_idx])
            top1_old2 = accuracy_counting(pred_old[old_idx], label[old_idx])

            if session == 0:
                top1_new = 0
                top1_new2 = 0
            else:
                top1_new = accuracy_counting(pred[new_idx], label[new_idx])
                top1_new2 = accuracy_counting(pred_new[new_idx], label_new[new_idx])

            self.results['acc_old'][session] = top1_old
            self.results['acc_old2'][session] = top1_old2

            self.results['acc_new'][session] = top1_new
            self.results['acc_new2'][session] = top1_new2

            # Compute nVAR
            cur_num_classes = args.base_class + session * args.way
            nvar = compute_nVar(model, test_loader, cur_num_classes)

        return self.results['acc'][session], tot_loss.val(), nvar
