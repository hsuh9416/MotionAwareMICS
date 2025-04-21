import os
import math
import datetime
from copy import deepcopy
from data.dataloader.data_utils import *
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def count_mix_acc(logits, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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
        super().__init__(args)
        self.args = args
        self.model = model
        self.results = self.set_acc_table(self.args) # Init the accuracy table
        self.save_path = self.set_save_path() # Save the model

        self.best_model_dict = self.model.state_dict() # Init the best model dict by initial state dict

    def set_acc_table(self, args):
        """ Set the accuracy table of the model """
        results = dict()
        results["acc"] = np.zeros([args.sessions])
        results["acc_base"] = np.zeros([args.sessions])
        results["acc_novel"] = np.zeros([args.sessions])
        results["acc_old"] = np.zeros([args.sessions])
        results["acc_new"] = np.zeros([args.sessions])
        results["acc_base2"] = np.zeros([args.sessions])
        results["acc_novel2"] = np.zeros([args.sessions])
        results["acc_old2"] = np.zeros([args.sessions])
        results["acc_new2"] = np.zeros([args.sessions])

        return results

    def set_save_path(self):
        """ Set the save path of the model """
        time_str = datetime.datetime.now().strftime('%m%d%Y')
        save_path = '%s/%s' % (self.args.dataset, time_str) # e.g.ucf101/04202025
        save_path = os.path.join('results', save_path)  # e.g.results/ucf101/04202025
        os.makedirs(save_path, exist_ok=True)

        return save_path

    def average_embedding(self, trainset, transform):
        """ replace fc.weight with the embedding average of train data """
        model = self.model.eval() # Evaluation mode
        current_mode = model.mode

        # Data
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                  num_workers=8, pin_memory=True, shuffle=False)

        # Replace the transform in the linked data loader with the transform used in the test set.
        trainloader.dataset.transform = transform

        embedding_list = []
        label_list = []

        # Embedding
        with torch.no_grad(): # pure forwarding without gradient
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.mode = 'encoder' # Encoder mode
                embedding = model(data) # Feature extraction

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
            data_index = (label_list == class_index).nonzero() # nonzero: returns the indices of all non-zero elements of input tensor.
            embedding_this = embedding_list[data_index.squeeze(-1)] # [N, 1] -> [N] then embedding for each data point by index
            embedding_mean = embedding_this.mean(0) # Averaging
            proto_list.append(embedding_mean)

        # Update classifier weights by prototypes
        model.fc.weight.data[:self.args.base_class] = torch.stack(proto_list, dim=0) # Convert(stack) to a single torch.tensor

        # Recover the mode of the model
        model.mode = current_mode

        return model

    def get_logits(self, x, fc):
        """A function that calculates the raw score (logit) to be used as a classification result."""
        # Cosine similarity computation
        x = F.linear(F.normalize(x, p=2, dim=1), F.normalize(fc, p=2,dim=1))

        # Temperature scaling is a hyperparameter applied to the softmax function and is used to adjust the output distribution of the model.
        # The lower the temperature, the higher the confidence of the classification.
        # T = 0.1, MICS has the concept of Boundary Thickness, so it has a relatively small value.
        x = x * self.args.temperature

        return x

    def get_optimizer_new(self):
        optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.learning_rate},
                                     {'params': self.model.fc.parameters(), 'lr': 0}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        return optimizer, scheduler

    def get_session_trainable_param_idx(self, model):
        param_dict = dict(model.named_parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_st = math.floor(num_trainable_params * self.args.st_ratio)  # The number of session trainable parameters

        # Remove not trainable parameter from parameter dictionary
        remove_list = [k for k, v in param_dict.items() if not v.requires_grad]
        for k in remove_list:
            param_dict.pop(k)

        # Pre-procession parameter data
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
            param_idx_dict[k] = [np.unravel_index(flattened_idx, param_dict[k].shape)
                                 for flattened_idx in param_flattened_idx_list]
            count += v

        return param_idx_dict

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if self.args.st_ratio < 1:
            P_st_idx = self.get_session_trainable_param_idx(model)
            return model, P_st_idx
        else:
            return model, None

    def update_novel_proto(self, dataloader, class_list):
        model = self.model.eval()

        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = model.encode(data).detach()

        # Update prototype weights
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero()  # nonzero: returns the indices of all non-zero elements of input tensor.
            embedding = data[data_index.squeeze(-1)] # [N, 1] -> [N] then embedding for each data point by index
            proto = embedding.mean(0) # Averaging
            new_fc.append(proto)
            model.fc.weight.data[class_index] = proto

        return model

    def base_train(self):
        # Base session training
        base_optimizer = torch.optim.SGD([
            {'param': self.model.encoder.parameters(), 'lr': self.args.learning_rate}, # Encoder
            {'params': self.model.fc.parameters(), 'lr': self.args.learning_rate}], # Fully connected layer
            momentum=0.9, # Accelerate the slope and suppress the sedation
            nesterov=True, # Move in the corrected direction, move in the momentum direction
            weight_decay=self.args.decay) # L2 regularization

        # Adjusting learning rate: STEP method
        base_scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # Get dataset
        train_set, trainloader, testloader = get_dataloader(self.args, 0)
        self.model = self.average_embedding(train_set, testloader.dataset.transform)

        # Training base session
        max_acc = 0
        best_model_dict = None

        for epoch in range(self.args.epochs_base):
            # For each epoch
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)

                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()

            base_scheduler.step()

        # Evaluation & Save
        tsl, tsa = self.test(self.model, testloader, args, 0)
        self.trlog['max_acc'][0] = float('%.3f' % (tsa * 100))
        save_model_dir = os.path.join(self.save_path, 'session' + str(0 + 1) + '_max_acc.pth')
        torch.save(dict(params=self.model.state_dict()), save_model_dir)

        print('Test acc = {:.2f}%'.format(self.trlog['max_acc'][0]))

    def train(self):
        self.model.load_state_dict(self.best_model_dict)
        self.model.mode = self.args.new_mode # avg_cos

        optimizer, scheduler = self.get_optimizer_new()


        for session in range(1, self.args.sessions):

            print("\nSession: [%d]" % (session + 1))
            train_set, trainloader, testloader = get_dataloader(self.args, session)

            train_transform = deepcopy(trainloader.dataset.transform)
            trainloader.dataset.transform = testloader.dataset.transform
            self.model = self.update_novel_proto(trainloader, np.unique(train_set.targets))

            # Incremental MICS
            trainloader.dataset.transform = train_transform
            for epoch in range(self.args.epochs_new):
                self.model, P_st_idx = self.update_param(self.model, self.best_model_dict)
                tl, ta = self.new_train(self.model, trainloader, optimizer, scheduler, epoch, self.args, P_st_idx, session)

            trainloader.dataset.transform = testloader.dataset.transform
            self.model = self.update_novel_proto(trainloader, np.unique(train_set.targets))

            # Evaluation & Save
            tsl, tsa = self.test(self.model, testloader, self.args, session)
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            save_model_dir = os.path.join(self.save_path, 'session' + str(session + 1) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)

            print('Test acc = {:.2f}%'.format(self.trlog['max_acc'][session]))

        self.save_path = os.path.join(self.save_path, 'last_session.pth')
        print(f"Save last session model to {self.save_path}")
        torch.save(dict(params=self.model.state_dict()), self.save_path)

        print('\n*************** Final results ***************')
        print(self.trlog['max_acc'], '\n')
        self.print_table(self.results, self.args)



    def test(self, model, testloader, args, session):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()

        pred = []
        label = []

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                logits = model(data)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)

                vl.add(loss.item())
                pred.extend(logits)
                label.extend(test_label)

            vl = vl.item()

            # Incremental accuracy table
            pred = torch.stack(pred, 0)
            label = torch.stack(label, 0)

            # Ex. CUB: args.base_class + (session - 1) * args.way = 100 + (N-1) * 10
            pred_base = pred[:, :args.base_class + (session - 1) * args.way]
            pred_novel = pred[:, args.base_class + (session - 1) * args.way:]
            label_novel = label - (args.base_class + (session - 1) * args.way)

            # Ex. CUB: args.base_class = 100
            pred_old = pred[:, :args.base_class]
            pred_new = pred[:, args.base_class:]
            label_new = label - args.base_class

            self.results['acc'][session] = count_acc(pred, label)

            if session == 0:
                top1_base = count_acc(pred, label)
                top1_base2 = top1_base

                top1_novel = 0
                top1_novel2 = 0
            else:
                base_idx = label < args.base_class + (session - 1) * args.way
                novel_idx = label >= args.base_class + (session - 1) * args.way

                top1_base = count_acc(pred[base_idx], label[base_idx])
                top1_base2 = count_acc(pred_base[base_idx], label[base_idx])

                top1_novel = count_acc(pred[novel_idx], label[novel_idx])
                top1_novel2 = count_acc(pred_novel[novel_idx], label_novel[novel_idx])

            self.results['acc_base'][session] = top1_base
            self.results['acc_base2'][session] = top1_base2

            self.results['acc_novel'][session] = top1_novel
            self.results['acc_novel2'][session] = top1_novel2

            old_idx = label < args.base_class
            new_idx = label >= args.base_class

            top1_old = count_acc(pred[old_idx], label[old_idx])
            top1_old2 = count_acc(pred_old[old_idx], label[old_idx])

            if session == 0:
                top1_new = 0
                top1_new2 = 0
            else:
                top1_new = count_acc(pred[new_idx], label[new_idx])
                top1_new2 = count_acc(pred_new[new_idx], label_new[new_idx])

            self.results['acc_old'][session] = top1_old
            self.results['acc_old2'][session] = top1_old2

            self.results['acc_new'][session] = top1_new
            self.results['acc_new2'][session] = top1_new2

        return vl, self.results['acc'][session]

    def print_table(self, results, args):
        print("{:<13}{}".format("Pretraining:", args.base_mode.split('_')[-1]))
        print("{:<13}{}".format("Similarity:", args.new_mode.split('_')[-1]))

        str_head = "{:<9}".format('')
        str_acc = "{:<9}".format('Acc:')
        str_acc_base = "{:<9}".format('Base:')
        str_acc_novel = "{:<9}".format('Novel:')
        str_acc_old = "{:<9}".format('Old:')
        str_acc_new = "{:<9}".format('New:')

        for i in range(len(results['acc'])):
            str_head = str_head + "{:<9}".format('sess' + str(int(i + 1)))
            str_acc = str_acc + "{:<9}".format(str(round(results['acc'][i] * 100.0, 2)) + "%")
            str_acc_base = str_acc_base + "{:<9}".format(str(round(results['acc_base'][i] * 100.0, 2)) + "%")
            str_acc_novel = str_acc_novel + "{:<9}".format(str(round(results['acc_novel'][i] * 100.0, 2)) + "%")
            str_acc_old = str_acc_old + "{:<9}".format(str(round(results['acc_old'][i] * 100.0, 2)) + "%")
            str_acc_new = str_acc_new + "{:<9}".format(str(round(results['acc_new'][i] * 100.0, 2)) + "%")

        print(str_head)
        print(str_acc)
        print(str_acc_base)
        print(str_acc_novel)
        print(str_acc_old)
        print(str_acc_new)
        print('\n')

        str_acc_base2 = "{:<9}".format('Base2:')
        str_acc_novel2 = "{:<9}".format('Novel2:')
        str_acc_old2 = "{:<9}".format('Old2:')
        str_acc_new2 = "{:<9}".format('New2:')

        for i in range(len(results['acc'])):
            str_acc_base2 = str_acc_base2 + "{:<9}".format(str(round(results['acc_base2'][i] * 100.0, 2)) + "%")
            str_acc_novel2 = str_acc_novel2 + "{:<9}".format(str(round(results['acc_novel2'][i] * 100.0, 2)) + "%")
            str_acc_old2 = str_acc_old2 + "{:<9}".format(str(round(results['acc_old2'][i] * 100.0, 2)) + "%")
            str_acc_new2 = str_acc_new2 + "{:<9}".format(str(round(results['acc_new2'][i] * 100.0, 2)) + "%")

        print(str_acc_base2)
        print(str_acc_novel2)
        print(str_acc_old2)
        print(str_acc_new2)
        print('\n')

    def new_train(self, model, trainloader, optimizer, scheduler, epoch, args, P_st_idx=None, session=None):
        tl = Averager()
        ta = Averager()
        bce_loss = torch.nn.BCELoss().cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
        train_class = args.base_class + session * args.way

        model = model.train()
        tqdm_gen = tqdm(trainloader)

        for i, batch in enumerate(tqdm_gen, 1):
            num_loss, loss, acc = 0, 0., 0.
            data, label = [_.cuda() for _ in batch]

            output, retarget = model.forward_mix(args, Variable(data), Variable(label))

            if retarget.shape[1] > softmax(output).shape[1]:
                retarget = retarget[:, :softmax(output).shape[1]]
            elif retarget.shape[1] < softmax(output).shape[1]:
                output = output[:, :retarget.shape[1]]
            loss += bce_loss(softmax(output), retarget)
            acc += count_mix_acc(output, label)[0] * 0.01

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'
                                     .format(epoch, lrc, loss.item(), acc))
            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            updated_param_dict = deepcopy(model.state_dict())  # parameters after update
            model.load_state_dict(self.best_model_dict)
            for k, v in dict(model.named_parameters()).items():
                if v.requires_grad:
                    for idx in P_st_idx[k]:
                        v.data[idx] = updated_param_dict[k].data[idx]

        tl = tl.item()
        ta = ta.item()
        return tl, ta
