import numpy as np
import torch
import matplotlib

matplotlib.use('agg')

def to_one_hot(labels, num_classes):
    """ Converts labels to one-hot vectors."""
    one_hot = torch.zeros(labels.size(0), num_classes).cuda()
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

def get_middle_label(label1, label2, num_base_classes):
    label1 = np.argmax(label1.cpu().numpy(), axis=1)
    label2 = np.argmax(label2.cpu().numpy(), axis=1)

    mix_label_hash = dict()
    middle_label = list()
    counter = 0

    for i in range(num_base_classes):
        mix_label_hash[(i, i)] = counter
        counter += 1

    for i in range(label1.shape[0]):
        target_tuple = (min(label1[i], label2[i]), max(label1[i], label2[i]))

        if target_tuple not in mix_label_hash:
            mix_label_hash[target_tuple] = counter
            counter += 1

        middle_label.append(mix_label_hash[target_tuple])

    mix_label_mask = list()
    for i in range(2):
        mix_label_mask.append([int(key[i]) for key in mix_label_hash.keys()])

    return mix_label_mask, to_one_hot(torch.tensor(middle_label), len(mix_label_hash))

def middle_label_mix_process(label1, label2, num_base_classes, lamb, gamma):
    mix_label_mask, label3 = get_middle_label(label1, label2, num_base_classes)
    if label3.size(1) > label1.size(1):
        zero_stack = torch.zeros([label1.size(0), label3.size(1) - label1.size(1)]).cuda()
    else:
        zero_stack = None

    # Paper 4.1 Construction of Mixup Samples: soft labeling process - Formula (3)
    slope = 1 / (1 - gamma) # 1 / (1 - gamma)
    y1 = np.max((1 - lamb - gamma) * slope, 0) # y1 = max((1 - lambda - gamma)/(1 - gamma), 0)
    y2 = np.max((lamb - gamma) * slope, 0) # y2 = max((lambda - gamma)/(1 - gamma), 0)
    y3 = (1 - y1 - y2) # y3 = 1- y1 - y2

    if zero_stack is not None:
        label = torch.hstack((label1, zero_stack)) * y1 + torch.hstack((label2, zero_stack)) * y2 + label3 * y3
    else:
        label = label1[:, :label3.size(1)] * y1 + label2[:, :label3.size(1)] * y2 + label3 * y3

    return label, mix_label_mask

def middle_mixup_process(out, labels, num_base_classes, lamb, gamma=0.2):
    indices = np.random.permutation(out.size(0))
    out = out * lamb + out[indices] * (1 - lamb)
    target_reweighted, mix_label_mask = middle_label_mix_process(labels, labels[indices], num_base_classes, lamb, gamma)
    return out, target_reweighted, mix_label_mask