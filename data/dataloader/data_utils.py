# MIT License
#
# Copyright (c) 2023 Solang Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This function is based on the code from https://github.com/solangii/MICS
# with modifications to adapt it to the Motion-Aware MICS implementation

import numpy as np
import torch
import data.dataloader.cifar100.cifar as CifarDataset
import data.dataloader.ucf101.ucf101 as UCF101Dataset


def set_up_datasets(args):
    if args.dataset == 'cifar100':
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
        args.Dataset = CifarDataset
    elif args.dataset == 'ucf101':
        args.base_class = 60  # Base classes for UCF101
        args.num_classes = 101
        args.way = 5  # Number of new classes per session
        args.shot = 5  # Number of shots per class
        args.sessions = 8  # Total incremental sessions
        args.Dataset = UCF101Dataset
        args.frames_per_clip = 16  # Number of frames per video clip
        args.step_between_clips = 8  # Step size between clips
        args.fold = 1  # Which fold to use (1, 2, or 3)
    return args


def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader


def get_base_dataloader(args):
    class_index = np.arange(args.base_class)
    is_autoaug = "autoaug" in args.train if hasattr(args, 'train') else False

    if hasattr(args, 'is_autoaug') and args.is_autoaug:
        is_autoaug = True

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True, autoaug=is_autoaug)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True, autoaug=is_autoaug)
    elif args.dataset == 'ucf101':
        trainset = args.Dataset.UCF101Dataset(root=args.dataroot, train=True, download=True,
                                              index=class_index, base_sess=True, autoaug=is_autoaug,
                                              frames_per_clip=args.frames_per_clip,
                                              step_between_clips=args.step_between_clips,
                                              fold=args.fold)
        testset = args.Dataset.UCF101Dataset(root=args.dataroot, train=False, download=False,
                                             index=class_index, base_sess=True, autoaug=is_autoaug,
                                             frames_per_clip=args.frames_per_clip,
                                             step_between_clips=args.step_between_clips,
                                             fold=args.fold)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True,
                                              drop_last=args.drop_last if hasattr(args, 'drop_last') else False)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_new_dataloader(args, session):
    if args.dataset == 'cifar100':
        txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    elif args.dataset == 'ucf101':
        # For UCF101, we'll use numeric class indices for incremental sessions
        start_idx = args.base_class + (session - 1) * args.way
        end_idx = args.base_class + session * args.way
        class_index = np.arange(start_idx, end_idx)

        trainset = args.Dataset.UCF101Dataset(root=args.dataroot, train=True, download=False,
                                              index=class_index, base_sess=False,
                                              frames_per_clip=args.frames_per_clip,
                                              step_between_clips=args.step_between_clips,
                                              fold=args.fold)

    if hasattr(args, 'batch_size_new') and args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        batch_size = args.batch_size_new if hasattr(args, 'batch_size_new') else args.batch_size_base
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    elif args.dataset == 'ucf101':
        testset = args.Dataset.UCF101Dataset(root=args.dataroot, train=False, download=False,
                                             index=class_new, base_sess=False,
                                             frames_per_clip=args.frames_per_clip,
                                             step_between_clips=args.step_between_clips,
                                             fold=args.fold)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_session_classes(args, session):
    class_list = np.arange(args.base_class + session * args.way)
    return class_list