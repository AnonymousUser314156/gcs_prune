# -*- coding: utf-8 -*-

"""
Created on 11/17/2021
main_prune_gcs.
@author: ***
"""
# -*- coding: utf-8 -*-


import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from models.model_base import ModelBase
# from tensorboardX import SummaryWriter
# from tqdm import tqdm
from models.base.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
# from utils.data_utils import get_dataloader
from utils.data_utils import get_transforms
from utils.network_utils import get_network

import multiprocessing

# torch tpu & google colab

# PyTorch/XLA GPU Setup (only if GPU runtime)
if os.environ.get('COLAB_GPU', '0') == '1':
    os.environ['GPU_NUM_DEVICES'] = '1'
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda/'

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


def init_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='configs/cifar10/resnet32/GSS_98.json')
    parser.add_argument('--config', type=str, default='configs/cifar10/vgg19/GSS_98.json')
    # parser.add_argument('--config', type=str, default='configs/mnist/lenet/GSS_98.json')
    parser.add_argument('--mask_path', type=str, default='runs/storage_mask/cifar10_vgg19_prune_ratio_98_tpu_test')
    parser.add_argument('--run', type=str, default='expsim1')
    parser.add_argument('--epoch', type=str, default='666')
    parser.add_argument('--batch_size', type=int, default=666)
    parser.add_argument('--data_mode', type=int, default=1)
    parser.add_argument('--grad_mode', type=int, default=2)
    parser.add_argument('--prune_mode', type=int, default=2)
    parser.add_argument('--num_group', type=int, default=666)
    parser.add_argument('--remain', type=float, default=666)
    parser.add_argument('--lr_mode', type=str, default='cosine', help='cosine or preset')
    parser.add_argument('--l2', type=str, default='666')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--dp', type=str, default='../Data', help='dataset path')
    args = parser.parse_args()

    runs = None
    if len(args.run) > 0:
        runs = args.run
    config = process_config(args.config, runs)
    if args.epoch != '666':
        config.epoch = args.epoch
        print("set new epoch:{}".format(config.epoch))
    if args.batch_size != 666:
        config.batch_size = args.batch_size
        print("set new batch_size:{}".format(config.batch_size))
    if args.remain != 666:
        config.target_ratio = (100 - args.remain) / 100.0
        print("set new target_ratio:{}".format(config.target_ratio))
    if args.l2 != '666':
        config.weight_decay = args.l2
        print("set new weight_decay:{}".format(config.weight_decay))
    if args.num_group != 666:
        if 10 % args.num_group == 0:
            config.num_group = args.num_group
            print("set label group:{}".format(config.num_group))
        else:
            config.num_group = None
            print("num_group must be divisible by the number of tags")
    else:
        config.num_group = None
    config.mask_path = args.mask_path
    config.data_mode = args.data_mode
    config.grad_mode = args.grad_mode
    config.prune_mode = args.prune_mode
    config.lr_mode = args.lr_mode
    config.debug = args.debug
    config.dp = args.dp
    config.send_mail_str = ('Wubba lubba dub dub\n')
    config.best_acc = 0
    config.best_epoch = 0

    return config


def print_mask_information(mb, logger):
    ratios = mb.get_ratio_at_each_layer()
    # logger.info('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    re_str = '** Mask information of %s. Overall Remaining: %.2f%%\n' % (mb.get_name(), ratios['ratio'])
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        # logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
        re_str += '  (%d) %.2f%%\n' % (count, v)
        count += 1
    return re_str


config = init_config()

# ============ load state =============
if config.prune_mode != 0:
    state = torch.load(config.mask_path, map_location='cpu')
    if 'args' in state.keys():
        print(state['args'])
        # config = state['args']
    print("=> load state finish")

# --- use tpu ---
SERIAL_EXEC = xmp.MpSerialExecutor()

# build model
model_cpu = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', True))
# Only instantiate model weights once in memory.
WRAPPED_MODEL = xmp.MpModelWrapper(model_cpu)


def train(net, loader, optimizer, criterion, lr_scheduler, epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # import pdb; pdb.set_trace()
        loss.backward()
        xm.optimizer_step(optimizer)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # if batch_idx % 50 == 0:
    if config.lr_mode == 'cosine':
        _lr = lr_scheduler.get_last_lr()
    elif 'preset' in config.lr_mode:
        lr_scheduler(optimizer, epoch)
        _lr = lr_scheduler.get_lr(optimizer)
    desc = ('[xla:%d] => [LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (xm.get_ordinal(), _lr, train_loss / (batch_idx + 1), 100. * correct / total,
             correct, total))
    print('Epoch: %d' % epoch, desc, flush=True)


def test(net, loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        desc = ('[xla:%d] => Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (xm.get_ordinal(), test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        print(desc, flush=True)

    # Save checkpoint.
    acc = 100. * correct / total

    return acc


def train_once(net, trainloader, testloader, learning_rate, weight_decay, num_epochs, device):
    global config

    acc_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    if config.lr_mode == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)
    elif 'preset' in config.lr_mode:
        lr_schedule = {0: learning_rate,
                       int(num_epochs * 0.5): learning_rate * 0.1,
                       int(num_epochs * 0.75): learning_rate * 0.01}
        lr_scheduler = PresetLRScheduler(lr_schedule)
    else:
        print('===!!!=== Wrong learning rate decay setting! ===!!!===')
        exit()

    for epoch in range(num_epochs):
        train_loader = pl.ParallelLoader(trainloader, [device])
        test_loader = pl.ParallelLoader(testloader, [device])
        train(net, train_loader.per_device_loader(device), optimizer, criterion, lr_scheduler, epoch)
        test_acc = test(net, test_loader.per_device_loader(device), criterion)
        if config.lr_mode == 'cosine':
            lr_scheduler.step()

        if test_acc > config.best_acc:
            config.best_acc = test_acc
            config.best_epoch = epoch
        acc_list.append(test_acc)

    return acc_list


def run_tpu():
    # preprocessing
    # ====================================== get dataloader ======================================
    num_workers = 0
    root = config.dp

    def get_dataset():
        transform_train, transform_test = get_transforms(config.dataset)
        trainset, testset = None, None
        if config.dataset == 'mnist':
            trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
        if config.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=root + '/cifar-10-python', train=True, download=True,
                                                    transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root=root + '/cifar-10-python', train=False, download=True,
                                                   transform=transform_test)
        if config.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
        if config.dataset == 'cinic-10':
            trainset = torchvision.datasets.ImageFolder(root + '/cinic-10/trainval', transform=transform_train)
            testset = torchvision.datasets.ImageFolder(root + '/cinic-10/test', transform=transform_test)
        if config.dataset == 'tiny_imagenet':
            num_workers = 16
            trainset = torchvision.datasets.ImageFolder(root + '/tiny_imagenet/train', transform=transform_train)
            testset = torchvision.datasets.ImageFolder(root + '/tiny_imagenet/val', transform=transform_test)
        return trainset, testset

    # Using the serial executor avoids multiple processes to download the same data.
    trainset, testset = SERIAL_EXEC.run(get_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % config.dataset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, sampler=train_sampler,
                                              num_workers=num_workers, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,
                                             num_workers=num_workers, drop_last=True)

    # ====================================== get dataloader ======================================
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    model.apply(weights_init)

    # ========== register mask ==================
    if config.prune_mode != 0:
        masks = dict()
        mask_key = [x for x in state['mask'].keys()]
        _layer_cnt = 0
        for idx, layer in enumerate(model.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # print(layer.weight.shape)  # [n, c, k, k]
                masks[layer] = state['mask'][mask_key[_layer_cnt]].to(device)
                _layer_cnt += 1
        mb = ModelBase(config.network, config.depth, config.dataset, model)
        mb.register_mask(masks)
        # print pruning details
        if xm.get_ordinal() == 6:
            print_inf = print_mask_information(mb, 0)
            print(print_inf)
            config.send_mail_str += print_inf

    # ====================================== fetch training schemes ======================================
    learning_rates = str_to_list(config.learning_rate, ',', float)
    weight_decays = str_to_list(config.weight_decay, ',', float)
    training_epochs = str_to_list(config.epoch, ',', int)
    # for idx in range(len(learning_rates)):
    #     # Scale learning rate to world size
    #     learning_rates[idx] *= xm.xrt_world_size()

    # ====================================== start pruning ======================================
    # ========== finetuning =======================
    acc_list = train_once(net=model,
                          trainloader=trainloader,
                          testloader=testloader,
                          learning_rate=learning_rates[0],
                          weight_decay=weight_decays[0],
                          num_epochs=training_epochs[0],
                          device=device)

    return acc_list


best_acc = multiprocessing.Value("d", 0.0)
best_epoch = multiprocessing.Value("d", 0.0)


def _mp_fn(rank):
    torch.set_default_tensor_type('torch.FloatTensor')
    acc_list = run_tpu()
    max_acc = max(acc_list)
    print('[%d] => best acc: %.4f, epoch: %d\n' % (rank, max_acc, acc_list.index(max_acc)))
    if max_acc > best_acc.value:
        best_acc.value = max_acc
        best_epoch.value = acc_list.index(max_acc)


xmp.spawn(_mp_fn, nprocs=8, start_method='fork')

config.send_mail_str += 'best acc: %.4f, epoch: %d\n' % (best_acc.value, best_epoch.value)
print(config.send_mail_str)
