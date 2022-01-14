# -*- coding: utf-8 -*-

"""
Created on 8/10/2021
main_prune_gcs.
@author: ***
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_base import ModelBase
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.base.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import get_network
from pruner.GCS import Gcs


def init_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='configs/cifar10/resnet32/GSS_98.json')
    parser.add_argument('--config', type=str, default='configs/cifar10/vgg19/GSS_98.json')
    # parser.add_argument('--config', type=str, default='configs/mnist/lenet/GSS_98.json')
    # parser.add_argument('--config', type=str, default='configs/fashionmnist/lenet/GSS_98.json')
    parser.add_argument('--pretrained', type=str, default='666')
    parser.add_argument('--run', type=str, default='my_test')
    parser.add_argument('--prune', type=str, default='666')  # For fast selection of pruning algorithms
    parser.add_argument('--epoch', type=str, default='666')
    parser.add_argument('--batch_size', type=int, default=666)
    parser.add_argument('--data_mode', type=int, default=1)
    parser.add_argument('--grad_mode', type=int, default=2)
    parser.add_argument('--prune_mode', type=int, default=2)
    parser.add_argument('--num_group', type=int, default=2)
    parser.add_argument('--remain', type=float, default=666)
    parser.add_argument('--samples_per', type=int, default=666)
    parser.add_argument('--num_iters', type=int, default=666)
    parser.add_argument('--l2', type=str, default='666')
    parser.add_argument('--lr_mode', type=str, default='cosine', help='cosine or preset')
    parser.add_argument('--optim_mode', type=str, default='SGD', help='SGD or Adam')
    parser.add_argument('--storage_mask', type=int, default=0)  # storage mask and transfer paddle
    parser.add_argument('--debug', type=int, default=0)  # Debug flags (printing data, plotting, etc.)
    parser.add_argument('--dp', type=str, default='../Data', help='dataset path')
    args = parser.parse_args()

    runs = None
    if len(args.run) > 0:
        runs = args.run
    config = process_config(args.config, runs)
    if args.pretrained != '666':
        config.pretrained = args.pretrained
        print("use pre-trained mode:{}".format(config.pretrained))
    else:
        config.pretrained = None
    if args.epoch != '666':
        config.epoch = args.epoch
        print("set new epoch:{}".format(config.epoch))
    if args.batch_size != 666:
        config.batch_size = args.batch_size
        print("set new batch_size:{}".format(config.batch_size))
    if args.remain != 666:
        config.target_ratio = (100 - args.remain) / 100.0
        print("set new target_ratio:{}".format(config.target_ratio))
    if args.samples_per != 666:
        config.samples_per_class = args.samples_per
        print("set new samples_per_class:{}".format(config.samples_per_class))
    if args.num_iters != 666:
        config.num_iters = args.num_iters
        print("set new num_iters:{}".format(config.num_iters))
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
    config.data_mode = args.data_mode
    config.grad_mode = args.grad_mode
    config.prune_mode = args.prune_mode
    config.storage_mask = args.storage_mask
    config.lr_mode = args.lr_mode
    config.optim_mode = args.optim_mode
    config.debug = args.debug
    config.dp = args.dp
    config.send_mail_str = "Wubba lubba dub dub\n=>"

    if args.prune != '666':
        # gcs: --data_mode 1 --grad_mode 2 --prune_mode 2 --num_group 2
        # gcs-group: --data_mode 1 --grad_mode 2 --prune_mode 4 --num_group 5
        # grasp: --data_mode 0 --grad_mode 0 --prune_mode 1 --num_group 1
        # snip: --data_mode 0 --grad_mode 3 --prune_mode 1 --num_group 1
        if args.prune.lower() == 'gcs':
            config.data_mode = 1
            config.grad_mode = 2
            config.prune_mode = 2
            config.num_group = 2
        elif args.prune.lower() == 'gcs-group':
            config.data_mode = 1
            config.grad_mode = 2
            config.prune_mode = 4
            config.num_group = 5
        elif args.prune.lower() == 'gcs-max':
            config.data_mode = 1
            config.grad_mode = 4
            config.prune_mode = 4
            config.num_group = 5
        elif args.prune.lower() == 'grasp':
            config.data_mode = 0
            config.grad_mode = 0
            config.prune_mode = 1
            config.num_group = 1
        elif args.prune.lower() == 'snip':
            config.data_mode = 0
            config.grad_mode = 3
            config.prune_mode = 1
            config.num_group = 1
        else:
            print("Select pruning mode: GCS, GCS-Group, GraSP, SNIP")
            exit()
        print("Select pruning mode:{}".format(args.prune))

    return config


def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'models/base/%s.py' % config.network.lower())
    path_main = os.path.join(path, 'main_prune_gcs.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner_file)
    logger = get_logger('log', logpath=config.summary_dir + '/',
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    return logger, writer


def print_mask_information(mb, logger):
    ratios = mb.get_ratio_at_each_layer()
    logger.info('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    re_str = '** Mask information of %s. Overall Remaining: %.2f%%\n' % (mb.get_name(), ratios['ratio'])
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
        re_str += '  (%d) %.2f%%\n' % (count, v)
        count += 1
    return re_str


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, iteration, lr_mode):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if lr_mode == 'cosine':
        _lr = lr_scheduler.get_last_lr()
    elif 'preset' in lr_mode:
        _lr = lr_scheduler.get_lr(optimizer)
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (_lr, 0, 0, correct, total))
    writer.add_scalar('iter_%d/train/lr' % iteration, _lr, epoch)

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if lr_mode == 'cosine':
            _lr = lr_scheduler.get_last_lr()
        elif 'preset' in lr_mode:
            lr_scheduler(optimizer, epoch)
            _lr = lr_scheduler.get_lr(optimizer)
        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (_lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('iter_%d/train/loss' % iteration, train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('iter_%d/train/acc' % iteration, 100. * correct / total, epoch)


def test(net, loader, criterion, epoch, writer, iteration):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100. * correct / total

    writer.add_scalar('iter_%d/test/loss' % iteration, test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('iter_%d/test/acc' % iteration, 100. * correct / total, epoch)
    return acc


def train_once(mb, net, trainloader, testloader, writer, config, ckpt_path, learning_rate, weight_decay, num_epochs,
               iteration, logger, pretrain=None, lr_mode='cosine', optim_mode='SGD'):
    criterion = nn.CrossEntropyLoss()
    if optim_mode == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if lr_mode == 'cosine':
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif 'preset' in lr_mode:
        lr_schedule = {0: learning_rate,
                       int(num_epochs * 0.5): learning_rate * 0.1,
                       int(num_epochs * 0.75): learning_rate * 0.01}
        lr_scheduler = PresetLRScheduler(lr_schedule)
    else:
        print('===!!!=== Wrong learning rate decay setting! ===!!!===')
        exit()

    print_inf = ''
    best_epoch = 0
    if pretrain:
        best_acc = pretrain['acc']
        continue_epoch = pretrain['epoch']
    else:
        best_acc = 0
        continue_epoch = -1
    for epoch in range(num_epochs):
        if epoch > continue_epoch:  # idling at other times
            train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer, iteration, lr_mode)
            test_acc = test(net, testloader, criterion, epoch, writer, iteration)

            if test_acc > best_acc and epoch > 10:
                print('Saving..')
                state = {
                    'net': net,
                    'acc': test_acc,
                    'epoch': epoch,
                    'args': config,
                    'mask': mb.masks,
                    'ratio': mb.get_ratio_at_each_layer()
                }
                path = os.path.join(ckpt_path, 'finetune_%s_%s%s_r%s_it%d_best.pth.tar' % (config.dataset, config.network, config.depth, config.target_ratio, iteration))
                torch.save(state, path)
                best_acc = test_acc
                best_epoch = epoch
        if lr_mode == 'cosine':
            lr_scheduler.step()
        else:
            lr_scheduler(optimizer, epoch)

    logger.info('Iteration [%d], best acc: %.4f, epoch: %d' %
                (iteration, best_acc, best_epoch))
    return 'Iteration [%d], best acc: %.4f, epoch: %d\n' % (iteration, best_acc, best_epoch), print_inf


def main(config):
    # init logger
    classes = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'fashionmnist': 10,
        'tiny_imagenet': 200
    }
    logger, writer = init_logger(config)

    state = None
    # build/load model
    if config.pretrained:
        state = torch.load(config.pretrained)
        model = state['net']
        masks = state['mask']
        config.send_mail_str += f"use pre-trained mode -> acc:{state['acc']} epoch:{state['epoch']}\n"
        config.network = state['args'].network
        config.depth = state['args'].depth
        config.dataset = state['args'].dataset
        config.batch_size = state['args'].batch_size
        config.learning_rate = state['args'].learning_rate
        config.weight_decay = state['args'].weight_decay
        config.epoch = state['args'].epoch
        config.target_ratio = state['args'].target_ratio
        print('load model finish')
        print(state['args'])
    else:
        model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', True))
        masks = None

    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()

    # preprocessing
    # ====================================== get dataloader ======================================
    trainloader, testloader = get_dataloader(config.dataset, config.batch_size, 256, 4, root=config.dp)
    # ====================================== fetch configs ======================================
    ckpt_path = config.checkpoint_dir
    num_iterations = config.iterations
    target_ratio = config.target_ratio
    normalize = config.normalize
    # ====================================== fetch training schemes ======================================
    ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
    learning_rates = str_to_list(config.learning_rate, ',', float)
    weight_decays = str_to_list(config.weight_decay, ',', float)
    training_epochs = str_to_list(config.epoch, ',', int)
    logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
                (normalize, num_iterations, target_ratio, ratio))
    logger.info('Basic Settings: ')
    for idx in range(len(learning_rates)):
        logger.info('  %d: LR: %.5f, WD: %.5f, Epochs: %d' % (idx,
                                                              learning_rates[idx],
                                                              weight_decays[idx],
                                                              training_epochs[idx]))

    # ====================================== start pruning ======================================
    iteration = 0
    if masks is None:
        logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                    ratio,
                                                                                    1,
                                                                                    num_iterations))
        mb.model.apply(weights_init)
        print("=> Applying weight initialization(%s)." % config.get('init_method', 'kaiming'))
        masks, _ = Gcs(mb.model, ratio, trainloader, 'cuda',
                       num_classes=classes[config.dataset],
                       samples_per_class=config.samples_per_class,
                       num_iters=config.get('num_iters', 1),
                       data_mode=config.data_mode,
                       grad_mode=config.grad_mode,
                       prune_mode=config.prune_mode,
                       num_group=config.num_group
                       )
    # ========== register mask ==================
    mb.register_mask(masks)
    # ========== print pruning details ============
    print_inf = print_mask_information(mb, logger)
    config.send_mail_str += print_inf
    logger.info('  LR: %.5f, WD: %.5f, Epochs: %d' %
                (learning_rates[iteration], weight_decays[iteration], training_epochs[iteration]))
    config.send_mail_str += 'LR: %.5f, WD: %.5f, Epochs: %d, Batch: %d \n' % (
        learning_rates[iteration], weight_decays[iteration], training_epochs[iteration], config.batch_size)

    # ========== storage mask ==========
    if config.storage_mask == 1:
        state = {
            'args': config,
            'mask': masks,
        }
        path = os.path.join(ckpt_path, config.exp_name)
        torch.save(state, path)
        print("=> storage mask finish", config.exp_name)
        exit()

    # ========== finetuning =======================
    tr_str, print_inf = train_once(mb=mb,
                                   net=mb.model,
                                   trainloader=trainloader,
                                   testloader=testloader,
                                   writer=writer,
                                   config=config,
                                   ckpt_path=ckpt_path,
                                   learning_rate=learning_rates[iteration],
                                   weight_decay=weight_decays[iteration],
                                   num_epochs=training_epochs[iteration],
                                   iteration=iteration,
                                   logger=logger,
                                   pretrain=state,
                                   lr_mode=config.lr_mode,
                                   optim_mode=config.optim_mode
                                   )

    config.send_mail_str += print_inf
    config.send_mail_str += tr_str

    # Send the test results to my email
    print(config.send_mail_str)


if __name__ == '__main__':
    config = init_config()
    main(config)
