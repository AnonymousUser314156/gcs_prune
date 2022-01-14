# -*- coding: utf-8 -*-

"""
Created on 11/17/2021
main_prune_gcs.
@author: ***
"""

import argparse
import json
import math
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from models.model_base import ModelBase
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.base.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import get_network
from pruner.GCS_exp import Gcs, fetch_data


def init_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='configs/cifar10/resnet32/GSS_98.json')
    parser.add_argument('--config', type=str, default='configs/cifar10/vgg19/GSS_98.json')
    # parser.add_argument('--config', type=str, default='configs/mnist/lenet/GSS_90.json')
    # parser.add_argument('--config', type=str, default='configs/fashionmnist/lenet/GSS_90.json')
    parser.add_argument('--run', type=str, default='exp_lr001')
    parser.add_argument('--prune', type=str, default='666')
    parser.add_argument('--epoch', type=str, default='666')
    parser.add_argument('--data_mode', type=int, default=1)
    parser.add_argument('--grad_mode', type=int, default=2)
    parser.add_argument('--prune_mode', type=int, default=0)
    parser.add_argument('--num_group', type=int, default=2)
    parser.add_argument('--ignore_grad', type=int, default=1)
    parser.add_argument('--remain', type=float, default=666)
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
    if args.remain != 666:
        config.target_ratio = (100 - args.remain) / 100.0
        print("set new target_ratio:{}".format(config.target_ratio))
    if args.num_group != 666:
        if 10 % args.num_group == 0:
            config.num_group = args.num_group
            print("set label group:{}".format(config.num_group))
        else:
            config.num_group = None
            print("num_group must be divisible by the number of tags")
    else:
        config.num_group = None
    config.ignore_grad = args.ignore_grad
    config.data_mode = args.data_mode
    config.grad_mode = args.grad_mode
    config.prune_mode = args.prune_mode
    config.debug = args.debug
    config.dp = args.dp
    config.send_mail_str = ('Wubba lubba dub dub\n')

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
    path_main = os.path.join(path, 'main_prune_gcs_exp.py')
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


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, iteration):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # debug(coupled gradient)
    delta_loss = 0

    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_last_lr(), 0, 0, correct, total))

    writer.add_scalar('iter_%d/train/lr' % iteration, lr_scheduler.get_last_lr(), epoch)

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # debug(coupled gradient calculation)
        outputs = net(inputs)
        after_loss = criterion(outputs, targets)
        delta_loss += (loss - after_loss).item()

        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_last_lr(), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('iter_%d/train/loss' % iteration, train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('iter_%d/train/acc' % iteration, 100. * correct / total, epoch)
    writer.add_scalar('iter_%d/train/delta_loss' % iteration, delta_loss / batch_idx, epoch)


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


def writer_exp(net, masks, loader, criterion, scheduler, writer, epoch, num_classes, gtg_mode, ignore_grad, exp_data, exp_name):
    # net.train()
    # Recalculate loss and map for easy first- and second-order derivatives
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)
    # layer_key
    layer_key = [x for x in masks.keys()]

    # 10 categories, 10 samples from each category
    samples_per_class = 10
    if num_classes == 100:
        samples_per_class = 2
    if gtg_mode == 0 or gtg_mode == 5:
        inputs, targets = fetch_data(loader, num_classes, samples_per_class)
        _index = []
        for i in range(samples_per_class):
            _index.extend([i + j * samples_per_class for j in range(0, num_classes)])
        inputs = inputs[_index]
        targets = targets[_index]
        # print(targets[:num_classes])
    elif gtg_mode == 1:
        inputs, targets = fetch_data(loader, num_classes, samples_per_class)
    elif gtg_mode == 2:
        inputs, targets = fetch_data(loader, 1, 128, 1)  # 128 random samples
    elif gtg_mode == 3:
        inputs, targets = fetch_data(loader, 1, 10, 1)  # 10 random samples
    elif gtg_mode == 4:
        inputs, targets = fetch_data(loader, 1, 2, 1)  # 2 random samples
    else:
        inputs, targets = exp_data
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Consider the gradient of the two-norm
    if gtg_mode == 5:
        l2_loss = []
        _layer = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                l2_loss.append((weights[_layer] ** 2).sum() / 2.0)
                _layer += 1
        _l2_reg = 0.0005 * sum(l2_loss)
        writer.add_scalar(exp_name + '/l2_reg', _l2_reg, epoch)
        # print(_l2_reg)
    else:
        _l2_reg = 0

    is_second_order = 1
    num_group = 2
    N = inputs.shape[0]
    equal_parts = N // num_group
    grad_ls = []
    grad_nograph = []
    gss_ls = []
    gcs_second_i = []
    gcs_second_j = []
    for i in range(num_group):
        _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
        _loss = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts]) + _l2_reg
        grad_ls.append(autograd.grad(_loss, weights, create_graph=True))
        if is_second_order:
            grad_nograph.append(list(autograd.grad(_loss, weights, retain_graph=True)))

    for i in range(num_group):
        for j in range(i + 1, num_group):
            _gz = 0
            _gz_hi = 0
            _gz_hj = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if ignore_grad == 0:
                        _gz += (grad_ls[i][_layer] * grad_ls[j][_layer]).sum()  # g1 * g2
                        if is_second_order:
                            _gz_hi += (grad_nograph[i][_layer] * grad_ls[j][_layer]).sum()  # g1 * g2
                            _gz_hj += (grad_ls[i][_layer] * grad_nograph[j][_layer]).sum()  # g1 * g2
                    else:
                        _gz += (grad_ls[i][_layer] * grad_ls[j][_layer] * masks[layer_key[_layer]]).sum()  # g1 * g2
                        if is_second_order:
                            _gz_hi += (grad_nograph[i][_layer] * grad_ls[j][_layer] * masks[layer_key[_layer]]).sum()  # g1 * g2
                            _gz_hj += (grad_ls[i][_layer] * grad_nograph[j][_layer] * masks[layer_key[_layer]]).sum()  # g1 * g2
                    _layer += 1
            gss_ls.append(_gz)
            if is_second_order:
                gcs_second_i.append(_gz_hi)
                gcs_second_j.append(_gz_hj)

    # D1 + D2
    _outputs = net.forward(inputs)
    _loss = criterion(_outputs, targets) + _l2_reg
    _g = autograd.grad(_loss, weights, create_graph=True)
    if is_second_order:
        _g_nograph= list(autograd.grad(_loss, weights, retain_graph=True))
    gtg = 0
    _layer = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if ignore_grad == 0:
                gtg += _g[_layer].pow(2).sum()  # g * g
            else:
                gtg += (_g[_layer] * masks[layer_key[_layer]]).pow(2).sum()  # g * g
            _layer += 1

    # alpha theta g
    # atg = 0
    # for i, x in enumerate(_g):
    #     atg += 0.0005 * (weights[i] * x).sum()
    # writer.add_scalar(exp_name + '/atg', atg, epoch)
    # print(atg)

    # ggi ggj
    ggi = 0
    ggj = 0
    ggi_sec = 0
    for i, x in enumerate(_g):
        if ignore_grad == 0:
            ggi += (grad_ls[0][i] * x).sum()
            ggj += (grad_ls[1][i] * x).sum()
            if is_second_order:
                ggi_sec += (grad_ls[0][i] * _g_nograph[i]).sum()
        else:
            ggi += (grad_ls[0][i] * x * masks[layer_key[i]]).sum()
            ggj += (grad_ls[1][i] * x * masks[layer_key[i]]).sum()
            if is_second_order:
                ggi_sec += (grad_ls[0][i] * _g_nograph[i] * masks[layer_key[i]]).sum()
    writer.add_scalar(exp_name + '/ggi', ggi, epoch)
    writer.add_scalar(exp_name + '/ggj', ggj, epoch)

    # # ------------- Gradient sine component vector product -------------
    # _sin_aph_list = []
    # for i in range(num_group):
    #     g_gi = 0
    #     gigi = 0
    #     for _l in range(len(grad_ls[i])):
    #         g_gi += (_g[_l] * grad_ls[i][_l]).sum()
    #         gigi += (grad_ls[i][_l]).pow(2).sum()
    #     _sin_aph_list.append((1 - (g_gi / (gtg.sqrt() * gigi.sqrt())).pow(2)).sqrt())
    # _sin_gij_list = []
    # for i in range(num_group):
    #     for j in range(i + 1, num_group):
    #         _sin_gij = 0
    #         for _layer in range(len(grad_ls[i])):
    #             _sin_gij += ((grad_ls[i][_layer]*_sin_aph_list[i]) * (grad_ls[j][_layer]*_sin_aph_list[j])).sum()  # g1 * g2
    #         _sin_gij_list.append(_sin_gij)
    #
    # for i, gg in enumerate(_sin_gij_list):
    #     writer.add_scalar(exp_name + '/sin_gij_%d' % i, gg, epoch)
    # # sin alpha and alpha
    # for i, sin in enumerate(_sin_aph_list):
    #     writer.add_scalar(exp_name + '/sin_%d' % i, sin, epoch)
    #     writer.add_scalar(exp_name + '/alpha_%d' % i, torch.asin(sin), epoch)

    # gi gj gradient norm
    gtg_ls = []
    for i in range(num_group):
        _gz = 0
        _layer = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if ignore_grad == 0:
                    _gz += (grad_ls[i][_layer] * grad_ls[i][_layer]).sum()  # g * g
                else:
                    _gz += (grad_ls[i][_layer] * grad_ls[i][_layer] * masks[layer_key[_layer]]).sum()  # g * g
                _layer += 1
        gtg_ls.append(_gz)
    for i, gg in enumerate(gtg_ls):
        writer.add_scalar(exp_name + '/gij_%d' % i, gg, epoch)

    # # gi gj angle
    # _cos_gij = gss_ls[0] / (gtg_ls[0].sqrt() * gtg_ls[1].sqrt())
    # writer.add_scalar(exp_name + '/alpha_ij', torch.acos(_cos_gij), epoch)

    # second order term gHg
    if is_second_order:
        gHg = 0
        Hg = autograd.grad(gtg, weights, retain_graph=True)
        for i, x in enumerate(_g):
            if ignore_grad == 0:
                gHg += (Hg[i] * x).sum()
            else:
                gHg += (Hg[i] * x * masks[layer_key[i]]).sum()
        writer.add_scalar(exp_name + '/gHg', gHg, epoch)

        # gi Hj gi  &  gj Hi gj  &  g Hi g
        gHgi = 0  # gi Hj gi
        gHgj = 0  # gj Hi gj
        gHig = 0  # g Hi g

        # Hg_i = autograd.grad(gcs_second_i[0], weights, retain_graph=True)
        # Hg_j = autograd.grad(gcs_second_j[0], weights, retain_graph=True)
        # Hi_g = autograd.grad(ggi_sec, weights)
        # # => gi Hj gi
        # for i, x in enumerate(grad_ls[0]):
        #     if ignore_grad == 0:
        #         gHgi += (Hg_i[i] * x).sum()
        #     else:
        #         gHgi += (Hg_i[i] * x * masks[layer_key[i]]).sum()
        # # => gj Hi gj
        # for i, x in enumerate(grad_ls[1]):
        #     if ignore_grad == 0:
        #         gHgj += (Hg_j[i] * x).sum()
        #     else:
        #         gHgj += (Hg_j[i] * x * masks[layer_key[i]]).sum()
        # => g Hi g
        # for i, x in enumerate(_g):
        #     if ignore_grad == 0:
        #         gHig += (Hi_g[i] * x).sum()
        #     else:
        #         gHig += (Hi_g[i] * x * masks[layer_key[i]]).sum()

        # writer.add_scalar(exp_name + '/gHgi', gHgi, epoch)
        # writer.add_scalar(exp_name + '/gHgj', gHgj, epoch)

        epsilon = scheduler.get_last_lr()[0]

        # writer.add_scalar(exp_name + '/gcdi', gss_ls[0]-(0.5*epsilon*gHgi), epoch)
        # writer.add_scalar(exp_name + '/gcdj', gss_ls[0]-(0.5*epsilon*gHgj), epoch)

        # => LDi_1 = -e * ggi + 0.5e**2 * gHig
        # => LDi_2 = LDi_1 - 0.5e * gss
        # => LD_ = -e * gg + 0.5e**2 * gHg
        # LDi_1 = -epsilon * ggi + 0.5 * epsilon * epsilon * gHig
        # LDi_2 = LDi_1 - 0.5 * epsilon * gss_ls[0]
        LD_ = -epsilon * gtg
        LD__ = LD_ + 0.5 * epsilon * epsilon * gHg
        # print(LDi_1)
        # print(LDi_2)
        # writer.add_scalar(exp_name + '/LDi_1', LDi_1, epoch)
        # writer.add_scalar(exp_name + '/LDi_2', LDi_2, epoch)
        writer.add_scalar(exp_name + '/LD_', LD_, epoch)
        writer.add_scalar(exp_name + '/LD__', LD__, epoch)
        print('first-order approximate:', LD_)
        print('second order approximate:', LD__)

    # ------------- debug ----------------
    # use_layer = 2
    # print('all_g', torch.mean(_g[use_layer]), torch.sum(_g[use_layer]))
    # print('g1', torch.mean(grad_ls[0][use_layer]), torch.sum(grad_ls[0][use_layer]))
    # print('g2', torch.mean(grad_ls[1][use_layer]), torch.sum(grad_ls[1][use_layer]))
    """
        if num_group is 2: 
            2*gtg - gtg_ls.sum()/2 = gss_ls
    """
    # print(gtg)
    # print(gtg_ls)
    # print(gss_ls)
    # print(gHg)
    # print(gHg_ig)

    # tensorboard
    writer.add_scalar(exp_name + '/gtg', gtg, epoch)
    writer.add_scalar(exp_name + '/gss', gss_ls[0], epoch)


def loss_coupling(net, masks, loader, criterion, scheduler, writer, epoch, num_classes, gtg_mode, ignore_grad, exp_data, exp_name):

    net = copy.deepcopy(net).train()
    net.zero_grad()
    _optim = optim.SGD(net.parameters(), lr=scheduler.get_last_lr()[0], momentum=0.9, weight_decay=0.0005)

    samples_per_class = 10
    if num_classes == 100:
        samples_per_class = 2
    if gtg_mode == 0:
        inputs, targets = fetch_data(loader, num_classes, samples_per_class)
        _index = []
        for i in range(samples_per_class):
            _index.extend([i + j * samples_per_class for j in range(0, num_classes)])
        inputs = inputs[_index]
        targets = targets[_index]
        # print(targets[:num_classes])
    elif gtg_mode == 1:
        inputs, targets = fetch_data(loader, num_classes, samples_per_class)
    elif gtg_mode == 2:
        inputs, targets = fetch_data(loader, 1, 128, 1)
    elif gtg_mode == 3:
        inputs, targets = fetch_data(loader, 1, 10, 1)
    elif gtg_mode == 4:
        inputs, targets = fetch_data(loader, 1, 2, 1)
    else:
        inputs, targets = exp_data
    inputs = inputs.cuda()
    targets = targets.cuda()

    num_group = 2
    N = inputs.shape[0]
    equal_parts = N // num_group
    # Calculate the loss value of dj before di training
    i = 1
    _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
    before_loss_j = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
    # Train with di
    i = 0
    _optim.zero_grad()
    _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
    _loss = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
    _loss.backward()
    _optim.step()
    # Calculate the loss value of dj after di training
    i = 1
    _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
    after_loss_j = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])

    # writer.add_scalar(exp_name + '/before_loss_j', before_loss_j, epoch)
    # writer.add_scalar(exp_name + '/after_loss_j', after_loss_j, epoch)
    writer.add_scalar(exp_name + '/coupling_loss', before_loss_j-after_loss_j, epoch)
    # print(before_loss_j-after_loss_j)

    # # batch Loss change before and after training
    # # training
    # _optim.zero_grad()
    # _outputs = net.forward(inputs)
    # _loss = criterion(_outputs, targets)
    # _loss.backward()
    # _optim.step()
    # # Calculate the loss value after training
    # _outputs = net.forward(inputs)
    # after_loss_j = criterion(_outputs, targets)
    #
    # writer.add_scalar(exp_name + '/before_loss', _loss, epoch)
    # writer.add_scalar(exp_name + '/after_loss', after_loss_j, epoch)
    # writer.add_scalar(exp_name + '/delta_loss', after_loss_j-_loss, epoch)
    # print('really delta_loss:', after_loss_j-_loss)


def train_once(mb, net, trainloader, testloader, writer, config, ckpt_path, learning_rate, weight_decay, num_epochs,
               iteration, ratio, num_classes, logger, prune_masks=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    exp_data = fetch_data(trainloader, 1, 128, 1)

    print_inf = ''
    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epochs):

        # writer_exp(net, prune_masks, trainloader, criterion, lr_scheduler, writer, epoch, num_classes, 0, config.ignore_grad, exp_data, 'data')
        # writer_exp(net, prune_masks, trainloader, criterion, writer, epoch, num_classes, 1, ignore_grad, 'label')
        # writer_exp(net, prune_masks, trainloader, criterion, writer, epoch, num_classes, 2, ignore_grad, 'random')
        # writer_exp(net, prune_masks, trainloader, criterion, writer, epoch, num_classes, 3, ignore_grad, 'random10')
        # writer_exp(net, prune_masks, trainloader, criterion, writer, epoch, num_classes, 4, ignore_grad, 'random2')
        # writer_exp(net, prune_masks, trainloader, criterion, writer, epoch, num_classes, 5, ignore_grad, 'data100_reg')
        # loss_coupling(net, prune_masks, trainloader, criterion, lr_scheduler, writer, epoch, num_classes, 0, config.ignore_grad, exp_data, 'data')

        # writer_exp(net, prune_masks, trainloader, criterion, lr_scheduler, writer, epoch, num_classes, 6, config.ignore_grad, exp_data, 'data')
        # loss_coupling(net, prune_masks, trainloader, criterion, lr_scheduler, writer, epoch, num_classes, 6, config.ignore_grad, exp_data, 'data')

        # writer_exp(net, prune_masks, trainloader, criterion, lr_scheduler, writer, epoch, num_classes, 2, config.ignore_grad, exp_data, 'random')
        # loss_coupling(net, prune_masks, trainloader, criterion, lr_scheduler, writer, epoch, num_classes, 2, config.ignore_grad, exp_data, 'random')

        train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer, iteration=iteration)
        test_acc = test(net, testloader, criterion, epoch, writer, iteration)
        lr_scheduler.step()

        if test_acc > best_acc and epoch > 10:
            # print('Saving..')
            # state = {
            #     'net': net,
            #     'acc': test_acc,
            #     'epoch': epoch,
            #     'args': config,
            #     'mask': mb.masks,
            #     'ratio': mb.get_ratio_at_each_layer()
            # }
            # path = os.path.join(ckpt_path, 'finetune_%s_%s%s_r%s_it%d_best.pth.tar' % (config.dataset,
            #                                                                            config.network,
            #                                                                            config.depth,
            #                                                                            config.target_ratio,
            #                                                                            iteration))
            # torch.save(state, path)
            best_acc = test_acc
            best_epoch = epoch

    logger.info('Iteration [%d], best acc: %.4f, epoch: %d' %
                (iteration, best_acc, best_epoch))
    return 'Iteration [%d], best acc: %.4f, epoch: %d\n' % (iteration, best_acc, best_epoch), print_inf


def main(config):
    # init logger
    classes = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'tiny_imagenet': 200
    }
    logger, writer = init_logger(config)

    # torch config
    # torch.backends.cudnn.benchmark = True
    # print('Using cudnn.benchmark.')
    # build model
    model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', True))
    mask = None
    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()
    if mask is not None:
        mb.register_mask(mask)
        print_mask_information(mb, logger)

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
                   num_group=config.num_group,
                   debug_mode=config.debug
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
                                   ratio=ratio,
                                   num_classes=classes[config.dataset],
                                   logger=logger,
                                   prune_masks=masks
                                   )

    config.send_mail_str += print_inf
    config.send_mail_str += tr_str

    print(config.send_mail_str)


if __name__ == '__main__':
    config = init_config()
    main(config)
