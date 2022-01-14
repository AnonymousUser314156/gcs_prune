# -*- coding: utf-8 -*-

"""
Created on 8/13/2021
Gss.
@author: ***
"""
# -*- coding: utf-8 -*-


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

import copy
import types


def fetch_data(dataloader, num_classes, samples_per_class, mode=0):
    if mode == 0:
        datas = [[] for _ in range(num_classes)]
        labels = [[] for _ in range(num_classes)]
        mark = dict()
        dataloader_iter = iter(dataloader)
        while True:
            inputs, targets = next(dataloader_iter)
            for idx in range(inputs.shape[0]):
                x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
                category = y.item()
                if len(datas[category]) == samples_per_class:
                    mark[category] = True
                    continue
                datas[category].append(x)
                labels[category].append(y)
            if len(mark) == num_classes:
                break

        X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    else:
        dataloader_iter = iter(dataloader)
        inputs, targets = next(dataloader_iter)
        X, y = inputs[0:samples_per_class * num_classes], targets[0:samples_per_class * num_classes]
    return X, y


def hessian_gradient_product(net, train_dataloader, device,
                             num_classes=10, samples_per_class=10, T=200, reinit=True,
                             data_mode=0, grad_mode=0, num_group=None):
    """
        data_mode:
            0 - grouping with different labels
            1 - grouping with same labels
        gard_mode:
            0 - gradient norm squared
            1 - gradient coupling (one-to-many)
            2 - gradient coupling (gcs)
            3 - loss
            4 - max gradient coupling (gcs-max)
    """

    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)

    print("fetch data")
    inputs, targets = fetch_data(train_dataloader, num_classes, samples_per_class)
    print("inputs:", inputs.shape)
    N = inputs.shape[0]
    if num_group is None:
        num_group = 2
    equal_parts = N // num_group
    if data_mode == 0:
        pass
    else:
        # Arrangement of different label groups
        _index = []
        for i in range(samples_per_class):
            _index.extend([i + j * samples_per_class for j in range(0, num_classes)])
        inputs = inputs[_index]
        targets = targets[_index]
        # print(_index)
    # print(targets[:num_classes])
    inputs = inputs.to(device)
    targets = targets.to(device)
    print("gradient => g")
    gradg_list = []

    if grad_mode == 0:
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad = autograd.grad(_loss, weights, create_graph=True)
            _gz = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    _gz += _grad[_layer].pow(2).sum()  # g * g
                    _layer += 1
            gradg_list.append(autograd.grad(_gz, weights))
    elif grad_mode == 1:
        _grad_ls = []
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad_ls.append(autograd.grad(_loss, weights, create_graph=True))
        _grad_and = []
        _layer = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _gand = 0
                for i in range(num_group):
                    _gand += _grad_ls[i][_layer]
                _grad_and.append(_gand)
                _layer += 1
        for i in range(num_group):
            _gz = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    _gz += (_grad_and[_layer] * _grad_ls[i][_layer]).sum()  # ga * gn
                    _layer += 1
            gradg_list.append(autograd.grad(_gz, weights, retain_graph=True))
    elif grad_mode == 2:
        _grad_ls = []
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad_ls.append(autograd.grad(_loss, weights, create_graph=True))

        for i in range(num_group):
            for j in range(i + 1, num_group):
                _gz = 0
                _layer = 0
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        _gz += (_grad_ls[i][_layer] * _grad_ls[j][_layer]).sum()  # g1 * g2
                        _layer += 1
                gradg_list.append(autograd.grad(_gz, weights, retain_graph=True))
    elif grad_mode == 3:
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            gradg_list.append(autograd.grad(_loss, weights, retain_graph=True))
    elif grad_mode == 4:
        _grad_ls = []
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad_ls.append(autograd.grad(_loss, weights, create_graph=True))

        # Pick the largest group (gigj)
        _max_gigj = -31415
        _max_index_i = 666
        for i in range(num_group):
            _sum_gigi = 0
            for j in range(num_group):
                _layer = 0
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        _sum_gigi += (_grad_ls[i][_layer] * _grad_ls[j][_layer]).sum()  # g1 * g2
                        _layer += 1
            print(f'_sum_gigi:{_sum_gigi}')
            if _max_gigj < _sum_gigi:
                _max_gigj = _sum_gigi
                _max_index_i = i
        # Calculate hg for the largest gigj
        gradg_list = []
        for j in range(num_group):
            _gz = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    _gz += (_grad_ls[_max_index_i][_layer] * _grad_ls[j][_layer]).sum()  # g1 * g2
                    _layer += 1
            gradg_list.append(autograd.grad(_gz, weights, retain_graph=True))

    return gradg_list


def Gcs(net, ratio, train_dataloader, device,
        num_classes=10, samples_per_class=10, num_iters=1, T=200, reinit=True,
        data_mode=0, grad_mode=0, prune_mode=0, num_group=None):
    eps = 1e-10
    keep_ratio = (1 - ratio)
    old_net = net
    net = copy.deepcopy(net)
    net.zero_grad()

    if prune_mode == 0:
        keep_masks = dict()
        old_modules = list(old_net.modules())
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                keep_masks[old_modules[idx]] = torch.ones_like(layer.weight.data).float()
        return keep_masks, 0

    # Data and group matching
    if samples_per_class == 5 and num_group == 2:
        samples_per_class = 6

    gradg_list = None
    for it in range(num_iters):
        print("Iterations %d/%d." % (it, num_iters))
        _hessian_grad = hessian_gradient_product(net, train_dataloader, device,
                                          num_classes, samples_per_class, T, reinit,
                                          data_mode, grad_mode, num_group)
        if gradg_list is None:
            gradg_list = _hessian_grad
        else:
            for i in range(len(gradg_list)):
                _grad_i = _hessian_grad[i]
                gradg_list[i] = [gradg_list[i][_l] + _grad_i[_l] for _l in range(len(_grad_i))]
    # print(len(gradg_list))

    # === Pruning part ===
    """
        prune_mode:
            1 - sum
            2 - sum absolute value
            3 - product
            4 - Euclidean distance
        
        default:
        gss: --data_mode 1 --grad_mode 2 --prune_mode 2 --num_group 2
        gss-group: --data_mode 1 --grad_mode 2 --prune_mode 4 --num_group 5
        grasp: --data_mode 0 --grad_mode 0 --prune_mode 1 --num_group 1
        snip: --data_mode 0 --grad_mode 3 --prune_mode 1 --num_group 1
    """

    # === Calculate the score ===
    layer_cnt = 0
    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            kxt = 0
            if prune_mode == 1:
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt]  # theta_q grad
                    # kxt += torch.abs(_qhg)
                    kxt += _qhg
                #     print(torch.mean(_qhg), torch.sum(_qhg))
                # print('-' * 20)

            if prune_mode == 2:
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt += _qhg
                kxt = torch.abs(kxt)

            if prune_mode == 3:
                kxt = 1e6  # is approximately equal to the hyperparameter, estimated value
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt *= torch.abs(_qhg)
                    # print(torch.mean(torch.abs(_qhg)), torch.sum(torch.abs(_qhg)))
                # print('-' * 20)

            if prune_mode == 4:
                aef = 1e6  # is approximately equal to the hyperparameter, estimated value
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt] * aef  # theta_q grad
                    kxt += _qhg.pow(2)
                kxt = kxt.sqrt()

            # assessment score
            grads[old_modules[idx]] = kxt
            # print(torch.mean(kxt), torch.sum(kxt))
            # print('-' * 20)

            layer_cnt += 1

    # === Determine masks based on importance ===
    keep_masks = dict()

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * keep_ratio)
    threshold, _index = torch.topk(all_scores, num_params_to_rm)
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)

    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) >= acceptable_score).float()

    print('Remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    def _get_connected_scores(info='', mode=0):
        # Calculate connectivity
        _connected_scores = 0
        _last_filter = None
        for m, g in keep_masks.items():
            if isinstance(m, nn.Conv2d) and 'padding' in str(m):  # Does not consider the shortcut part of resnet
                # [n, c, k, k]
                _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
                _channel = np.sum(_2d, axis=0)  # vector

                if _last_filter is not None:  # The first layer is not considered
                    for i in range(_channel.shape[0]):  # Traverse channel filter
                        if _last_filter[i] == 0 and _channel[i] != 0:
                            _connected_scores += 1
                        if mode == 1:
                            if _last_filter[i] != 0 and _channel[i] == 0:
                                _connected_scores += 1

                _last_filter = np.sum(_2d, axis=1)

        print(f'{info}-{mode}->_connected_scores: {_connected_scores}')
        return _connected_scores

    _connected_scores = _get_connected_scores(f"{'-' * 20}\nBefore", 1)

    return keep_masks, _connected_scores
