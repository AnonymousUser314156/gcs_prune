# -*- coding: utf-8 -*-

"""
Created on 11/17/2021
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

    return gradg_list


def Gcs(net, ratio, train_dataloader, device,
        num_classes=10, samples_per_class=10, num_iters=1, T=200, reinit=True,
        data_mode=0, grad_mode=0, prune_mode=0, num_group=None, debug_mode=0):
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

    # 数据和组匹配
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
                gradg_list[i] = [gradg_list[i][l] + _grad_i[l] for l in range(len(_grad_i))]

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

    # === debug ===
    if debug_mode == 2:
        grad_a = dict()
        grad_b = dict()
        grad_c = dict()

    # === score ===
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
                kxt = 1e6
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt *= torch.abs(_qhg)
                    # print(torch.mean(torch.abs(_qhg)), torch.sum(torch.abs(_qhg)))
                # print('-' * 20)

            if prune_mode == 4:
                aef = 1e6
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt] * aef  # theta_q grad
                    kxt += _qhg.pow(2)
                kxt = kxt.sqrt()

            if debug_mode == 1:
                pass

            if debug_mode == 2:
                grad_a[old_modules[idx]] = torch.abs(layer.weight.data * gradg_list[2][layer_cnt])
                grad_b[old_modules[idx]] = torch.abs(layer.weight.data * gradg_list[6][layer_cnt])
                grad_c[old_modules[idx]] = torch.abs(layer.weight.data * gradg_list[7][layer_cnt])

            grads[old_modules[idx]] = kxt
            # print(torch.mean(kxt), torch.sum(kxt))
            # print('-' * 20)

            layer_cnt += 1

    # === masks ===
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
        _connected_scores = 0
        _last_filter = None
        for m, g in keep_masks.items():
            if isinstance(m, nn.Conv2d) and 'padding' in str(m):
                # [n, c, k, k]
                _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
                _channel = np.sum(_2d, axis=0)

                if _last_filter is not None:
                    for i in range(_channel.shape[0]):
                        if _last_filter[i] == 0 and _channel[i] != 0:
                            _connected_scores += 1
                        if mode == 1:
                            if _last_filter[i] != 0 and _channel[i] == 0:
                                _connected_scores += 1

                _last_filter = np.sum(_2d, axis=1)

        print(f'{info}-{mode}->_connected_scores: {_connected_scores}')
        return _connected_scores

    _connected_scores = _get_connected_scores(f"{'-' * 20}\nBefore", 1)

    # === debug ===
    grad_key = [x for x in grads.keys()]
    if debug_mode == 2:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        user_layer = 1
        gain = 1000
        dpi = 100
        xpixels = 1200
        ypixels = 1200
        xinch = xpixels / dpi
        yinch = ypixels / dpi
        fig = plt.figure(figsize=(xinch, yinch))
        ax = fig.add_subplot(111, projection='3d')

        # (n, c, k, k)
        _filter_n = int(grad_a[grad_key[user_layer]].shape[0])
        _channel_n = int(grad_a[grad_key[user_layer]].shape[1])
        _a = torch.cat([torch.flatten(torch.mean(grad_a[grad_key[user_layer]], dim=(2, 3)))])
        _b = torch.cat([torch.flatten(torch.mean(grad_b[grad_key[user_layer]], dim=(2, 3)))])
        _c = torch.cat([torch.flatten(torch.mean(grad_c[grad_key[user_layer]], dim=(2, 3)))])
        _np_a = _a.cpu().detach().numpy() * gain
        _np_b = _b.cpu().detach().numpy() * gain
        _np_c = _c.cpu().detach().numpy() * gain
        _data_len = len(_np_a)
        _threshold = np.sort(_np_a)[_data_len-(_data_len//5)]

        _channel_pos = np.ones(_filter_n)
        _np_x = _channel_pos*0
        for i in range(1, _channel_n):
            _np_x = np.hstack((_np_x, _channel_pos*i))

        _filter_pos = np.arange(0, _filter_n)
        _np_y = _filter_pos
        for n in range(_channel_n-1):
            _np_y = np.hstack((_np_y, _filter_pos))

        surface_x = np.linspace(-10, _channel_n+10, _channel_n+20)
        surface_y = np.linspace(-10, _filter_n+10, _filter_n+20)
        s_X, s_Y = np.meshgrid(surface_x, surface_y)
        # surface z
        # ax.plot_surface(s_X, s_Y, s_X*0+_threshold, color='white', alpha=0.3)

        #
        # ax.scatter(xs=_np_x, ys=_np_y, zs=_np_a, c='red', s=20, alpha=0.8, marker='.')
        # ax.scatter(xs=_np_x, ys=_np_y, zs=_np_b, c='blue', s=20, alpha=0.8, marker='*')
        # ax.scatter(xs=_np_x, ys=_np_y, zs=_np_c, c='green', s=10, alpha=0.7, marker='.')

        #
        # ax.plot_trisurf(_np_x, _np_y, _np_a, color='red')
        # surf1 = ax.plot_trisurf(_np_x, _np_y, _np_a, alpha=0.8, cmap="cool")
        # surf2 = ax.plot_trisurf(_np_x, _np_y, _np_b, alpha=0.8, cmap="Wistia")
        # fig.colorbar(surf1, shrink=0.5, aspect=10)
        # fig.colorbar(surf2, shrink=0.5, aspect=10)

        #
        _np_x = _np_x.reshape(_filter_n, _channel_n)
        _np_y = _np_y.reshape(_filter_n, _channel_n)
        _np_a = _np_a.reshape(_filter_n, _channel_n)
        _np_b = _np_b.reshape(_filter_n, _channel_n)
        surf1 = ax.plot_surface(_np_x, _np_y, _np_a, alpha=0.6, cmap="cool")
        # surf2 = ax.plot_surface(_np_x, _np_y, _np_b, alpha=0.8, cmap="Wistia")
        surf2 = ax.plot_surface(_np_x, _np_y, _np_b, alpha=0.6, cmap="hot")
        fig.colorbar(surf1, shrink=0.5)
        fig.colorbar(surf2, shrink=0.5)

        # plt.savefig(debug_path + 'epoch_' + str(debug_epoch) + '.png', dpi=dpi)
        plt.show()

    return keep_masks, _connected_scores
