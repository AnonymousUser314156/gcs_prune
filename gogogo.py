# -*- coding: utf-8 -*-

"""
Created on 11/17/2021
gogogo.
@author: ***
"""

import os

os.system("python main_prune_gcs.py --config configs/cifar10/vgg19/GSS_90.json --run gss_data_z1 --data_mode 1 --grad_mode 2 --prune_mode 2 --num_group 2")
os.system("python main_prune_gcs.py --config configs/cifar10/vgg19/GSS_95.json --run gss_data_z2 --data_mode 1 --grad_mode 2 --prune_mode 2 --num_group 2")
os.system("python main_prune_gcs.py --config configs/cifar10/vgg19/GSS_98.json --run gss_data_z2 --data_mode 1 --grad_mode 2 --prune_mode 2 --num_group 2")

os.system("python main_prune_gcs.py --config configs/cifar10/resnet32/GSS_98.json --run gss_data_z4 --data_mode 1 --grad_mode 2 --prune_mode 2 --num_group 2")
os.system("python main_prune_gcs.py --config configs/cifar10/resnet32/GSS_95.json --run gss_data_z5 --data_mode 1 --grad_mode 2 --prune_mode 2 --num_group 2")
os.system("python main_prune_gcs.py --config configs/cifar10/resnet32/GSS_90.json --run gss_data_z6 --data_mode 1 --grad_mode 2 --prune_mode 2 --num_group 2")


