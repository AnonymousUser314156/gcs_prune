# Gradient Coupled Flow: Performance Boosting On Network Pruning By Utilizing Implicit Loss Decrease

## Introduce

Use PyTorch to implement our Sensitive to Gradient Coupling(GCS) pruning before training

## Structure
 - configs: Configuration files for different models and different datasets
 - models: (lenet,resnet,vgg) Three models and the model class containing the mask
 - pruner: Pruning algorithm
 - runs: Output folder to store weights, evaluation information, etc.
 - utils: custom helper function
 - main_prune_gcs.py and main_prune_gcs_tpu.py: Main function (corresponding to running on GPU or TPU)
 - main_prune_gcs_exp.py: For experiment debugging and plotting


## Environment

The code has been tested to run on Python 3.8.

Some package versions are as follows:
* torch == 1.7.1
* numpy == 1.18.5
* tensorboardX==2.4

## Run

* cifar10/vgg19 prune ratio: 90%
```
# GCS
python main_prune_gcs.py --config configs/cifar10/vgg19/GSS_90.json --run test --prune 'gcs'
```
```
# GCS-Group
python main_prune_gcs.py --config configs/cifar10/vgg19/GSS_90.json --run test --prune 'gcs-group'
```

* mnist/lenet5 prune ratio: 98%
```
python main_prune_gcs.py --config configs/mnist/lenet/GSS_98.json --run test --prune 'gcs'
```

* Train with TPU (requires GPU to calculate mask first)
```
python main_prune_gcs.py --config configs/cifar10/vgg19/GSS_98.json --run test_tpu --storage_mask 1
python main_prune_gcs_tpu.py --config configs/cifar10/vgg19/GSS_98.json --run test_tpu --mask_path runs/pruning/cifar10/vgg19/cifar10_vgg19_prune_ratio_98/test_tpu/checkpoint/cifar10_vgg19_prune_ratio_98_test_tpu
```

- Model optional: ```lenet, vgg, resnet```

- Dataset optional: ```fashionmnist, mnist, cifar10, cifar100```(Other datasets need to be manually downloaded to the local)

- All parameters(The default parameters are determined by the config file):

    | Console Parameters | Remark |
    | :---- | :---- |
    | config = '' | # Configuration file paths for multiple models and multiple datasets |
    | pretrained = '' | # Path to load pretrained model |
    | run = 'test' | # Used to remark this experiment |
    | prune = '666' | # For fast selection of pruning algorithms (GCS,GCS-Group,GraSP,SNIP)|
    | epoch = '666' | # Modify the number of training rounds |
    | batch_size = 256 | # Modify the batch size of training samples |
    | data_mode = 1 | Used for pruning before training, the data pattern used (see GCS.py for details) |
    | grad_mode = 2 | # Pattern for pre-training pruning, solving for Hessian gradient products (see GCS.py for details) |
    | prune_mode = 2 | # Used for pre-training pruning to get a pattern of metrics (see GCS.py for details) |
    | num_group = 2 | # Used for pre-training pruning, number of gradient groups（GCS-Group） |
    | remain = 2 | # Used to temporarily modify the pruning rate (remaining percentage) |
    | lr_mode = 'cosine' | # Set the learning rate decay method (cosine or preset) |
    | storage_mask = 0 | # store the resulting mask |
    | debug = 0 | # for debugging |
    | dp = '../Data' | # Modify the path of the dataset |


## ...
To be added ...

