'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, c=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(c, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 1*28*28 => 6*24*24
        out = F.max_pool2d(out, 2)  # 6*24*24 => 6*12*12
        out = F.relu(self.conv2(out))  # 6*12*12 => 16*8*8
        out = F.max_pool2d(out, 2)  # 16*8*8 => 16*4*4
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet_5_Caffe(nn.Module):
    """
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.
    """

    def __init__(self, c=1):
        super().__init__()

        self.conv1 = nn.Conv2d(c, 20, 5, padding=0)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
        # x = F.log_softmax(self.fc4(x), dim=1)
        x = self.fc4(x)

        return x


def lenet(c=1):
    # return LeNet(c)
    return LeNet_5_Caffe(c)

