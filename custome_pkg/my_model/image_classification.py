from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def FashionMnistNeuralNetwork():
    return NeuralNetwork()


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1
            ),  # 输入通道1，输出通道32，卷积核大小3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2最大池化
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1
            ),  # 输入通道32，输出通道64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2最大池化
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # 根据卷积和池化后的输出尺寸进行调整
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def FashionMnistConvolutionalNeuralNetwork():
    return ConvolutionalNeuralNetwork()


class Cifar10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def Cifar10NeuralNetwork():
    return Cifar10Net()
