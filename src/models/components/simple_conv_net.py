import torch.nn as nn
from torch import Tensor

class SimpleConvNet(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 conv1_channels: int = 32,
                 conv2_channels: int = 64,
                 conv3_channels: int = 128,
                 conv4_channels: int = 256,
                 num_classes: int = 100,
            ):

        super(SimpleConvNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv3_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv3_channels, conv4_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv4_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv4_channels, num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x: Tensor):
        batch_size, channels, width, height = x.size()
        return self.model(x)

if __name__ == "__main__":
    _ = SimpleConvNet()

    # Test the model with a random input
    import torch
    x = torch.rand(1, 1, 64, 64)
    print(_.forward(x).shape)