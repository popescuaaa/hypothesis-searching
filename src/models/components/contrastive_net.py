import torch.nn as nn
from torch import Tensor


class ContrastiveNet(nn.Module):
    def __init__(self, backbone, head):
        super(ContrastiveNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x