import torch.nn as nn
from torchvision.models import resnet50
from torch import Tensor

class ResNet50(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=True, progress=True)

        # Make a linear from output of resnet to num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        print("input shape: ", x.shape)
        out = self.model(x)
        print("output shape: ", out.shape)
        return out

if __name__ == "__main__":
    _ = ResNet50(num_classes=100)

    # Test the model
    import torch
    x = torch.randn(1, 1, 64, 64)
    model = ResNet50(num_classes=100)
    out = model(x)
    print(out.shape)