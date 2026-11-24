from torch import nn
from torchvision import models

class VGG16Classifier(nn.Module):
    """VGG16 feature extractor + the exact 512 -> 512 -> 38 head in the checkpoint."""
    def __init__(self, num_classes: int = 38) -> None:
        super().__init__()
        base = models.vgg16(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)
