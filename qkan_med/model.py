"""QKAN-Med: ResNet18, bounded angle encoding, and a 16-qubit quantum head."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from .quantum import QuantumKAN


class QKANMed(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad_(name.startswith(("layer3.", "layer4.")))
        self.adapter = nn.Linear(512, 16)
        self.dropout = nn.Dropout(0.5)
        self.readout = nn.Linear(16, 9)
        with torch.random.fork_rng(devices=[]):
            self.bottleneck = QuantumKAN()
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for name in ("bn1", "layer1", "layer2"):
            getattr(self.encoder, name).eval()
        return self

    def forward(self, images):
        z = self.adapter(self.encoder(images)).sigmoid()
        return self.readout(self.dropout(self.bottleneck(z)))
