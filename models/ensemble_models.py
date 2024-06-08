import torch.nn as nn
import torch
from typing import List


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], num_classes=1, fix_backbone=True):
        super(EnsembleModel, self).__init__()
        # modelsはモデルのインスタンス化されたもののリスト
        self.models = models
        if fix_backbone:
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
        self.fc = nn.Linear(
            sum([model.fc.in_features for model in self.models]), num_classes
        )

    def to(self, device):
        # Override the to method to move all models to the target device
        self = super().to(device)
        self.models = [model.to(device) for model in self.models]
        return self

    def forward(self, x, return_feature=False):
        features = [model.forward(x, return_feature=True) for model in self.models]
        features = torch.cat(features, dim=1)
        if return_feature:
            return features
        return self.fc(features)
