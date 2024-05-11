import torch.nn as nn
import torch


class DinoModel(nn.Module):
    def __init__(self, name, num_classes=1, model_name="dinov2_vitb14", channels=768):
        super(DinoModel, self).__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model(x)
        if return_feature:
            return features
        return self.fc(features)
