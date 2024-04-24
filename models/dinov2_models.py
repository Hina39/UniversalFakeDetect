import torch.nn as nn
import torch


class DinoModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(DinoModel, self).__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        num_classes = 1  # For binary classification
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model(x)
        if return_feature:
            return features
        return self.fc(features)


# dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
#  (linear_head): Linear(in_features=1920, out_features=1000, bias=True)

# dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
# (linear_head): Linear(in_features=3840, out_features=1000, bias=True)

# dinov2_vitl14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
# (linear_head): Linear(in_features=5120, out_features=1000, bias=True)

# dinov2_vitg14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc')
#  (linear_head): Linear(in_features=7680, out_features=1000, bias=True)
