import torch.nn as nn
import torch
from models.synclr.vision_transformer_synclr import VisionTransformer


class SynCLRModel(nn.Module):
    def __init__(self, name, num_classes=1, channels=768):
        super(SynCLRModel, self).__init__()
        self.model = VisionTransformer()
        state_dict = torch.load(
            "pretrained_weights/synclr_vit_b_16.pth",
        )
        new_state_dict = {
            key.replace("module.visual.", ""): value
            for key, value in state_dict["model"].items()
        }
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.head = nn.Identity()
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.forward_features(x)
        if return_feature:
            return features
        return self.fc(features)
