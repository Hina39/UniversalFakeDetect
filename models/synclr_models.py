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

    def get_attention_weights_dict(self, x):
        return self.model.get_all_layers_attention_weights(x)


if __name__ == "__main__":

    model = SynCLRModel(name="a").cuda().eval()
    attention_weights = model.get_attention_weights_dict(
        torch.rand(1, 3, 224, 224).cuda()
    )
    for layer, weights in attention_weights.items():
        print(f"{layer}: {weights.shape}")
