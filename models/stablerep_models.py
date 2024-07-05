import torch.nn as nn
import torch
from models.synclr.vision_transformer_synclr import VisionTransformer


class StableRepModel(nn.Module):
    def __init__(self, name, num_classes=1, channels=768):
        super(StableRepModel, self).__init__()
        self.model = VisionTransformer()
        state_dict = torch.load(
            "pretrained_weights/laion_stablerep_pp_50m.pth",
        )
        new_state_dict = {
            key.replace("visual.", ""): value
            for key, value in state_dict["model"].items()
            if "visual." in key
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

    model = StableRepModel(name="a").cuda().eval()
    attention_weights = model.get_attention_weights_dict(
        torch.rand(1, 3, 224, 224).cuda()
    )
    for layer, weights in attention_weights.items():
        print(f"{layer}: {weights.shape}")
