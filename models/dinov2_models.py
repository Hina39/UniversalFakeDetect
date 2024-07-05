import torch.nn as nn
import torch

from submodules.dinov2_attention_visualization.dinov2.hub.backbones import dinov2_vitb14


class DinoModel(nn.Module):
    def __init__(self, name, num_classes=1, model_name="dinov2_vitb14", channels=768):
        super(DinoModel, self).__init__()
        # self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = dinov2_vitb14()
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model(x)
        if return_feature:
            return features
        return self.fc(features)

    def get_attention_weights_dict(self, x):
        return self.model.get_attention_weights_dict(x)


if __name__ == "__main__":

    # _model = DinoModel(name="DINOv2:ViT-B/14").cuda().eval()
    # print(_model.model.patch_embed.img_size)
    # print(_model.model.patch_embed.embed_dim)
    # print(_model(torch.rand(1, 3, 224, 224).cuda()))

    model = dinov2_vitb14().cuda().eval()
    # summary(model, input_size=[[3, 224, 224]])
    attention_weights = model.get_attention_weights_dict(torch.rand(1, 3, 224, 224).cuda())
    for layer, weights in attention_weights.items():
        print(f"{layer}: {weights.shape}")