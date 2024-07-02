import torch.nn as nn
from submodules.open_clip_attention_visualization.src import open_clip


class OpenCLIPE32Model(nn.Module):
    def __init__(
        self,
        name,
        num_classes=1,
        model_name="ViT-B-16",
        pretrained="laion400m_e32",
        channels=512,
    ):
        super(OpenCLIPE32Model, self).__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

    def get_attention_weights_dict(self, x):
        return self.model.get_attention_weights_dict(x)


if __name__ == "__main__":
    import torch
    model = OpenCLIPE32Model(name="ViT-B/16").cuda().eval()
    # print(model)
    # summary(model, input_size=[[3, 224, 224]])
    attention_weights = model.get_attention_weights_dict(torch.rand(1, 3, 224, 224).cuda())
    for layer, weights in attention_weights.items():
        print(f"{layer}: {weights.shape}")