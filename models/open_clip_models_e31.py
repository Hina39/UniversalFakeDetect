import torch.nn as nn
from submodules.open_clip_attention_visualization.src import open_clip


class OpenCLIPE31Model(nn.Module):
    def __init__(
        self,
        name,
        num_classes=1,
        model_name="ViT-B-16",
        pretrained="laion400m_e31",
        channels=512,
    ):
        super(OpenCLIPE31Model, self).__init__()
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