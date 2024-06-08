import torch.nn as nn
import torch
import timm


class DeitModel(nn.Module):
    def __init__(self, name, num_classes=1, channels=768):
        super(DeitModel, self).__init__()
        self.model = timm.models.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.1,
            num_classes=21000,
        )
        state_dict = torch.load(
            "pretrained_weights/vit_base_with_visualatom_21k.pth.tar",
            map_location="cpu",
        )
        self.model.load_state_dict(state_dict["state_dict"], strict=False)
        self.model.head = nn.Identity()
        self.model.head_drop = nn.Identity()
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model(x)
        if return_feature:
            return features
        return self.fc(features)

# いま行っていることのメモ
#   (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#   (fc_norm): Identity()
#   (head_drop): Dropout(p=0.0, inplace=False) -> Identity()
#   (head): Linear(in_features=768, out_features=21000, bias=True) -> Identity()
