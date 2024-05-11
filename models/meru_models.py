from models.meru.text_encoders import TransformerTextEncoder
from models.meru.meru import LazyCall, build_timm_vit, CheckpointManager, MERU
import torch.nn as nn
import torch
from hydra.utils import instantiate


class MeruModel(nn.Module):
    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder = LazyCall(TransformerTextEncoder)(
            arch="L12_W512", vocab_size=49408, context_length=77
        ),
        embed_dim: int = 512,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        num_classes=1,
        channels=512,
    ):
        super(MeruModel, self).__init__()
        device = (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = LazyCall(MERU)(
            visual=LazyCall(build_timm_vit)(
                arch="vit_large_patch16_224",
                global_pool="token",
                use_sincos2d_pos=True,
            ),
            textual=LazyCall(TransformerTextEncoder)(
                arch="L12_W512", vocab_size=49408, context_length=77
            ),
            embed_dim=512,
            curv_init=1.0,
            learn_curv=True,
            entail_weight=0.2,
        )
        # 今回はModel: MERU ViT-smallを使うことにする
        model.visual.arch = "vit_small_mocov3_patch16_224"
        device = device or torch.cuda.current_device()
        model = instantiate(model).to(device).eval()
        CheckpointManager(model=model).load("pretrained_weights/meru_vit_s.pth")
        self.model = model
        # MERUにはない、新しいFC層をつける
        self.fc = nn.Linear(channels, num_classes)

    def forward(
        self, images: torch.Tensor, return_feature=False
    ) -> dict[str, torch.Tensor]:

        image_feats = self.model.encode_image(images, project=True)
        if return_feature:
            return image_feats
        return self.fc(image_feats)
