from .clip_models import CLIPModel
from .imagenet_models import ImagenetModel
from .dinov2_models import DinoModel
from .meru_models import MeruModel
from .deit_models import DeitModel
from .open_clip_models_e32 import OpenCLIPE32Model
from .open_clip_models_e31 import OpenCLIPE31Model
from .synclr_models import SynCLRModel


VALID_NAMES = [
    "Imagenet:resnet18",
    "Imagenet:resnet34",
    "Imagenet:resnet50",
    "Imagenet:resnet101",
    "Imagenet:resnet152",
    "Imagenet:vgg11",
    "Imagenet:vgg19",
    "Imagenet:swin-b",
    "Imagenet:swin-s",
    "Imagenet:swin-t",
    "Imagenet:vit_b_16",
    "Imagenet:vit_b_32",
    "Imagenet:vit_l_16",
    "Imagenet:vit_l_32",
    "CLIP:RN50",
    "CLIP:RN101",
    "CLIP:RN50x4",
    "CLIP:RN50x16",
    "CLIP:RN50x64",
    "CLIP:ViT-B/32",
    "CLIP:ViT-B/16",
    "CLIP:ViT-L/14",
    "CLIP:ViT-L/14@336px",
    "Meru:Vit-B",
    "Dino:Vit-B/14",
    "Deit:ViT-B/16",
    "Open_CLIPE32:ViT-B/16",
    "Open_CLIPE31:ViT-B/16",
    "SynCLR:ViT-B/16",
]


def get_model(name):
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:])
    elif name.startswith("CLIP:"):
        return CLIPModel(name[5:])
    elif name.startswith("Dino:"):
        return DinoModel(name[5:])
    elif name.startswith("Meru:"):
        return MeruModel(name[5:])
    elif name.startswith("Deit:"):
        return DeitModel(name[5:])
    elif name.startswith("Open_CLIPE32:"):
        return OpenCLIPE32Model(name[13:])
    elif name.startswith("Open_CLIPE31:"):
        return OpenCLIPE31Model(name[13:])
    elif name.startswith("SynCLR:"):
        return SynCLRModel(name[7:])
    else:
        assert False
