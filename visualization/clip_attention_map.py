import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from io import BytesIO
from scipy.ndimage.filters import gaussian_filter
from random import random, choice
import torchvision.transforms.functional as TF

import cv2
from PIL import Image

# from torchvision import transforms
from models import get_model

MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}

STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}


def get_attention_map(img, model, transform, get_mask=False):
    x = transform(img)
    print(x.unsqueeze(0).size())

    att_mat_dict = model.get_attention_weights_dict(x.unsqueeze(0))
    result_list = []
    for att_mat in att_mat_dict.values():
        if att_mat.size(1) == 1:
            continue

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        if get_mask:
            result = cv2.resize(mask / mask.max(), img.size)
        else:
            mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
            result = (mask * img).astype("uint8")
        result_list.append(result)
    return result_list


def save_attention_map(original_img, att_map, save_path):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title("Original")
    ax1.imshow(original_img)
    ax2.set_title("Attention Map Last Layer")
    ax2.imshow(att_map)

    # Save the original image
    fig.savefig(save_path)


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < 0.5:
        sig = sample_continuous([0.0, 3.0])
        gaussian_blur(img, sig)

    if random() < 0.5:
        method = sample_discrete(["cv2", "pil"])
        qual = sample_discrete([30, 100])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}


def custom_resize(img, opt):
    return TF.resize(img, 256, interpolation=rz_dict["bilinear"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--arch", type=str, default="CLIP:ViT-B/16")
    # parser.add_argument("--arch", nargs="+", help="see my_models/__init__.py")
    opt = parser.parse_args()

    opt.cropSize = 224
    crop_func = transforms.CenterCrop(opt.cropSize)
    flip_func = transforms.Lambda(lambda img: img)
    rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    stat_from = (
        "imagenet"
        if any(arch.lower().startswith("imagenet") for arch in opt.arch)
        else "clip"
    )
    transform = transforms.Compose(
        [
            rz_func,
            transforms.Lambda(lambda img: data_augment(img, opt)),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ]
    )

    # Load the model
    model = get_model(opt.arch).eval()

    img_path = "datasets/test/progan/bird/0_real/06154.png"

    img = Image.open(img_path)
    result_list = get_attention_map(img, model, transform, get_mask=False)
    for index, result in enumerate(result_list):
        save_attention_map(
            np.array(img), result, save_path=f"outputs/attention_map/{index}.png"
        )
