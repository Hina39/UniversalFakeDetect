import pathlib
from typing import List, NamedTuple

import numpy as np
import torch

from visualization.umap.realfake_dataset_umap import RealFakeUmapDataset
from validate import RealFakeDataset
import argparse
from models import get_model
from dataset_paths import DATASET_PATHS
import random


class ExtractFeaturesReturn(NamedTuple):
    """ExtractFeaturesの返り値として使うNamedTuple.

    Attributes:
        labels_list (List[np.ndarray]): クラスラベル.
        features_list (List[np.ndarray]): 特徴量.

    """

    labels_list: List[np.ndarray]
    features_list: List[np.ndarray]


SEED = 0


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def extract_features() -> ExtractFeaturesReturn:
    """特徴量などの情報を抽出する.

    Returns:
        ExtractFeaturesReturn: 特徴量などの情報.

    """
    model = get_model(opt.arch)

    print("Model loaded..")

    model.eval()
    model.cuda()

    if (opt.real_path is None) or (opt.fake_path is None) or (opt.data_mode is None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [
            dict(
                real_path=opt.real_path,
                fake_path=opt.fake_path,
                data_mode=opt.data_mode,
                key=opt.key,
            )
        ]

    labels_list: List[np.ndarray] = list()
    features_list: List[np.ndarray] = list()

    for dataset_path in dataset_paths:
        set_seed()

        if opt.gan_dm:
            dataset = RealFakeUmapDataset(
                dataset_path["real_path"],
                dataset_path["fake_path"],
                dataset_path["data_mode"],
                opt.max_sample,
                dataset_path["key"],
                opt.arch,
                jpeg_quality=opt.jpeg_quality,
                gaussian_sigma=opt.gaussian_sigma,
            )
        else:
            dataset = RealFakeDataset(
                dataset_path["real_path"],
                dataset_path["fake_path"],
                dataset_path["data_mode"],
                opt.max_sample,
                opt.arch,
                jpeg_quality=opt.jpeg_quality,
                gaussian_sigma=opt.gaussian_sigma,
            )

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4
        )

        for img, label in loader:
            labels_list.append(label.cpu().data.numpy())
            img = img.cuda()
            if opt.arch == "Meru:Vit-B":
                with torch.no_grad():
                    features = model.model.encode_image(img, project=False)
            elif opt.arch == "SynCLR:ViT-B/16" or opt.arch == "StableRep:ViT-B/16":
                with torch.no_grad():
                    features = model.model.forward_features(img)
            elif (
                opt.arch == "CLIP:ViT-B/16"
                or opt.arch == "Open_CLIPE32:ViT-B/16"
                or opt.arch == "Deit:ViT-B/16"
            ):
                with torch.no_grad():
                    features = model.model.encode_image(img)
            elif opt.arch == "Dino:Vit-B/14":
                with torch.no_grad():
                    features = model.model(img)
            else:
                with torch.no_grad():
                    features = model.model(img)

            features_list.append(features.cpu().data.numpy())

    return ExtractFeaturesReturn(
        labels_list=labels_list,
        features_list=features_list,
    )


def main(file_name) -> None:
    """特徴量と画像の真のラベルを抽出して保存する."""

    extracted_features = extract_features()

    output_dir = pathlib.Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / file_name,
        labels=np.concatenate(extracted_features.labels_list),
        features=np.concatenate(extracted_features.features_list),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--real_path",
        type=str,
        default="datasets/test/progan/",
        help="dir name or a pickle",
    )
    parser.add_argument(
        "--fake_path",
        type=str,
        default="datasets/test/progan/",
        help="dir name or a pickle",
    )
    parser.add_argument(
        "--data_mode", type=str, default="wang2020", help="wang2020 or ours"
    )
    parser.add_argument("--key", type=str, default="progan", help="")
    parser.add_argument(
        "--max_sample",
        type=int,
        default=1000,
        help="only check this number of images for both fake/real",
    )

    parser.add_argument("--arch", type=str, default="Open_CLIPE32:ViT-B/16")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=None,
        help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=int,
        default=None,
        help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None",
    )
    parser.add_argument(
        "--gan_dm",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--file_name", type=str, default="otamesi.npz", help="file name to save"
    )
    opt = parser.parse_args()
    main(opt.file_name)
