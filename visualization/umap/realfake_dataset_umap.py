import os
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
from PIL import Image
import pickle
from io import BytesIO
from copy import deepcopy

import random
from scipy.ndimage.filters import gaussian_filter

SEED = 0


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}

STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0 : N // 2].max() <= y_pred[N // 2 : N].min():  # perfectly separable case
        return (y_pred[0 : N // 2].max() + y_pred[N // 2 : N].min()) / 2

    best_acc = 0
    best_thres = 0
    # もし本当のラベルの最大予測値が偽のラベルの最小予測値以下であれば（つまり，予測値が完全に分離可能であれば）
    # この関数はこれら2つの値の平均をしきい値として返す．
    # 予測が完全に分離可能でない場合，関数は best_acc （最高の精度）と best_thres （最高のしきい値）を0に初期化する．
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format="jpeg", quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    # r_acc:0である真のラベルの予測精度 f_acc:1である真のラベルの予測精度　# acc:全体の予測精度
    return r_acc, f_acc, acc


def validate(model, loader, find_thres=False):

    with torch.no_grad():
        y_true, y_pred = [], []
        print("Length of dataset: %d" % (len(loader)))
        device = torch.device("cuda:0")
        for img, label in loader:
            in_tens = img.to(device)
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ================== save this if you want to plot the curves =========== #
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #

    # Get AP
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    # r_acc0:0である真のラベルの予測精度 f_acc0:1である真のラベルの予測精度　# acc0:全体の予測精度
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    print(f"rootdir: {rootdir}")
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split(".")[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=""):
    if ".pickle" in path:
        with open(path, "rb") as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeUmapDataset(Dataset):
    def __init__(
        self,
        real_path,
        fake_path,
        data_mode,
        max_sample,
        key,
        arch,
        jpeg_quality=None,
        gaussian_sigma=None,
    ):
        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        # = = = = = = data path = = = = = = = = = #
        if isinstance(real_path, str) and isinstance(fake_path, str):
            real_list, fake_list = self.read_path(
                real_path, fake_path, data_mode, max_sample
            )
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        print(f"real_list: {len(real_list)}")
        print(f"fake_list: {len(fake_list)}")
        self.total_list = real_list + fake_list

        # = = = = = =  label = = = = = = = = = #
        self.labels_dict = {}
        # GANのラベルを1に設定
        if key in ["progan", "cyclegan", "biggan", "gaugan", "stargan", "ffhq"]:
            for i in real_list:
                self.labels_dict[i] = 0
            for i in fake_list:
                self.labels_dict[i] = 1
        # DMのラベルを2に設定
        elif key in [
            "guided",
            "ldm_200",
            "ldm_200_cfg",
            "glide_100_27",
            "glide_100_10",
            "elsa",
        ]:
            for i in real_list:
                self.labels_dict[i] = 0
            for i in fake_list:
                self.labels_dict[i] = 2
        else:
            raise ValueError()

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
            ]
        )

    def read_path(self, real_path, fake_path, data_mode, max_sample):

        # 'wang2020'：0_realと1_fakeを含むディレクトリ
        # 　'ours'：それ以外のディレクトリ
        if data_mode == "wang2020":
            real_list = get_list(real_path, must_contain="0_real")
            fake_list = get_list(fake_path, must_contain="1_fake")
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        assert len(real_list) == len(fake_list)

        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        # print(label)
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label
