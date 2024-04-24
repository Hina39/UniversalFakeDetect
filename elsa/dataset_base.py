import dataclasses
import pathlib
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import lightning.pytorch
import numpy as np
import torch
from hydra.utils import instantiate
from kornia.augmentation import Normalize
from kornia.augmentation.container import ImageSequential
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset


class DeepFakeDetectionLabel(IntEnum):
    """DeepFake検出の正解ラベルの種類を保持する列挙型. REALとFAKEのどちらが0で,
    どちらが1だったが, 分からなくならないように列挙型を使用します.

    Note:
        新しいDeepFakeデータセットが追加された際はこのクラスは別の場所に移動して
        クラス間で共有するべき.

    """

    REAL = 0
    FAKE = 1


class ImageSize(NamedTuple):
    """画像の大きさの情報を保有するNamedTuple.

    Attributes:
        height (int): 画像の縦方向の次元数.
        width (int): 画像の横方向の次元数.

    """

    height: int
    width: int


class RgbColor(NamedTuple):
    """RGBの情報を保存するNamedTuple.

    Attributes:
        r (float): Red.
        g (float): Green.
        b (float): Blue.

    """

    r: float
    g: float
    b: float


@dataclasses.dataclass(frozen=True)
class ImageDatasetStats:
    """データセットの統計情報を保持しているデータクラス.

    Attributes:
        num_classes (int): データセットのクラス数.
        image_size (ImageSize): データセットの画像の大きさ.
        mean (RgbColor): ピクセル値のRGBのそれぞれの平均値.
        std (RgbColor): ピクセル値のRGBのそれぞれの標準偏差.

    """

    num_classes: int
    image_size: ImageSize
    mean: RgbColor
    std: RgbColor


class DataPoint(NamedTuple):
    """Datasetクラスの__getitem__メソッドからの返り値として使うNamedTuple.
    PyTorchのDatasetクラスは__getitem__の返り値としてdataclassをサポートして
    いないのでNamedTupleを使用している.

    Attributes:
        image (Union[np.ndarray, torch.Tensor]): 画像データ.
        target (Union[int, torch.Tensor]): 画像データのラベル.
        landmarks (Union[np.ndarray, torch.Tensor]): 画像データの
            ランドマーク情報. PyTorchのデフォルトのcollate関数はNoneを扱えない
            ので, デフォルトでは空のnp.ndarrayを設定.
        latents (Any): 潜在変数.
        meta (Dict[str, Any]): その他のメタ情報.

    Note:
        PyTorchのDataLoaderのデフォルトのcollate_fnはnp.ndarrayやintなどを
        すべてtorch.Tensorに変換するので, このクラスの属性の型はtorch.Tensorとの
        Unionになっている.

    """

    image: Union[np.ndarray, torch.Tensor]
    target: Union[int, torch.Tensor]
    landmarks: Union[np.ndarray, Dict[str, torch.Tensor]] = np.array([])
    latents: Any = np.array([])
    meta: Dict[str, Any] = dict()


# TODO: このクラスはsrc/retriever/base_triplet.pyに移動している.
# DFDC datasetの更新が完了次第このクラスは削除.
class TripletDataPoint(NamedTuple):
    """TripletDatasetクラスからの返り値の型を定義するNamedTuple.
    PyTorchのDatasetクラスは__getitem__の返り値としてdataclassをサポートして
    いないのでNamedTupleを使用しています.

    Attributes:
        anchor (DataPoint): アンカーのデータ.
        positive (DataPoint): アンカーに対してポジティブのデータ.
        negative (DataPoint): アンカーに対してネガティブのデータ.

    """

    anchor: DataPoint
    positive: DataPoint
    negative: DataPoint


@runtime_checkable
class BaseDataset(Protocol):
    """このコードベース内において通常の分類タスクのためのDatasetクラスが従うべき,
    Protocol(ダックタイピングのために使用する). TripletDatasetクラスは
    このProtocolに従っている, Datasetを__init__で渡されることを前提として実装が
    されています. 主にmypyがこの情報を参照する.

    Note:
        現在はtransformの型はImageSequential

    """

    transform: Optional[ImageSequential]

    def __getitem__(self, index: int) -> DataPoint:  # noqa: D105
        ...

    def __len__(self) -> int:  # noqa: D105
        ...

    def __iter__(self) -> Iterator[DataPoint]:  # noqa: D105
        """NOTE: Pythonでは__iter__が定義されていなくても__getitem__が定義されて
        いればIterableになりますが, mypyのために__iter__を陽に定義する必要が
        ある. 以下のような自明な__iter__をDatasetクラスで定義する必要がある.

        for i in range(len(self)):
            yield self[i]

        """
        ...


class BaseDataModule(ABC, lightning.pytorch.LightningDataModule):
    """2次元画像を扱うLightningDataModuleのABC(Abstruct Base Class).

    Attributes:
        Phase (Literal["train", "val", "test"]): 学習フェーズを表すLiteral.
        batch_size (int): バッチサイズ.
        num_workers (int): データセットを読み込むワーカーの数.
        data_root_dir (Union[str, pathlib.Path]): 読み込むデータがおかれ
            ているルートディレクトリのパス. Hydraではpathlib.Pathの値をYAML
            ファイルに記載することが出来ないので, strも渡せるようにしている.
        dataset_stats (ImageDatasetStats): データセットの統計情報.
        augmentation_config (DictConfig): データ拡張の情報を保持した
            DictConfig. Hydraと組み合わせてYAMLファイルから読み込む想定.
        data_retriever_class (Optional[Callable]): BaseDatasetクラス
            のインスタンスに適用して, データを取得する方法を変えるクラス.
            例えば, VanillaTripletDataRetrieverなどが渡される. 直接
            BaseDatasetクラスのインスタンスに適用出来る用に, __init__に
            base_dataset以外の引数がある場合は, functools.partialを使用
            して引数を潰しておく必要がある.
        train_dataset: (Dataset): 学習用データセット.
        val_dataset: (Dataset): 検証用データセット.
        test_dataset: (Dataset): テスト用データセット.

    """

    Phase = Literal["train", "val", "test"]

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root_dir: pathlib.Path,
        dataset_stats: ImageDatasetStats,
        augmentation_config: DictConfig,
        data_retriever_class: Optional[Callable] = None,
    ) -> None:
        """BaseDataModuleクラスを初期化する.

        Args:
            batch_size(int): バッチサイズ.
            num_workers(int): データセットを読み込むワーカーの数.
            data_root_dir (pathlib.Path): 読み込むデータがおかれているルート
                ディレクトリのパス.
            dataset_stats (ImageDatasetStats): データセットの統計情報.
            augmentation_config (DictConfig): データ拡張の情報を保持した
                DictConfig. Hydraと組み合わせてYAMLファイルから読み込む想定.
            data_retriever_class (Optional[Callable]): BaseDatasetクラス
                のインスタンスに適用して, データを取得する方法を変えるクラス.
                例えば, VanillaTripletDataRetrieverなどが渡される. 直接
                BaseDatasetクラスのインスタンスに適用出来る用に, __init__に
                base_dataset以外の引数がある場合は, functools.partialを使用
                して引数を潰しておく必要がある.

        """
        super().__init__()
        self.batch_size: Final = batch_size
        self.num_workers: Final = num_workers
        self.data_root_dir: Final = data_root_dir
        self.dataset_stats: Final = dataset_stats
        self.augmentation_config: Final = augmentation_config
        self.data_retriever_class: Final = data_retriever_class

        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset

    @abstractmethod
    def prepare_data(  # type: ignore
        self,
        *args,
        **kwargs,
    ) -> None:
        """ディスクに書き込む可能性のある処理や, 分散処理において単一のGPUからのみ
        実行する必要がある処理を行う場合に使用する.
        """
        pass

    @abstractmethod
    def setup(  # type: ignore
        self,
        stage: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """全てのGPUで実行したいデータ操作等を記述する."""
        pass

    def train_dataloader(self) -> DataLoader:
        """trainフェーズにおけるデータローダーを取得する."""
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """valフェーズにおけるデータローダーを取得する."""
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """testフェーズにおけるデータローダーを取得する."""
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _get_transform(
        self,
        stage: Phase,
        normalize: bool,
        augmentation_config: DictConfig,
    ) -> ImageSequential:
        """ListConfigに格納された情報からデータ拡張の変換群を作成する.

        Args:
            stage (Phase): 学習フェーズ.
            normalize (bool): Trueならデータを正規化する.
            augmentation_config (DictConfig): データ拡張の情報を保持した
                DictConfig. Hydraと組み合わせてYAMLファイルから読み込む想定.

        Returns:
            ImageSequential: データ拡張の変換群.

        Note:
            augmentation_config["stage"]としたもの自体を引数とせずに, stageを
            引数に加えてこのメソッドの中で, augmentation_config["stage"]とする
            のは, このクラスを継承した小クラスにおいてこのメソッドの処理がstageの値
            に依存する形にオーバーライドされてもインタフェースが変わらないようにする
            ため.

        """
        # hydraのinstantiate関数を使用して再帰的にaugmentation_configから
        # データ拡張に使用する変換のクラスをインスタンス化する.
        transform = (
            instantiate(augmentation_config[stage])
            if augmentation_config[stage]
            else list()
        )

        if normalize:
            transform.append(
                Normalize(
                    mean=torch.tensor(self.dataset_stats.mean),
                    std=torch.tensor(self.dataset_stats.std),
                    keepdim=True,
                )
            )

        return ImageSequential(*transform)

    @classmethod
    def calculate_stats(
        cls, dataset: Iterable[Tuple[torch.Tensor, Any]]
    ) -> Tuple[RgbColor, RgbColor]:
        """RGB画像からなるデータセットについてRGBのそれぞれのチャネルについて平均と
        標準偏差を計算する.

        Args:
            dataset (Iterable[Tuple[torch.Tensor, Any]]): データセット.
                データセット内のテンソルは[0.0, 1.0]の間の値をとるものでなければ
                なりません.

        Returns:
            Tuple[RgbColor, RgbColor]: データセットの平均と標準偏差.

        """
        # 全データをスタックして[B,C,H,W]というshapeにする.
        stacked_data = torch.stack([data for data, _ in dataset])
        assert len(stacked_data.size()) == 4, "Shape is invalid."
        assert stacked_data.size(1) == 3, "Second dim should be color channel."
        assert stacked_data.min().item() >= 0.0, "Min should be >= 0.0."
        assert stacked_data.max().item() <= 1.0, "Max should be <= 1.0."

        # Colorチャネル以外にわたって平均と標準偏差を計算する.
        mean: Final = stacked_data.mean(dim=(0, 2, 3))
        std: Final = stacked_data.std(dim=(0, 2, 3))

        return RgbColor(*mean.tolist()), RgbColor(*std.tolist())


# TODO: このクラスの機能はBaseTripletDataRetrieverクラスとして
# src/retriever/base_triplet.pyに移動している. DFDC datasetの更新が完了次第
# このクラスは削除.
class BaseTripletDataset(ABC, Dataset):
    """全TripletDatasetクラスのベースとなるABC."""

    def __init__(self, base_dataset: BaseDataset) -> None:  # noqa: D107
        self.base_dataset = base_dataset

    def __len__(self) -> int:  # noqa: D105
        return len(self.base_dataset)

    @abstractmethod
    def __getitem__(self, index: int) -> TripletDataPoint:  # noqa: D105
        raise NotImplementedError

    @abstractmethod
    def _get_positive(self, index: int) -> DataPoint:
        """indexで取得されるanchorに対してpositiveのデータを取得する処理が実装
        されるメソッド.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_negative(self, index: int) -> DataPoint:
        """indexで取得されるanchorに対してnegativeのデータを取得する処理が実装
        されるメソッド.
        """
        raise NotImplementedError
