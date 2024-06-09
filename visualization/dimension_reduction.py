from typing import cast

import numpy as np
from sklearn.manifold import TSNE  # type: ignore
from umap import UMAP  # type: ignore


def transform_data(
    data: np.ndarray,
    transform_algo: str,
    perplexity: float,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> np.ndarray:
    """Transform input data into an embedded space and return that
    transformed output.

    Args:
        data (np.ndarray): A input data whose shape have to be
            [number_of_samples, dimension_of_features].
        transform_algo (str): A name of the dimension reduction algorithm.
        perplexity (float): The perplexity is related to the number of
            nearest neighbors that is used in other manifold learning
            algorithms. Larger datasets usually require a larger
            perplexity. Consider selecting a value between 5 and 50.
        n_components (int): A dimension of the embedded space.
        n_neighbors (int): The size of local neighborhood (in terms of
            number of neighboring sample points) used for manifold
            approximation. Larger values result in more global views
            of the manifold, while smaller values result in more
            local data being preserved. In general values should be
            in the range 2 to
        min_dist (float): The effective minimum distance between
            embedded points. Smaller values will result in a more
            clustered/clumped embedding where nearby points on the
            manifold are drawn closer together, while larger values
            will result on a more even dispersal of points.
        metric (str): The metric to use to compute distances in high
            dimensional space. If “precomputed”, an affinity matrix is
            expected as input.

    Returns:
        np.ndarray: A transformed output whose shape should be
            [number_of_samples, n_components].

    """
    if transform_algo == "tsne":
        # 受け皿　＝　返値のある関数
        transformed_data = transform_tsne(
            data=data,
            perplexity=perplexity,
            n_components=n_components,
        )

    # 呼び出すときは引数に実際に代入する文字や数字を書く
    # 代入するのが変数でもいいよ
    elif transform_algo == "umap":
        transformed_data = transform_umap(
            data=data,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        )
    else:
        raise NotImplementedError(
            f"あなたが指定したアルゴリズムは{transform_algo}です。これは対応していません。"
        )
    # どっちもtransformed_dataで返す
    return transformed_data


def transform_tsne(
    data: np.ndarray,
    perplexity: float = 5.0,
    n_components: int = 2,
) -> np.ndarray:
    """Fit input data into an T-SNE embedded space and return that
    transformed output.

    Args:
        data (np.ndarray): A input data whose shape have to be
            [number_of_samples, dimension_of_features].
        perplexity (float): The perplexity is related to the number of
            nearest neighbors that is used in other manifold learning
            algorithms. Larger datasets usually require a larger
            perplexity. Consider selecting a value between 5 and 50.
        n_components (int): A dimension of the T-SNE embedded space.

    Returns:
        np.ndarray: A transformed output whose shape should be
            [number_of_samples, n_components].

    """
    tsne = TSNE(perplexity=perplexity, n_components=n_components, random_state=0)
    return cast(np.ndarray, tsne.fit_transform(data))


# 定義するときは変数の名前と型を書いておく
def transform_umap(
    data: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str = "correlation",
) -> np.ndarray:
    """Fit input data into an UMAP embedded space and return that
    transformed output.

    Args:
        data (np.ndarray): A input data whose shape have to be
            [number_of_samples, dimension_of_features].
        n_neighbors (int): The size of local neighborhood (in terms of
            number of neighboring sample points) used for manifold
            approximation. Larger values result in more global views
            of the manifold, while smaller values result in more
            local data being preserved. In general values should be
            in the range 2 to
        min_dist (float): The effective minimum distance between
            embedded points. Smaller values will result in a more
            clustered/clumped embedding where nearby points on the
            manifold are drawn closer together, while larger values
            will result on a more even dispersal of points.
        metric (str): The metric to use to compute distances in high
            dimensional space. If “precomputed”, an affinity matrix is
            expected as input.

    Returns:
        np.ndarray: A transformed output whose shape should be
            [number_of_samples, n_components].

    """
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    return cast(np.ndarray, umap.fit_transform(data))
