if __name__ == "__main__":
    from PIL import Image
    import requests
    import numpy as np
    import torch
    import pathlib
    from torchvision.utils import save_image
    from datasets import load_dataset
    from io import BytesIO

    elsa_data = load_dataset("elsaEU/ELSA_D3", split="validation", streaming=True)
    print(type(elsa_data))

    real_path = pathlib.Path("elsa_data/0_real")
    real_path.mkdir(parents=True, exist_ok=True)
    fake_path = pathlib.Path("elsa_data/1_fake")
    fake_path.mkdir(parents=True, exist_ok=True)

    for sample in elsa_data:  # 1つのデータに対して2つの画像があるため
        id = sample.pop("id")
        for label in range(2):
            if label == 0:
                url_real_image = sample.pop("url")
                # URLが無効だった場合にrequests.getが失敗するのでtry節で
                # 挟みます。
                try:
                    response = requests.get(url_real_image, timeout=5)
                    # status codeが200系以外のときはREAL画像が取得出来ないので
                    # continueする。
                    if not response:
                        continue

                    image = (
                        np.array(Image.open(BytesIO(response.content)).convert("RGB"))
                        .astype(np.float32)
                        .transpose(2, 0, 1)
                    )
                    save_image(
                        torch.from_numpy(image / 255.0),
                        real_path / f"{id}-real.png",
                    )
                    print("real")

                except Exception:
                    continue

            else:
                image = (
                    np.array(sample.pop("image_gen2").convert("RGB"))
                    .astype(np.float32)
                    .transpose(2, 0, 1)
                )
                save_image(
                    torch.from_numpy(image / 255.0),
                    fake_path / f"{id}-fake.png",
                )
                print("fake")
