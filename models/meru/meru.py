import torch.nn as nn
import torch

import timm
from hydra.utils import instantiate
from omegaconf import DictConfig
from .utils import CheckpointManager
from typing import Callable
from .text_encoders import TransformerTextEncoder
import math
from torch.nn import functional as F
from . import lorentz
from models.meru import distributed as dist


@timm.models.register_model
def vit_small_mocov3_patch16_224(**kwargs):
    """
    Small Vision Transformer used by MoCo-v3 (https://arxiv.org/abs/2104.02057)
    This model has 12 heads instead of 6, gives better empirical performance.
    """

    return timm.models.vision_transformer._create_vision_transformer(
        "vit_small_patch16_224",
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        **kwargs,
    )


def build_timm_vit(
    arch: str,
    global_pool: str = "token",
    use_sincos2d_pos: bool = True,
    grad_checkpointing: bool = False,
):
    """
    Build a Vision Transformer (ViT) image encoder from timm.

    Args:
        global_pool: How to perform global pooling after final layer? If this is
            `token` (default), we use the [cls] token. For `avg` we perform
            global average pooling like ConvNets.
        use_sincos2d_pos: Use a fixed 2D sine-cosine position embedding. If this
            is False; position embeddings are randomly initialized (and learned).
    """

    _supported = timm.list_models("vit_*")
    if arch not in _supported:
        raise ValueError(f"{arch} is not a supported ViT, choose: {_supported}")

    model = timm.create_model(
        arch,
        # `num_classes = 0` does not create the final classification head.
        num_classes=0,
        global_pool=global_pool,
        #
        # Do not use [CLS] token for models that use global average pooling.
        class_token=global_pool == "token",
        #
        # Use LayerNorm with default `eps = 1e-5` (timm uses 1e-6, not sure why)
        norm_layer=nn.LayerNorm,
    )

    model.set_grad_checkpointing(grad_checkpointing)

    # Set `width` attribute to access in model.
    model.width = model.embed_dim

    # ------------------------------------------------------------------------
    # 2D sine-cosine embedding for Vision Transformers. This implementation
    # is adapted from MoCo-v3 (https://github.com/facebookresearch/moco-v3) and
    # it produces exactly same embeddings.
    # ------------------------------------------------------------------------
    if use_sincos2d_pos:
        h, w = model.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)

        assert (
            model.embed_dim % 4 == 0
        ), "ViT embed_dim must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = model.embed_dim // 4

        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (10000.0**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        if global_pool == "token":
            pe_token = torch.zeros([1, 1, model.embed_dim], dtype=torch.float32)
            pos_emb = torch.cat([pe_token, pos_emb], dim=1)

        # Override position embedding.
        model.pos_embed.data.copy_(pos_emb)
        model.pos_embed.requires_grad = False
    return model


def callable_to_str(some_callable: Callable) -> str:
    # Return module and name of a callable (function or class) for OmegaConf.
    return f"{some_callable.__module__}.{some_callable.__qualname__}"


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed, but
    returns a dict that describes the call. Only supports keyword arguments.
    """

    def __init__(self, target: Callable):
        if not callable(target):
            raise TypeError(f"LazyCall target must be a callable! Got {target}")
        self.T = target

    def target_str(self):
        return

    def __call__(self, **kwargs):
        # Pop `_target_` if it already exists in kwargs. This happens when the
        # callable target is changed while keeping everything else same.
        _ = kwargs.pop("_target_", None)

        # Put current target first; it reads better in printed/saved output.
        kwargs = {"_target_": callable_to_str(self.T), **kwargs}
        return DictConfig(content=kwargs, flags={"allow_objects": True})


class CLIPBaseline(nn.Module):
    """
    Our re-implementation of the CLIP model that uses an image-text contrastive
    loss as a training objective and embeds images and text in a Euclidean space.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            visual: ConvNet or ViT image encoder to compute image features.
            textual: Transformer-based encoder to compute text features.
            embed_dim: Size of the visual and textual embedding vectors for
                computing pairwise similarity matrix.
            pixel_mean: Normalize input images by this color mean. Default value
                is of ImageNet color, set to `(0, 0, 0)` for no normalization.
            pixel_std: Normalize input images by this color std. Default value
                is of ImageNet color, set to `(1, 1, 1)` for no normalization.
        """
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim

        # Linear layers to project image and text features such that they have
        # same size before computing dot-product similarity.
        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)

        # CLIP-style initialization of projection layers.
        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # Color mean/std to normalize image.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # Get rank of current GPU process for gathering features.
        self._rank = dist.get_rank()

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Project features to a unit hypersphere through L2 normalization.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """
        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)

        if project:
            image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Project features to a unit hypersphere through L2 normalization.
        """

        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens

        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(tokens)

        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # Compute logits for image-text contrastive loss: cosine similarity.
        image_logits = _scale * image_feats @ all_text_feats.T
        text_logits = _scale * text_feats @ all_image_feats.T

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        batch_size = image_feats.shape[0]
        targets = torch.arange(batch_size, device=image_logits.device)
        targets = targets + batch_size * self._rank

        loss = 0.5 * (
            F.cross_entropy(image_logits, targets)
            + F.cross_entropy(text_logits, targets)
        )
        output_dict = {
            "loss": loss,
            "logging": {"contrastive_loss": loss, "logit_scale": _scale},
        }
        return output_dict


class MERU(CLIPBaseline):
    """
    Our MERU model, that modifies CLIP to embed images and text in a hyperbolic
    space. Modifications are as follows:

    1. Lift embeddings from encoders onto the Lorentz hyperboloid using
       exponential map operator, instead of projecting via L2-normalization.

    2. Modify the contrastive loss to use the negative Lorentzian distance as
       a similarity measure instead of cosine similarity.

    3. Use a textual entailment loss to enforce a partial-order relationship
       between paired text and image embeddings (Note that an equivalent loss
       for CLIP is not mathematically defined).
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `CLIPBaseline`.

        Args:
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
        """
        super().__init__(visual, textual, embed_dim, pixel_mean, pixel_std)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        image_feats = super().encode_image(images, project=False)

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                image_feats = lorentz.exp_map0(image_feats, self.curv.exp())

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = super().encode_text(tokens, project=False)

        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                text_feats = lorentz.exp_map0(text_feats, self.curv.exp())

        return text_feats

    def forward(
        self,
        images: torch.Tensor,
        tokens: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -lorentz.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -lorentz.pairwise_dist(text_feats, all_image_feats, _curv)

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = lorentz.oxy_angle(text_feats, image_feats, _curv)
            _aperture = lorentz.half_aperture(text_feats, _curv)
            entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }


if __name__ == "__main__":
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # modelがc_train
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
    model.visual.arch = "vit_small_mocov3_patch16_224"
    device = device or torch.cuda.current_device()
    model = instantiate(model).to(device).eval()
    print(model)
    CheckpointManager(model=model).load("pretrained_weights/meru_vit_s.pth")
