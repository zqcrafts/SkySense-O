import json
import torch
from torch import nn
from typing import Dict
from open_clip import tokenizer
from detectron2.layers import ShapeSpec
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from .visual_decoder import Aggregator
from skysense_o.modeling.backbone.skysense_clip import SkySenseCLIP
from skysense_o.third_party import clip


@SEM_SEG_HEADS_REGISTRY.register()
class SkySenseOHead(nn.Module):

    @configurable
    def __init__(self,
                 *,
                 text_guidance_dim: int,
                 text_guidance_proj_dim: int,
                 appearance_guidance_dim: int,
                 appearance_guidance_proj_dim: int,
                 decoder_dims: list,
                 decoder_guidance_dims: list,
                 decoder_guidance_proj_dims: list,
                 num_heads: int,
                 num_layers: tuple,
                 hidden_dims: tuple,
                 pooling_sizes: tuple,
                 feature_resolution: tuple,
                 window_sizes: tuple,
                 attention_type: str,
                 clip_cfg_path=None,
                 clip_ckpt_path=None,
                 ignore_value=65535):
        super().__init__()

        # Load CLIP Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # load skysense clip
        clip_model = SkySenseCLIP(clip_cfg_path).to(device)
        checkpoint = torch.load(clip_ckpt_path, map_location="cpu")
        checkpoint_new = checkpoint['clip'].copy()
        for k, v in checkpoint['model'].items():
            if k.startswith('backbone_gep.'):
                checkpoint_new[k.replace('backbone_gep.', 'visual.')] = v
        checkpoint = checkpoint_new
        msg = clip_model.load_state_dict(checkpoint, strict=False)
        self.clip_model = clip_model.float()
        # # load vitl clip
        # clip_model, _ = clip.load('/gruntdata/rs_nas/workspace/xingsu.zq/CAT-Seg/ViT-L-14-336px.pt', device=device, jit=False)
        # self.clip_model = clip_model.float()

        # Load Transformer Head
        self.transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads,
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes
        )
        self.text_cache = None
        self.text_tokens = None
        self.ignore_value = ignore_value

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            # CLIP Config
            "clip_cfg_path": cfg.MODEL.CLIP_CFG_PATH,
            "clip_ckpt_path": cfg.MODEL.CLIP_CKPT_PATH,
            # Head Config
            "text_guidance_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM,
            "text_guidance_proj_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM,
            "appearance_guidance_dim": cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM,
            "appearance_guidance_proj_dim": cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM,
            "decoder_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS,
            "decoder_guidance_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS,
            "decoder_guidance_proj_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS,
            "num_layers": cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS,
            "num_heads": cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS,
            "hidden_dims": cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS,
            "pooling_sizes": cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "window_sizes": cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES,
            "attention_type": cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        }

    def get_text_embeds(self, classnames, templates, clip_model):
        """
        Get text embeddings for classnames with cache mechanism.
        """
        # Get text tokens
        text_tokens = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = tokenizer.tokenize(texts).cuda()  # torch.Size([1, 77]) max_text_length =77
            text_tokens.append(texts)
        self.text_tokens = torch.stack(text_tokens, dim=0).squeeze(1)
        if self.text_tokens.dim() == 3:  # When multiple templates, the text_tokens is a 3D tensor
            t, p, n = self.text_tokens.shape
            self.text_tokens = self.text_tokens.reshape(t * p, n)
        # Get text embeddings
        class_embeddings = clip_model.encode_text(self.text_tokens)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        if len(templates) > 1:
            class_embeddings = class_embeddings.reshape(t, p, -1)
        else:
            class_embeddings = class_embeddings.unsqueeze(1)
        if not self.training:
            self.text_cache = class_embeddings

        return class_embeddings

    def forward(self, image_features, text_embeds, visual_guidance):
        """
        Args:
            img_feats: (B, C, H, W)
            guidance_features: (B, C, H, W)
        """
        out = self.transformer(image_features, text_embeds, visual_guidance)

        return out