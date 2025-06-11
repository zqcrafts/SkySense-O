import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from einops import rearrange
from .text_encoder import TextTransformer, text_global_pool
from .visual_encoder import SwinTransformerV2


class SkySenseCLIP(nn.Module):
    """
    SkySenseCLIP is a modified version of SkySense and CLIP, which is used for zero-shot classification and semantic segmentation.
    """
    def __init__(self, cfg_path, init_logit_scale: float = np.log(1 / 0.07)):
        super().__init__()
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)['clip']
        embed_dim = cfg['embed_dim']
        init_logit_bias = cfg['init_logit_bias']
        context_length = cfg['text_cfg']['context_length']
        vocab_size = cfg['text_cfg']['vocab_size']
        width = cfg['text_cfg']['width']
        heads = cfg['text_cfg']['heads']
        layers = cfg['text_cfg']['layers']
        swin_params = cfg['vision_cfg'].copy()
        swin_params.pop('pool_dim')
        self.visual = SwinTransformerV2(**swin_params)
        text = TextTransformer(context_length=context_length,
                               vocab_size=vocab_size,
                               width=width,
                               heads=heads,
                               layers=layers,
                               output_dim=embed_dim)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        scale = cfg['vision_cfg']['pool_dim']**-0.5
        self.vision_projection = nn.Parameter(
            scale * torch.randn(cfg['vision_cfg']['pool_dim'], embed_dim))
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)  # 1.155
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        # if 'init_cfg' in cfg.keys() and cfg['init_cfg'] is not None and cfg[
        #         'init_cfg']['checkpoint'] is not None:
        #     self.load_pretrained(cfg['init_cfg']['checkpoint'])
        # for n, p in self.named_parameters():
        #     if 'visual' in n:
        #         # print(n, '-->', p.requires_grad)
        #         continue
        #     else:
        #         p.requires_grad = False
        #         # print(n, '-->', p.requires_grad)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False, dense=False):
        image_features = self.visual(image)
        if not dense:
            image_features = image_features[-1]
            image_features = F.adaptive_avg_pool2d(image_features, output_size=1).squeeze()
            image_features = image_features @ self.vision_projection
            return F.normalize(image_features, dim=-1) if normalize else image_features
        else:
            image_feature = image_features[-1]
            image_feature = image_feature.flatten(start_dim=2).permute(0, 2, 1)
            image_feature = image_feature @ self.vision_projection
            image_feature = rearrange(image_feature, "B (H W) C -> B C H W", H=np.sqrt(image_feature.shape[1]).astype(int))
            if normalize:
                image_feature = F.normalize(image_feature, dim=-1)
            image_features[-1] = image_feature
            return image_features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        with torch.cuda.amp.autocast(enabled=False):
            text_features = self.encode_text(text, normalize=True)
            image_logits = self.logit_scale.exp(
            ) * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(
            image, normalize=True) if image is not None else None
        with torch.cuda.amp.autocast(enabled=False):
            text_features = self.encode_text(
                text, normalize=True) if text is not None else None
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(
            ), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()

    def load_pretrained(self, ckpt_path):
        pretrained_dict = torch.load(ckpt_path, map_location={'cuda:0': 'cpu'})
        missing_keys, unexpected_keys = self.load_state_dict(pretrained_dict,
                                                             strict=False)
        print('clip missing_keys:', missing_keys)
        print('clip unexpected_keys:', unexpected_keys)