import torch
import json
import cv2
import numpy as np
from typing import Tuple
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from einops import rearrange
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from detectron2.data import MetadataCatalog


@META_ARCH_REGISTRY.register()
class SkySenseO(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        sliding_window: bool,
        clip_finetune: str,
        regular_weight: float,
        pixel_weight: float,
        region_contrast_weight: float,
        contrast_scale: int,
        test_dataset_names: str,
        train_dataset_names: str,
        open_world_interpretation: str,
        prompt_ensemble_type: str,
        retrieval_augmentation: str,
        text_batch: int,
    ):
        super().__init__()
        # Input Image Config
        self.size_divisibility = size_divisibility
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        # Input Text Config
        self.test_class_texts = {}
        self.retrived_texts = {}
        self.retrieval_nums = {}
        self.retrieval_augmentation = retrieval_augmentation
        self.text_batch = text_batch
        # import pdb; pdb.set_trace()
        for test_dataset_name in test_dataset_names:
            meta_test = MetadataCatalog.get(test_dataset_name)
            if open_world_interpretation:
                test_class_json = open_world_interpretation
            else:
                test_class_json = meta_test.class_text
            with open(test_class_json, 'r') as f_in:
                self.test_class_texts[test_dataset_name] = json.load(f_in)
                
            if self.retrieval_augmentation:
                retrieved_result_path = meta_test.retrieved_result_path
                with open(retrieved_result_path, 'r') as rap:
                    retrieved_result = json.load(rap)
                    retrieval_num = [len(open_add_values) for open_add_keys, open_add_values in retrieved_result.items() if open_add_keys != "except"]  
                    if "except" in retrieved_result.keys():
                        retrieved_result.pop("except")           
                    retrived_text = [item for sublist in retrieved_result.values() for item in sublist]
                self.retrieval_nums[test_dataset_name] = retrieval_num
                self.retrived_texts[test_dataset_name] = retrived_text

        self.train_class_texts = {}
        self.train_ignore_labels = {}
        self.graph_with_score = {}
        for train_dataset_name in train_dataset_names:
            meta_train = MetadataCatalog.get(train_dataset_name)
            train_class_json = meta_train.class_text
            with open(train_class_json, 'r') as f_in:
                self.train_class_texts[train_dataset_name] = json.load(f_in)
            self.train_ignore_labels[train_dataset_name] = meta_train.ignore_label
            if "graph" in train_dataset_name:
                self.graph_with_score = meta_train.graph_with_score

        self.prompt_ensemble_type = prompt_ensemble_type
        if self.prompt_ensemble_type == "remote_sensing":
            self.prompt_templates = ['A remote sensing image of a {}.']
        else:
            print(prompt_ensemble_type)
            raise NotImplementedError
        # SemSeg Head
        self.sem_seg_head = sem_seg_head
        # CLIP Finetuning Config
        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.clip_model.named_parameters():
            if "transformer" in name or "visual" in name or "position" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                # Fine-tuning for attention blocks    
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                        params.requires_grad = True if "in_proj" in name or "qkv" in name or "q_bias" in name or "v_bias" in name or "cpb_mlp" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    elif 'visual.norm0.' in name or 'visual.norm1.' in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False
        # Connector Config
        self.clip_resolution = (384, 384)
        self.contrast_scale = contrast_scale
        self.pixel_shuffle_conv0 = nn.Conv2d(1024, 1024*256, kernel_size=1, stride=1)
        self.pixel_shuffle_conv1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        if self.contrast_scale > 1:
            self.pixel_shuffle_conv_scale = nn.Conv2d(1024, 1024*contrast_scale*contrast_scale, kernel_size=1, stride=1)
        self.region_contrast_negative_sample = "all_dataset" # or "current_image"
        self.region_contrast_mask = "gt"
        self.visual_guidance_conv1 = nn.Conv2d(704, 256, kernel_size=1, stride=1)
        self.visual_guidance_conv2 = nn.Conv2d(352, 128, kernel_size=1, stride=1)
        # Loss Config
        self.tokennizer_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.regular_weight = regular_weight
        self.pixel_weight = pixel_weight
        self.region_contrast_weight = region_contrast_weight
        # Sliding-Window Inference
        self.sliding_window = sliding_window   
            
    @classmethod
    def from_config(cls, cfg):
        sem_seg_head = build_sem_seg_head(cfg, None)
        return {
            # Input Image Config
            "size_divisibility": cfg.MODEL.SIZE_DIVISIBILITY,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            # Input Text Config
            "test_dataset_names": cfg.DATASETS.TEST,
            "train_dataset_names": cfg.DATASETS.TRAIN,
            "prompt_ensemble_type": cfg.MODEL.PROMPT_ENSEMBLE_TYPE,
            "open_world_interpretation": cfg.DATASETS.OPEN_WORLD_INTERPRETATION,
            "text_batch": cfg.TEXT_BATCH,
            # CLIP Finetuning Config
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            # Head Config
            "sem_seg_head": sem_seg_head,
            # Loss Config
            "contrast_scale": cfg.CONTRAST_SCALE,
            "region_contrast_weight": cfg.REGION_CONTRAST_WEIGHT,
            "regular_weight": cfg.REGULAR_WEIGHT,
            "pixel_weight": cfg.PIXEL_CONTRAST_WEIGHT,
            # Retrieval-Augmented Inference
            "retrieval_augmentation": cfg.DATASETS.RETRIEVAL_AUGMENTATION,
            # Sliding-Window Inference
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
        }

    @property
    def device(self):
        return self.clip_pixel_mean.device

    def forward(self, batched_inputs):
        """
        sh run_train.sh -a -n base_hyper_5_8_9_10_a -h "[5,8,9,10]"
        text_batch = 100
        ------------------------------------------------------------------
        Baseline:
        100 (Default) |  11.8  |  48.1  |  2.4  |  10.7  |  9.0  | 
        ------------------------------------------------------------------
            Hypara ablation   |  isaid | potsdam | fast |  sior  |  sota       
        ------------------------------------------------------------------
        50            |   9.7 ｜ 
        80            |  11.9
        100 (Default) |  11.0
        120           |  10.7
        200           |  11.8
        ------------------------------------------------------------------
        """
        assert len(batched_inputs) == 1            
        # Sliding-Window Inference
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)
        # Image Preprocessing
        images = [x["image"].to(self.device) for x in batched_inputs]
        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        # Text Preprocessing
        if self.training:
            if 'dynamic_sampler' in batched_inputs[0].keys(): # dynamic text sampling in training
                positive = batched_inputs[0]["positive"]
                negative = batched_inputs[0]["negative"]
                class_set = batched_inputs[0]["class_set"]
                if len(negative) > 0:
                    negative = list(
                        np.random.choice(list(negative), size=min(self.text_batch-len(positive), len(negative)), replace=False))
                else:
                    print(f"{batched_inputs[0]['meta']['dataset_name']} can not be negative sampled")
                    negative = list(
                        np.random.choice(list(class_set - set(positive)), size=self.text_batch-len(positive), replace=False))
                self.text = positive + negative
            else:
                self.text = self.train_class_texts[batched_inputs[0]["meta"]["dataset_name"]]
        else:
            self.test_class_text = self.test_class_texts[batched_inputs[0]["meta"]["dataset_name"]]
            # import pdb; pdb.set_trace()
            if self.retrieval_augmentation:
                self.text = self.test_class_text + self.retrived_texts[batched_inputs[0]["meta"]["dataset_name"]]
            else:
                self.text = self.test_class_text
        # Visual Features from SkySense-O backbone
        clip_features = self.sem_seg_head.clip_model.encode_image(clip_images_resized, dense=True)
        image_feats = clip_features[-1]
        res2 = self.visual_guidance_conv1(clip_features[-3])
        res1 = self.visual_guidance_conv2(clip_features[-4])  
        visual_guidance = [image_feats, res2, res1]
        # Text Features from SkySense-O backbone
        text_feats = self.sem_seg_head.get_text_embeds(self.text, self.prompt_templates, self.sem_seg_head.clip_model)
        text_feats = text_feats.repeat(image_feats.shape[0], 1, 1, 1)
        # Visual Decoder
        outputs = self.sem_seg_head(image_feats, text_feats, visual_guidance)
        # Optimization or Inference
        if self.training:
            # Process outputs
            outputs = F.interpolate(outputs, size=self.clip_resolution, mode="bilinear", align_corners=False) # torch.Size([1, 218, 384, 384])
            outputs = outputs.permute(0, 2, 3, 1) # torch.Size([1, 384, 384, 218])
            # Process targets
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0) # torch.Size([1, 384, 384])
            targets_onehot = torch.zeros(outputs.shape, device=self.device)
            if batched_inputs[0]["index_label"]=="dataset":
                num_classes = outputs.shape[-1] # 218
                ignore_mask = targets != self.train_ignore_labels[batched_inputs[0]["meta"]["dataset_name"]]  # torch.Size([1, 384, 384])
                onehot = F.one_hot(targets[ignore_mask], num_classes=num_classes).float().to(self.device) # torch.Size([147456, 218])
                targets_onehot[ignore_mask] = onehot # torch.Size([1, 384, 384, 218])
            elif batched_inputs[0]["index_label"]=="dataset_folder":
                num_classes = targets.shape[-1]
                targets_onehot[:, :, :, :num_classes][targets == 1] = 1  # For targets, 255: background; 1: foreground.
            # Pixel-level Finetuning for pixel space
            loss = F.binary_cross_entropy_with_logits(outputs, targets_onehot) * self.pixel_weight
            # Region-level Finetuning for latent space
            region_gt_index = torch.nonzero(targets_onehot.clone().view(-1, targets_onehot.shape[-1]).sum(dim=0)) # torch.Size([1, 2])  torch.Size([9, 1])
            region_gt_num = region_gt_index.size(0) # 9
            region_gt_index = region_gt_index.squeeze() # tensor([ 0,  2,  6,  8,  9, 23, 25, 56, 60], device='cuda:0')
            # wo/ sampling
            # loss = F.binary_cross_entropy_with_logits(outputs[:, :, :, region_gt_index], targets_onehot[:, :, :, region_gt_index]) * self.pixel_weight
            if (self.region_contrast_weight or self.regular_weight) and region_gt_num > 1:
                # region_mask
                region_mask = targets_onehot if self.region_contrast_mask == "gt" else outputs
                if self.region_contrast_negative_sample == "current_image":
                    region_mask = region_mask[:, :, :, region_gt_index]                
                if self.contrast_scale > 1:
                    region_mask = rearrange(region_mask, "B H W Cls -> B Cls H W")
                    region_mask = F.interpolate(region_mask, size=region_mask.shape[-1]*self.contrast_scale, mode='bilinear')
                    region_mask = rearrange(region_mask, "B Cls H W -> Cls (B H W)")
                else:
                    region_mask = rearrange(region_mask, "B H W Cls -> Cls (B H W)")
                # image_feats
                upscale_factor = 16
                image_feats_resized = self.pixel_shuffle_conv0(image_feats)
                image_feats_resized = F.pixel_shuffle(image_feats_resized, upscale_factor)
                image_feats_resized = self.pixel_shuffle_conv1(image_feats_resized)
                if self.contrast_scale > 1:
                    image_feats_resized = self.pixel_shuffle_conv_scale(image_feats_resized)
                    image_feats_resized = F.pixel_shuffle(image_feats_resized, self.contrast_scale)
                image_feats_resized = rearrange(image_feats_resized, "B C H W -> (B H W) C")
                # Mask Pooling --> Cls * C
                region_feats = region_mask @ image_feats_resized  
                region_feats = F.normalize(region_feats, dim=-1)
                text_feats = F.normalize(text_feats.squeeze(), dim=-1)
                region_text_logits = region_feats @ text_feats.T
                region_text_logits = (region_text_logits - region_text_logits.min()) / (region_text_logits.max() - region_text_logits.min())
                # Region-level contrastive learning
                if self.region_contrast_weight:
                    region_contrast_label = torch.zeros_like(region_text_logits)
                    for idx in region_gt_index:
                        region_contrast_label[idx, idx] = torch.tensor(1).to(self.device)
                    region_contrast_loss = F.binary_cross_entropy_with_logits(region_text_logits, region_contrast_label) * self.region_contrast_weight 
                else:
                    region_contrast_loss = torch.tensor(0).to(self.device)
                # Regularization for latent space
                if self.regular_weight:
                    if self.region_contrast_negative_sample == "current_image":
                        text_input = [self.text[i] for i in region_gt_index.cpu().tolist()]
                    else:
                        text_input = self.text
                    text_correlation = cosine_similarity(self.tokennizer_model.encode(text_input))
                    text_correlation = torch.tensor(text_correlation, dtype=torch.float32).to(self.device)
                    regularization = text_correlation * region_text_logits
                    if 'dynamic_sampler' in batched_inputs[0].keys():
                        vision_centric_similarity = torch.zeros_like(regularization)
                        for source_entity_index, source_entity in enumerate(self.text):
                            for target_entity_index, target_entity in enumerate(self.text):
                                # import pdb; pdb.set_trace()
                                try:
                                    vision_centric_similarity[source_entity_index][target_entity_index] = torch.tensor(11 - self.graph_with_score[source_entity][target_entity]).to(self.device)
                                except:
                                    vision_centric_similarity[source_entity_index][target_entity_index] = torch.tensor(0).to(self.device)
                        regularization = regularization * vision_centric_similarity
                    regularization = regularization[~torch.eye(len(self.text), dtype=torch.bool)]
                    regularization_loss = F.l1_loss(regularization, torch.zeros_like(regularization)) * self.regular_weight
                else:
                    regularization_loss = torch.tensor(0).to(self.device)
            else:
                region_contrast_loss = torch.tensor(0).to(self.device)
                regularization_loss = torch.tensor(0).to(self.device)

            losses = {"loss_sem_seg": loss, "region_contrast_loss": region_contrast_loss, "regularization_loss": regularization_loss}

            return losses
            
        elif not self.training:
            outputs = outputs.sigmoid()
            test_classes_num = len(self.test_class_text)
            if self.retrieval_augmentation:
                edge_num_counter = 0 
                self.retrieval_num = self.retrieval_nums[batched_inputs[0]["meta"]["dataset_name"]]
                for node_index, edge_num in enumerate(self.retrieval_num):
                    if edge_num and node_index!=(len(self.retrieval_num)-1):
                        outputs[0,node_index,:,:] = torch.max(torch.cat((outputs[0, node_index].unsqueeze(0), 
                                                                         outputs[0, (test_classes_num + edge_num_counter):(test_classes_num + edge_num_counter + edge_num)])),
                                                                         dim=0)[0]
                        edge_num_counter = edge_num_counter + edge_num
                    elif edge_num and node_index==(len(self.retrieval_num)-1):
                        outputs[0,node_index,:,:] = torch.max(torch.cat((outputs[0, node_index].unsqueeze(0), 
                                                                         outputs[0, (test_classes_num + edge_num_counter):])), dim=0)[0]
            outputs = outputs[:,:test_classes_num,:,:]
            output_height = batched_inputs[0].get("height")
            output_width = batched_inputs[0].get("width")
            output = sem_seg_postprocess(outputs[0], clip_images.image_sizes[0], output_height, output_width)
            output = [{'sem_seg': output}]

            return output