# Copyright (c) Facebook, Inc. and its affiliates.
import os
import copy
import logging
import numpy as np
import torch
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["SkySenseODatasetMapper"]


class SkySenseODatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by SkySense-O for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    ignore_label
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            - index_label = "dataset": Indexing class labels from 0 to num_class of current image or from 0 to num_class of whole dataset.
            - index_label = "dataset_folder": Indexing class labels in folder format for each image.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "The SkySense-O DatasetMapper should only be used for training!"

        # Load image
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Load sem_seg_gt
        if dataset_dict["index_label"] == "dataset":
            dir_path = dataset_dict.pop("sem_seg_file_name") 
            sem_seg_gt = utils.read_image(dir_path).astype("double")
        elif dataset_dict["index_label"] == "dataset_folder":
            dir_path = dataset_dict.pop("sem_seg_file_name")
            sem_seg_gt = [] 
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                sem_seg_gt_ = utils.read_image(file_path).astype("double")
                sem_seg_gt.append(sem_seg_gt_)
            sem_seg_gt = np.stack(sem_seg_gt, axis=-1)
        else:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]))

        # Perform augmentation
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)  # require image shape: H,W,C
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            dataset_dict['ori_size'] = image_size
            padding_size = [
                0,
                self.size_divisibility - image_size[1], # w: (left, right)
                0,
                self.size_divisibility - image_size[0], # h: 0,(top, bottom)
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
        image_shape = (image.shape[-2], image.shape[-1])  # h, w
        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt.long()

        # # NOTE
        # for other_benchmark in ["potsdam", "samrs_fast", "samrs_sior", "samrs_sota"]: 
        #     if other_benchmark in dataset_dict["file_name"].split('/'):
        #         is_sparse_train = True
        #         dataset_dict["sem_seg"] = dataset_dict["sem_seg"].to(torch.uint8) - 1
        #         dataset_dict["sem_seg"] = dataset_dict["sem_seg"].long()
        #         # dataset_dict["class_list"] = [item for item in dataset_dict["class_list"] if item != 'others']
        #         break

        # Prepare per-category binary masks
        if sem_seg_gt is not None and "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
            dataset_dict["instances"] = instances

        return dataset_dict
