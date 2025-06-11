import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import json
from detectron2.utils.file_io import PathManager
import logging
import numpy as np
from PIL import Image 

color_mapping = {}
class_text = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/SkySA/rs5m30k_wo_test_tag.json"
with open(class_text, 'r') as file:
    json_file = json.load(file)
    for index, key in  enumerate(json_file):
        color_mapping[key] = index


def _get_rs5m30k_meta():
    stuff_ids = color_mapping.values()
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = list(color_mapping.keys())

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "class_text": class_text
    }
    return ret

def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        # logger.warn(
        #     "Directory {} and {} has {} and {} files, respectively.".format(
        #         image_root, gt_root, len(input_files), len(gt_files)
        #     )
        # )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        # logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    # logger.info(
    #     "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    # )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path

        # class_number = np.unique(np.array(Image.open(img_path)))
        # class_number = class_number[(class_number != 255) & (class_number != 65535)]
        # class_number = len(class_number)
        # record["class_number"] = class_number
        dataset_dicts.append(record)

    return dataset_dicts


def register_rs5m30k(root):
    root = os.path.join(root, "SkySA")
    meta = _get_rs5m30k_meta()

    for name, image_dirname in [
        ("train", "img_dir"),
        ("test", "img_dir"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = "/gruntdata/rs_nas/workspace/xingsu.zq/dataset/RS5M_caption/aid30k_wo_test_tag/ann_dir"
        all_name = f"rs5m30k_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_rs5m30k(_root)
