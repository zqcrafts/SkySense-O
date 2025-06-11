import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import json



# class_text = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/isaid/isaid_open_all.json"
class_text = "/gruntdata/rs_nas/workspace/xingsu.zq/dataset/RS5M_caption/sky5k/sky5k.json"
color_mapping = {}
with open(class_text, 'r') as file:
    json_file = json.load(file)
    for index, key in  enumerate(json_file):
        color_mapping[key] = index
# color_mapping.pop('others', None)


def _get_isaid_visual_demo_open_meta():
    stuff_ids = color_mapping.values()
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = list(color_mapping.keys())
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "class_text": class_text
    }
    return ret


def register_isaid_visual_demo_open(root):
    # root = os.path.join(root, "SkySA")
    meta = _get_isaid_visual_demo_open_meta()

    for name in ["train", "test"]:
        image_dir = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/demo_set/isaid"
        gt_dir = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/demo_set/isaid_ann"
        all_name = f"isaid_visual_demo_open_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="png"
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
register_isaid_visual_demo_open(_root)
