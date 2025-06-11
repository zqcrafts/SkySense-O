import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import json


color_mapping = {}
class_text = '/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/coco-stuff/coco.json'
with open(class_text, 'r') as file:
    json_file = json.load(file)
    for index, key in  enumerate(json_file):
        color_mapping[key] = index

def _get_coco_meta():

    stuff_ids = color_mapping.values()
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = list(color_mapping.keys())
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "class_text": class_text
    }
    return ret


def register_coco(root):

    root = os.path.join(root, "coco-stuff")
    meta = _get_coco_meta()

    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train2017", "annotations_detectron2/train2017"),
        ("test", "images/val2017", "annotations_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)

        all_name = f"coco_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco(_root)
