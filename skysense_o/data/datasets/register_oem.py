import os
import json
import logging
from detectron2.utils.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)

color_mapping = {}
class_text = 'datasets/oem/oem.json'
retrieved_result_path = 'datasets/oem/o1.json'
with open(class_text, 'r') as file:
    json_file = json.load(file)
    for index, key in  enumerate(json_file):
        color_mapping[key] = index

def _get_meta():
    stuff_ids = color_mapping.values()
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = list(color_mapping.keys())
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "class_text": class_text,
        "retrieved_result_path": retrieved_result_path
    }
    return ret

def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):

    def file2id(folder_path, file_path):
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
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
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]
    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )
    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        record["index_label"] = "dataset"
        dataset_dicts.append(record)
        
    return dataset_dicts

def register_oem(root):
    meta = _get_meta()
    root = os.path.join(root, "oem")
    for name, image_dirname, sem_seg_dirname in [
        ("train", "img_dir/train", "ann_dir/train"),
        ("test", "img_dir_crop/val", "ann_dir_crop/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"oem_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="png"),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,
            **meta
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_oem(_root)


