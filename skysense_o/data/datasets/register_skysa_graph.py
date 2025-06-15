import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
import sys
sys.path.append(os.path.abspath(__file__))


hyper_use = False

if hyper_use == True:
    from detectron2.config import CfgNode as CN
    from detectron2.engine import default_argument_parser, default_setup
    def init_config():
        cfg = CN()
        cfg.INPUT = CN()
        cfg.INPUT.CROP = CN()
        cfg.DATASETS = CN()
        cfg.DATALOADER = CN()
        cfg.MODEL = CN()
        cfg.MODEL.SWIN = CN()
        cfg.MODEL.SEM_SEG_HEAD = CN()
        cfg.SOLVER = CN()
        cfg.SOLVER.AMP = CN()
        cfg.TEST = CN()
        cfg.VERSION = 2
        return cfg
    def setup(args):
        cfg = init_config()
        cfg.set_new_allowed(True) 
        cfg.merge_from_file(args.config_file)
        cfg.set_new_allowed(True) 
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        return cfg        
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    sampling_hyper = cfg.SAMPLING_HYPER
else:
    sampling_hyper = [1,3,9,10]

graph_path = '/gruntdata/rs_nas/workspace/xingsu.zq/SkySA_DataEngine/graph_maker/skysa_graph.json'
dataset_ann_json = '/gruntdata/rs_nas/workspace/xingsu.zq/SkySA_DataEngine/dataset/fused_dataset/decode_ann_itag_eng.json'

# For testing
def _get_skysa_graph_meta():

    color_mapping = {}
    class_text = '/gruntdata/rs_nas/workspace/xingsu.zq/dataset/RS5M_caption/sky5k/sky5k.json'
    with open(class_text, 'r') as file:
        json_file = json.load(file)
        for index, key in  enumerate(json_file):
            color_mapping[key] = index

    stuff_ids = color_mapping.values()
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = color_mapping.keys()

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "class_text": class_text
    }
    return ret


def load_graph():
    """
    Load Graph

    graph: Dict: { source_entity: {'positive': set(), 'negative': set()} }
    graph_with_score: Dict: { source_entity: { target_entity: score} } 
    class_set: Set()
    """
    graph = dict()
    graph_with_score = dict()
    class_set = set()

    with open(graph_path, 'r') as f:
        for lines in f:
            data = lines.strip().split('{tuple_delimiter}')
            if len(data) == 0:
                continue
            if data[0].replace('"', '').replace('(', '') != 'relationship':
                continue
            source_entity = data[1].replace('"', '').replace("'", '')
            target_entity = data[2].replace('"', '').replace("'", '')
            try:
                score = int(data[3].replace('"', ''))
            except:
                continue
            if source_entity not in graph_with_score.keys():
                graph[source_entity] = {'positive': set(), 'negative': set()}
                graph_with_score[source_entity] = set()
            graph_with_score[source_entity].add((target_entity, score))
            class_set.add(source_entity)
            class_set.add(target_entity)

        
        neg_thre_bottom, neg_thre_top, pos_thre_bottom, pos_thre_top = sampling_hyper
        for key in graph_with_score.keys():
            others = sorted(graph_with_score[key], key=lambda x: x[1])
            for name, score in others:
                if neg_thre_bottom <= score <= neg_thre_top:
                    graph[key]['negative'].add(name)
                elif pos_thre_bottom <= score <= pos_thre_top:
                    graph[key]['positive'].add(name)
    
    for key in graph_with_score.keys():
        graph_with_score[key] = {key: value for key, value in graph_with_score[key]}

    return graph, class_set, graph_with_score

graph, class_set, graph_with_score = load_graph()

def load_sem_seg(file_path):

    dataset_dicts = []
    with open(file_path, 'r') as f:
        for lines in f:
            record = {}
            data = json.loads(lines.strip())
            img_path = data['flickr_url']
            gt_dir = data['segmentation']
            # Classes in the current image.
            positive = []
            for category in data['ann_list']:
                positive.append(category['category_name'])
            if len(positive) == 0: # empty class
                continue
            # Negative sampler: Only maintain the unrelated categories to all positive sample categories.
            # TODO Add positive sampler
            negative = set()
            for i in range(len(positive)):
                if positive[i] not in graph.keys():
                    continue
                negative = negative | graph[positive[i]]['negative']
            for i in range(len(positive)):
                if positive[i] not in graph.keys():
                    continue
                negative = negative & graph[positive[i]]['negative']

            record["file_name"] = img_path
            record["sem_seg_file_name"] = gt_dir
            record["index_label"] = "dataset"
            record["positive"] = positive
            record["negative"] = negative
            record["class_set"] = class_set
            record["dynamic_sampler"] = True
            dataset_dicts.append(record)

    return dataset_dicts


def register_skysa_graph():

    meta = _get_skysa_graph_meta()
    for name in ["train", "test"]:
        file_path = dataset_ann_json
        all_name = f"skysa_graph_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=file_path: load_sem_seg(x),
        )
        MetadataCatalog.get(all_name).set(
            file_path=file_path,
            evaluator_type="sem_seg",
            ignore_label=255,
            graph_with_score=graph_with_score,
            **meta,
        )

register_skysa_graph()

