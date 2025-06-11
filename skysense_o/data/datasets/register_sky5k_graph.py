import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
import json


# For testing
def _get_sky5k_graph_meta():

    color_mapping = {}
    class_text = "/gruntdata/rs_nas/workspace/xingsu.zq/dataset/RS5M_caption/sky5k/sky5k.json"
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
    graph_: Dict: { source_entity: (target_entity, score) } 
    class_set: Set()
    """
    graph = dict()
    graph_ = dict()
    class_set = set()

    with open('datasets/SkySA/word_similar/word_similar_thre20_155.json', 'r') as f:
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
            if source_entity not in graph_.keys():
                graph[source_entity] = {'positive': set(), 'negative': set()}
                graph_[source_entity] = set()
            graph_[source_entity].add((target_entity, score))
            class_set.add(source_entity)
            class_set.add(target_entity)

        # wenshuo方案：取score分数从小往大排序的前150个为负样本，后面的为正样本
        # TODO: 选取方式有待优化
        # for key in graph_.keys():
        #     others = sorted(graph_[key], key=lambda x: x[1])
        #     for name, score in others:
        #         if len(graph[key]['negative']) >= 150:
        #             graph[key]['positive'].add(name)
        #         else:
        #             graph[key]['negative'].add(name)

        # 新方案：选取除了5-8分的难样本作为负样本
        for key in graph_.keys():
            others = sorted(graph_[key], key=lambda x: x[1])
            for name, score in others:
                if score >= 5 and score <= 8:
                    graph[key]['negative'].add(name)
                elif score >= 9:
                    graph[key]['positive'].add(name)

    return graph, class_set


graph, class_set = load_graph()


def load_sem_seg(file_path):

    dataset_dicts = []
    with open(file_path, 'r') as f:
        for lines in f:
            record = {}
            data = json.loads(lines.strip())
            img_path = data['flickr_url']
            # gt_dir = os.path.splitext(data['segmentation'])[0]
            positive = []
            for category in data['ann_list']:
                positive.append(category['category_name'])
            

            if len(positive) < 25:
                class_list = positive + ["others"]*(25-len(positive))
            
            print("class_list", len(class_list))
 
            record["file_name"] = img_path
            record["sem_seg_img"] = data['segmentation']
            record['class_list'] = class_list
            record['class_number'] = len(positive)
            dataset_dicts.append(record)

    return dataset_dicts


def register_sky5k_graph(root):

    meta = _get_sky5k_graph_meta()

    for name in ["train", "test"]:
        file_path = "/gruntdata/rs_nas/workspace/xingsu.zq/dataset/RS5M_caption/sky5k/ann_wash.json"
        all_name = f"sky5k_graph_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=file_path: load_sem_seg(x),
        )
        MetadataCatalog.get(all_name).set(
            file_path=file_path,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_sky5k_graph(_root)