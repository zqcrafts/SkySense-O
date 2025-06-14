import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
import json

graph_path = 'datasets/SkySA/word_similar/word_similar_thre20_155.json'
dataset_ann_json = '/gruntdata/rs_nas/workspace/xingsu.zq/dataset/RS5M_caption/aid30k_img_ann_thre20/decode_ann_itag.json'

# For testing
def _get_rs5m30k_graph_meta():

    color_mapping = {}
    class_text = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/SkySA/rs5m30k.json"
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


def load_sem_seg(file_path):

    dataset_dicts = []
    graph, class_set = load_graph()
    with open(file_path, 'r') as f:
        for lines in f:
            record = {}
            data = json.loads(lines.strip())
            img_path = data['flickr_url']
            gt_dir = os.path.splitext(data['segmentation'])[0]
            positive = []
            for category in data['ann_list']:
                positive.append(category['category_name'])
            if len(positive) == 0:
                continue
            negative = set()
            for i in range(len(positive)):
                if positive[i] not in graph.keys():
                    continue
                negative = negative | graph[positive[i]]['negative']
            for i in range(len(positive)):
                if positive[i] not in graph.keys():
                    continue
                negative = negative & graph[positive[i]]['negative']
            
            # TODO 添加正样本的选取，如果选上应该可以避免同语义的冲突
            if len(negative) > 0:
                negative = list(
                    np.random.choice(list(negative), size=min(100 - len(positive), len(negative)), replace=False))
            else:
                negative = list(
                    np.random.choice(list(class_set - set(positive)), size=100-len(positive), replace=False))
            class_list = positive + negative
            record["file_name"] = img_path
            record["sem_seg_dir"] = gt_dir
            record['class_list'] = class_list
            record['class_number'] = len(positive)
            dataset_dicts.append(record)

    return dataset_dicts


def register_rs5m30k_graph():

    meta = _get_rs5m30k_graph_meta()
    for name in ["train", "test"]:
        file_path = dataset_ann_json
        all_name = f"rs5m30k_graph_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=file_path: load_sem_seg(x),
        )
        MetadataCatalog.get(all_name).set(
            file_path=file_path,
            evaluator_type="sem_seg",
            ignore_label=65535,
            **meta,
        )

register_rs5m30k_graph()