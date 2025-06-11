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

        # wenshuo方案：取score分数从小往大排序的前150个为负样本，后面的为正样本
        # TODO: 选取方式有待优化
        # for key in graph_with_score.keys():
        #     others = sorted(graph_with_score[key], key=lambda x: x[1])
        #     for name, score in others:
        #         if len(graph[key]['negative']) >= 150:
        #             graph[key]['positive'].add(name)
        #         else:
        #             graph[key]['negative'].add(name)

        # 新方案：选取除了5-8分的难样本作为负样本
        """
        *================================================================*
                Choose Positive and Negative Samples by Threshold
        *================================================================*
        Hypara - List: [neg_thre_bottom, neg_thre_top, pos_thre_bottom, pos_thre_top] in [1, 10]   
        the sample whose score is in [neg_thre_bottom, neg_thre_top] is seen as **negative sample**.
                                ; in [pos_thre_bottom, pos_thre_top] is seen as **positive sample**. 
        
        Exer_1: Wo/ rag:
        sh run_train.sh -n base_hyper_5_8_9_10 -h "[5,8,9,10]"
        ------------------------------------------------------------------
            Hypara ablation   |  isaid | potsdam | fast |  sior  |  sota |      
        ------------------------------------------------------------------
        [1,8,9,10] (Default)  |  11.8  |  48.1  |  2.4  |  10.7  |  9.0  |   这个效果明显很差，负样本选多了
        [1,5,9,10]            |  23.1  |  54.6  |  3.4  |  24.3  |  15.4 |   稍微负样本放宽一些就好了
        [1,3,9,10]            |  18.1  |  55.7  |  2.9  |  17.4  |  12.9 |   继续放宽肯定又不好了
        [1,3,7,10]            |  20.3  |  49.6  |  4.0  |  22.9  |  15.3 |   正样本放宽一点还可以
        ------------------------------------------------------------------
        conclusion: maybe [1, 5, 7, 10] is better
        ------------------------------------------------------------------
        [1,5,7,10]            |  
        ------------------------------------------------------------------
        
        Exer_2: W/ rag:
        sh run_train.sh -a -n base_hyper_5_8_9_10_a -h "[5,8,9,10]"
        ------------------------------------------------------------------
            Hypara ablation   |  isaid | potsdam | fast |  sior  |  sota |      
        ------------------------------------------------------------------
        [5,8,9,10]            |  26.7(负样本数量不够) |  
        [1,8,9,10]  (Default) |  37.0  |
        [1,7,9,10]            |  36.7  |
        [1,5,9,10]            |  37.1  |
        [1,3,9,10]  (Choose)  |  38.8  |
        [1,5,7,10]            |  36.2  |
        ------------------------------------------------------------------
        前8k-iter到最大值了,后面越训越低,低到20+了
        这么看isaid低的原因也不是超参数了呀。
        难道就是那些噪声影响的？你随机取100 batch，肯定有噪声。
        调整一下batch的数量试试。

        """
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

