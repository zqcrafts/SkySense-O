# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
import os
sys.path.append('/gruntdata/rs_nas/workspace/xingsu.zq/detectron2-xyz-main')
sys.path.append('/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O')
os.environ['DETECTRON2_DATASETS'] = '/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O/datasets/'
os.chdir('/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O')
import copy
import itertools
import logging
import os
import torch
import json
import numpy as np
from PIL import Image  
from tabulate import tabulate
import torchvision.transforms as transforms
from typing import Any, Dict, List, Set
from collections import OrderedDict, defaultdict
from skysense_o import SkySenseODatasetMapper, SemanticSegmentorWithTTA
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
# from detectron2.data import build_detection_test_loader, build_detection_train_loader
from skysense_o.data import build_detection_test_loader, build_detection_train_loader
from predictor import SkySenseO_DEMO
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators, SemSegEvaluator, verify_results
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.file_io import PathManager
from detectron2.config import CfgNode as CN


"""
python demo/demo.py --dist-url auto --config-file configs/skysense_o_demo.yaml --dist-url 'auto' --eval-only --num-gpus 4
""" 

def init_config():
    """
    Build config node for SkySense-O.
    """
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


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                ))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type))
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
            
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):

        if cfg.INPUT.DATASET_MAPPER_NAME == "skysense_o_dataset_mapper":
            mapper = SkySenseODatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(
                    recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                hyperparams["param_name"] = ".".join(
                    [module_name, module_param_name])
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams[
                        "lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "clip_model" in module_name:
                    hyperparams[
                        "lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
                # for deformable detr
                if ("relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                        or "positional_embedding" in module_param_name
                        or "bias" in module_param_name):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                      and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                      and clip_norm_val > 0.0)

            class FullModelGradientClippingOptimizer(optim):

                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(
                torch.optim.SGD)(params,
                                 cfg.SOLVER.BASE_LR,
                                 momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(
                torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        # display the lr and wd of each param group in a table
        optim_info = defaultdict(list)
        total_params_size = 0
        for group in optimizer.param_groups:
            optim_info["Param Name"].append(group["param_name"])
            optim_info["Param Shape"].append("X".join(
                [str(x) for x in list(group["params"][0].shape)]))
            total_params_size += group["params"][0].numel()
            optim_info["Lr"].append(group["lr"])
            optim_info["Wd"].append(group["weight_decay"])
        # Counting the number of parameters
        optim_info["Param Name"].append("Total")
        optim_info["Param Shape"].append("{:.2f}M".format(total_params_size /
                                                          1e6))
        optim_info["Lr"].append("-")
        optim_info["Wd"].append("-")
        table = tabulate(
            list(zip(*optim_info.values())),
            headers=optim_info.keys(),
            tablefmt="grid",
            floatfmt=".2e",
            stralign="center",
            numalign="center",
        )
        logger = logging.getLogger("mask_former")
        logger.info("Optimizer Info:\n{}\n".format(table))
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg,
                                name,
                                output_folder=os.path.join(
                                    cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = init_config()
    cfg.set_new_allowed(True) 
    cfg.merge_from_file(args.config_file)
    cfg.set_new_allowed(True) 
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR,
                 distributed_rank=comm.get_rank(),
                 name="skysense_o")
    return cfg


def main(args):
    print("start")
    cfg = setup(args)
    torch.set_float32_matmul_precision("high")
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        output = model.run_demo()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
