import sys
import os
sys.path.append('/gruntdata/rs_nas/workspace/xingsu.zq/detectron2-xyz-main')
os.environ['DETECTRON2_DATASETS'] = '/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O/datasets/'
os.chdir('/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O')
from . import data  
from . import modeling
from .skysense_o_model import SkySenseO
from .data.dataset_mappers.skysense_o_dataset_mapper import SkySenseODatasetMapper
from .test_time_augmentation import SemanticSegmentorWithTTA

