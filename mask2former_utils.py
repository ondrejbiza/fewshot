from dataclasses import dataclass
from typing import Tuple

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from prereq.Mask2Former.mask2former import add_maskformer2_config
from detectron2.engine.defaults import DefaultPredictor


@dataclass
class COCOInstanceArgs:
    config_file: str = "prereq/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
    opts: Tuple[str, ...] = ("MODEL.WEIGHTS", "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl")
    confidence_threshold: float = 0.5


@dataclass
class COCOPanopticArgs:
    config_file: str = "prereq/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
    opts: Tuple[str, ...] = ("MODEL.WEIGHTS", "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl")
    confidence_threshold: float = 0.5


@dataclass
class ADE20kInstanceArgs:
    config_file: str = "prereq/Mask2Former/configs/ade20k/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml"
    opts: Tuple[str, ...] = ("MODEL.WEIGHTS", "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/instance/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_92dae9.pkl")
    confidence_threshold: float = 0.5


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_predictor(model_args):

    cfg = setup_cfg(model_args)
    return DefaultPredictor(cfg)
