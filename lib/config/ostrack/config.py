from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.VDRM
cfg.MODEL.VDRM = edict()
cfg.MODEL.VDRM.ENABLED = False
cfg.MODEL.VDRM.INSERT_LAYER = 6  # one-based block number; applied after this block
cfg.MODEL.VDRM.NUM_PARTS = 4
cfg.MODEL.VDRM.TOPK = 4
cfg.MODEL.VDRM.RELIABILITY_MODE = "topk"
cfg.MODEL.VDRM.NMS_RADIUS = 1
cfg.MODEL.VDRM.INITIAL_MATCH_SCALE = 5.0
cfg.MODEL.VDRM.INITIAL_MATCH_BIAS = -2.5
# 0 disables the V4 relative-norm residual bound for V1/V2/V3 compatibility.
cfg.MODEL.VDRM.RESIDUAL_MAX_RATIO = 0.0
# V7 adds ``candidate_consensus``. V8 adds ``part_aligned``, which routes
# every template part at the same search tokens used to build its residual.
# ``token_match`` keeps the exact V1-V6 forward path and state-dict schema.
cfg.MODEL.VDRM.SPATIAL_GATE_MODE = "token_match"
cfg.MODEL.VDRM.CANDIDATE_LOCAL_RADIUS = 1
cfg.MODEL.VDRM.CANDIDATE_CONSENSUS_PARTS = 2
cfg.MODEL.VDRM.CANDIDATE_INITIAL_MATCH_SCALE = 5.0
cfg.MODEL.VDRM.CANDIDATE_INITIAL_MATCH_BIAS = -2.5
cfg.MODEL.VDRM.PART_ROUTE_INITIAL_MATCH_SCALE = 5.0
cfg.MODEL.VDRM.PART_ROUTE_INITIAL_MATCH_BIAS = -2.5
# 0 preserves the direct LayerScale used by V1-V7. V8 uses a positive bound
# to prevent a shrinking spatial gate from being offset by alpha growth.
cfg.MODEL.VDRM.ALPHA_MAX = 0.0

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone
cfg.TRAIN.VDRM_VISIBILITY_WEIGHT = 0.5
cfg.TRAIN.VDRM_RANK_WEIGHT = 0.5
cfg.TRAIN.VDRM_CANDIDATE_WEIGHT = 0.0
cfg.TRAIN.VDRM_PART_ROUTE_WEIGHT = 0.0
cfg.TRAIN.VDRM_PART_TARGET_DILATION = 1.0
# Keep the released V8 behavior by default. Ablation A5 disables this so
# structured occlusion changes only the search image, while A6 restores the
# visibility target as a weight on part-route supervision.
cfg.TRAIN.VDRM_ROUTE_VISIBILITY_WEIGHTED = True
cfg.TRAIN.VDRM_RANK_MARGIN = 0.1
cfg.TRAIN.VDRM_AUX_WARMUP_EPOCHS = 20
# V5-only switch. When enabled, HNCP samples take the rank negative from the
# pasted distractor box instead of an unrelated maximum over all background.
cfg.TRAIN.VDRM_ALIGN_DISTRACTOR_RANK = False
# Read-only V6 diagnostics; this never changes rank-negative selection.
cfg.TRAIN.VDRM_LOG_DISTRACTOR_HARDNESS = False

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.VDRM_OCCLUSION_PROBABILITY = 0.0
cfg.DATA.SEARCH.VDRM_OCCLUSION_MIN_RATIO = 0.2
cfg.DATA.SEARCH.VDRM_OCCLUSION_MAX_RATIO = 0.5
cfg.DATA.SEARCH.VDRM_DISTRACTOR_PROBABILITY = 0.0
cfg.DATA.SEARCH.VDRM_DISTRACTOR_MIN_SCALE = 0.7
cfg.DATA.SEARCH.VDRM_DISTRACTOR_MAX_SCALE = 1.3
cfg.DATA.SEARCH.VDRM_DISTRACTOR_PLACEMENT = "random"
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.CHECKPOINT_CONFIG = ""
cfg.TEST.VDRM_ALPHA_OVERRIDE = None


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
