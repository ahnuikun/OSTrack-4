import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


_VDRM_AUXILIARY_WEIGHTS = (
    "VDRM_VISIBILITY_WEIGHT",
    "VDRM_RANK_WEIGHT",
    "VDRM_CANDIDATE_WEIGHT",
    "VDRM_PART_ROUTE_WEIGHT",
)


def validate_vdrm_experiment_contract(cfg, actual_seed=None):
    """Fail fast when a named clean-ablation arm is not actually clean."""
    arm = str(getattr(cfg.TRAIN, "VDRM_EXPERIMENT_ARM", "")).strip().lower()
    if not arm:
        if getattr(cfg.TRAIN, 'VDRM_FROZEN_BASE_CONFIG', '') or getattr(cfg.MODEL.VDRM, 'DISCRIMINATIVE_ROUTE', False):
            raise ValueError('frozen/discriminative options require a named Rfreeze/Rdisc arm')
        return
    if arm not in {"tclean", "ronly", "rfreeze", "rdisc"}:
        raise ValueError(
            "TRAIN.VDRM_EXPERIMENT_ARM must be empty, 'tclean', or "
            f"'ronly', 'rfreeze', or 'rdisc', got {arm!r}"
        )

    required_seed = getattr(cfg.TRAIN, "VDRM_REQUIRED_SEED", None)
    if required_seed is None:
        raise ValueError(f"{arm} requires TRAIN.VDRM_REQUIRED_SEED")
    if actual_seed is None or int(actual_seed) != int(required_seed):
        raise ValueError(
            f"{arm} requires seed={int(required_seed)}, got {actual_seed!r}"
        )

    if arm in {'rfreeze', 'rdisc'}:
        from .frozen_vdrm import validate_frozen_config
        validate_frozen_config(cfg)
        return
    if getattr(cfg.TRAIN, 'VDRM_FROZEN_BASE_CONFIG', '') or getattr(cfg.MODEL.VDRM, 'DISCRIMINATIVE_ROUTE', False):
        raise ValueError('Tclean/Ronly must not inherit frozen/discriminative configuration')

    vdrm_cfg = getattr(cfg.MODEL, "VDRM", None)
    if vdrm_cfg is None or not bool(vdrm_cfg.ENABLED):
        raise ValueError(f"{arm} requires MODEL.VDRM.ENABLED=True")

    nonzero_auxiliary = {
        name: float(getattr(cfg.TRAIN, name))
        for name in _VDRM_AUXILIARY_WEIGHTS
        if float(getattr(cfg.TRAIN, name)) != 0.0
    }
    if nonzero_auxiliary:
        raise ValueError(
            f"{arm} requires every VDRM auxiliary weight to be zero, got "
            f"{nonzero_auxiliary}"
        )

    train_alpha = bool(getattr(vdrm_cfg, "TRAIN_ALPHA", True))
    expected_train_alpha = arm == "ronly"
    if train_alpha != expected_train_alpha:
        raise ValueError(
            f"{arm} requires MODEL.VDRM.TRAIN_ALPHA="
            f"{expected_train_alpha}, got {train_alpha}"
        )


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET", "TRACKINGNET_TRAIN012"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "TRACKINGNET_TRAIN012":
            if settings.use_lmdb:
                raise ValueError("TRACKINGNET_TRAIN012 is only supported for folder datasets")
            datasets.append(TrackingNet(settings.env.trackingnet_dir, set_ids=[0, 1, 2],
                                        image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings,
                                                       same_class_distractor_min_scale=getattr(
                                                           cfg.DATA.SEARCH,
                                                           "VDRM_DISTRACTOR_MIN_SCALE",
                                                           0.7,
                                                       ),
                                                       same_class_distractor_max_scale=getattr(
                                                           cfg.DATA.SEARCH,
                                                           "VDRM_DISTRACTOR_MAX_SCALE",
                                                           1.3,
                                                       ),
                                                       same_class_distractor_placement=getattr(
                                                           cfg.DATA.SEARCH,
                                                           "VDRM_DISTRACTOR_PLACEMENT",
                                                           "random",
                                                       ))

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls,
                                            same_class_distractor_probability=getattr(
                                                cfg.DATA.SEARCH,
                                                "VDRM_DISTRACTOR_PROBABILITY",
                                                0.0,
                                            ))

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    if getattr(cfg.TRAIN, 'VDRM_FROZEN_SKIP_VAL', False):
        return loader_train, None

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        # VDRM is inserted inside the backbone for access to intermediate
        # tokens, but it is a newly initialized module and must use the main
        # learning rate rather than the reduced pretrained-backbone rate.
        param_dicts = [
            {
                "params": [
                    p for n, p in net.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in net.named_parameters()
                    if "backbone.vdrm" in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in net.named_parameters()
                    if "backbone" in n
                    and "backbone.vdrm" not in n
                    and p.requires_grad
                ],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        param_dicts = [group for group in param_dicts if group["params"]]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
