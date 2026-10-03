"""Audit a frozen candidate and Tclean without training or optimizer updates.

This standalone check uses fixed synthetic inputs to validate wiring after
training. The formal training entry separately audits a real training batch.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.config.ostrack.config import default_config, update_config_from_file
from lib.train.base_functions import validate_vdrm_experiment_contract
from lib.train.frozen_vdrm import prepare_frozen_models, audit_frozen_model, validate_frozen_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--checkpoint', required=True, help='Candidate checkpoint to validate, never the Tclean source')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cfg = default_config()
    update_config_from_file(PROJECT / 'experiments' / 'ostrack' / (args.config + '.yaml'), cfg)
    validate_vdrm_experiment_contract(cfg, actual_seed=cfg.TRAIN.VDRM_REQUIRED_SEED)
    torch.manual_seed(cfg.TRAIN.VDRM_REQUIRED_SEED)
    model, baseline, source = prepare_frozen_models(cfg, args.save_dir, args.config, args.checkpoint)
    validate_frozen_checkpoint(model, torch.load(args.checkpoint, map_location='cpu', weights_only=False),
                               cfg=cfg, expected_epoch=cfg.TEST.EPOCH)
    device = torch.device(args.device)
    model, baseline = model.to(device), baseline.to(device)
    data = dict(
        template_images=torch.randn(1, 2, 3, cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.SIZE, device=device),
        search_images=torch.randn(1, 2, 3, cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.SIZE, device=device),
        template_anno=torch.tensor([[[.25, .25, .5, .5]] * 2], device=device),
        search_anno=torch.tensor([[[.25, .25, .5, .5]] * 2], device=device),
        search_att=torch.zeros(1, 2, cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.SIZE, device=device, dtype=torch.bool),
        epoch=1,
    )
    result = audit_frozen_model(model, baseline, cfg, data)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()), source=str(source), inputs='synthetic wiring check')
    output = Path(args.save_dir) / 'analysis' / 'frozen_vdrm' / args.config / 'checkpoint_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('FROZEN CHECKPOINT CHECK PASSED: exact alpha=0 Tclean outputs; isolated route gradients; no optimizer step')
    print('Report:', output.resolve())


if __name__ == '__main__':
    main()
