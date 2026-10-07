"""Read-only Rresp checkpoint, bypass and gradient-isolation audit."""

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.config.ostrack.config import default_config, update_config_from_file
from lib.train.base_functions import validate_vdrm_experiment_contract
from lib.train.frozen_response import prepare_response_models, audit_response_model, validate_response_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cfg = default_config()
    update_config_from_file(PROJECT / 'experiments/ostrack' / (args.config + '.yaml'), cfg)
    validate_vdrm_experiment_contract(cfg, actual_seed=42)
    torch.manual_seed(42)
    model, baseline, source = prepare_response_models(cfg, args.save_dir, args.config, args.checkpoint)
    validate_response_checkpoint(model, torch.load(args.checkpoint, map_location='cpu', weights_only=False),
                                 cfg=cfg, expected_epoch=cfg.TEST.EPOCH)
    device = torch.device(args.device)
    model, baseline = model.to(device), baseline.to(device)
    data = dict(template_images=torch.randn(1, 2, 3, 128, 128, device=device),
        search_images=torch.randn(1, 2, 3, 256, 256, device=device),
        template_anno=torch.tensor([[[.25, .25, .5, .5]] * 2], device=device),
        search_anno=torch.tensor([[[.25, .25, .5, .5]] * 2], device=device),
        search_att=torch.zeros(1, 2, 256, 256, device=device, dtype=torch.bool), epoch=1)
    result = audit_response_model(model, baseline, cfg, data)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()), source=str(source),
                  inputs='synthetic wiring check; not tracking-performance evidence')
    output = Path(args.save_dir) / 'analysis/frozen_response' / args.config / 'checkpoint_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('RRESP CHECKPOINT CHECK PASSED: exact Tclean bypass; frozen regression; isolated response gradients; no optimizer step')
    print('Report:', output.resolve())


if __name__ == '__main__':
    main()
