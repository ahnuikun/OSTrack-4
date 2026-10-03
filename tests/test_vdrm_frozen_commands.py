"""Parse the exact copy/paste server commands shipped in the runbook."""

import re
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from tracking import train, test_uav_suite
from tracking import analyze_vdrm_module1_dev as analyze
from tracking import check_vdrm_frozen as check
from tracking import evaluate_vdrm_frozen_routes as offline

PROJECT = Path(__file__).resolve().parents[1]


class ParsedCommand(Exception):
    """Stop immediately after parsing, before any model/data access."""


class ServerCommandsTest(unittest.TestCase):
    def test_all_documented_commands_parse_and_use_registered_configs(self):
        document = (PROJECT / 'VDRM_FROZEN_ROUTE_PLAN.md').read_text(encoding='utf-8')
        blocks = re.findall(r'```\w+\n(.*?)```', document, re.S)
        counts = {}
        for block in blocks:
            for line in block.replace('\\\n', '').splitlines():
                if 'python tracking/' not in line:
                    continue
                tokens = shlex.split(line)
                python = tokens.index('python')
                script, args = tokens[python + 1], tokens[python + 2:]
                self.assertNotIn('--seed', args)
                counts[script] = counts.get(script, 0) + 1
                # Every referenced config must be a real file in this branch.
                for option in ('--config', '--tracker_param', '--reference'):
                    if option in args:
                        name = args[args.index(option) + 1]
                        self.assertTrue((PROJECT / 'experiments/ostrack' / (name + '.yaml')).is_file(), name)
                if '--tracker_params' in args:
                    start = args.index('--tracker_params') + 1
                    for name in args[start:]:
                        if name.startswith('--'):
                            break
                        self.assertTrue((PROJECT / 'experiments/ostrack' / (name + '.yaml')).is_file(), name)
                with self.subTest(script=script, args=args), patch.object(sys, 'argv', [script, *args]):
                    if script == 'tracking/train.py':
                        self.assertEqual(tokens[0], 'CUDA_VISIBLE_DEVICES=0,1,2,3')
                        with patch.object(train.os, 'system', return_value=0) as launch:
                            train.main()
                        self.assertIn('--seed 42', launch.call_args.args[0])
                        self.assertIn('--nproc_per_node 4', launch.call_args.args[0])
                    elif script == 'tracking/test_uav_suite.py':
                        parsed = test_uav_suite.parse_args()
                        self.assertEqual(tokens[0], 'CUDA_VISIBLE_DEVICES=0,1,2,3')
                        self.assertEqual(parsed.dataset, 'got10k_vdrm_dev')
                        self.assertEqual((parsed.threads, parsed.num_gpus), (4, 4))
                    elif script == 'tracking/check_vdrm_frozen.py':
                        name = args[args.index('--config') + 1]
                        checkpoint = args[args.index('--checkpoint') + 1]
                        self.assertEqual(checkpoint, f'output/checkpoints/train/ostrack/{name}/OSTrack_ep0300.pth.tar')
                        with patch.object(check, 'default_config', side_effect=ParsedCommand):
                            with self.assertRaises(ParsedCommand):
                                check.main()
                    elif script == 'tracking/evaluate_vdrm_frozen_routes.py':
                        with patch.object(offline, 'default_config', side_effect=ParsedCommand):
                            with self.assertRaises(ParsedCommand):
                                offline.main()
                    elif script == 'tracking/analyze_vdrm_module1_dev.py':
                        self.assertIn('--visibility_stages', args)
                        self.assertNotEqual(args[args.index('--report_name') + 1], 'vdrm_module1_dev')
                        with patch.object(analyze, 'get_dataset', side_effect=ParsedCommand):
                            with self.assertRaises(ParsedCommand):
                                analyze.main()
                    else:
                        self.fail('Untested command: ' + script)
        self.assertEqual(counts, {'tracking/train.py': 2, 'tracking/test_uav_suite.py': 3,
                                 'tracking/check_vdrm_frozen.py': 2,
                                 'tracking/analyze_vdrm_module1_dev.py': 3,
                                 'tracking/evaluate_vdrm_frozen_routes.py': 1})


if __name__ == '__main__':
    unittest.main()
