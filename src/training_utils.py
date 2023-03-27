"""
Contains helper functions for loading and saving hyperparameters and configs
of main() run scripts.
"""

import os
import datetime
import json
import time
import torch


def save_args(args, run_dir):
    """
    Saves arguments into the run directory in JSON format.
    :return run_cmd_path: path to the saved JSON file
    """
    ts = time.time()
    ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S%f')[:-3]  # including ms
    run_cmd = {}
    run_cmd['parsed_args'] = args.__dict__
    fn = "%s-runcmd.json" % (ts_str, )
    run_cmd_path = os.path.join(run_dir, fn)
    with open(run_cmd_path, 'w') as f:
        json.dump(run_cmd, f, indent=2, sort_keys=True)
    return run_cmd_path

def optimiser_to(optim, device):
    '''
    Moves optimiser to device.
    https://github.com/pytorch/pytorch/issues/2830
    https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    '''
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
