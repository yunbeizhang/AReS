import numpy as np
import torch
import random
import os


#Edit the Path
data_path = os.getenv('DATA_DIR', './data/')
results_path = './results'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

config_vm = {'lr': 0.01, 'epoch': 200,
             'blmp': {'lap': 1, 'topk_ratio': 0.15},
             'blm': {'lap': 1},
             'ft_lr': 0.1
             }
config_vlm = {'lr': 40, 'epoch': 200,
              'blmp': {'lap': 1, 'topk_ratio': 0.15},
              'blm': {'lap': 1}
              }
