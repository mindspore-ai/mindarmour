import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np
import yaml
import mindspore
import mindspore.nn as nn
import mindspore.dataset.vision as vision
from mindspore.dataset import MnistDataset, Cifar10Dataset, Cifar100Dataset
from mindspore.dataset.transforms import Compose
from config.load_config import load_config


support_attack_strategies = ["BadNets", "Blended", "WaNet", "Refool", "SIG"]

def get_BadNets_config(W=28,H=28):
    config, _, _ = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/BadNets.yaml"))
    attack_config = {
        'attack_strategy':'BadNets',
        'y_target':None,
        'poisoned_rate':None,
        'pattern': None,
        'weight':None,
        'train_schedule':None,
        'work_dir':None
    }  
    attack_config['y_target'] = config['BadNets']['y_target']
    attack_config['poisoned_rate'] = config['BadNets']['poisoned_rate']

    pattern = mindspore.ops.Zeros()((W, H), mindspore.uint8)   
    pattern[-3:, -3:] = 0
    weight = mindspore.ops.Zeros()((W, H), mindspore.float32)
    weight[-3:, -3:] = 1.0

    attack_config['pattern'] = pattern
    attack_config['weight'] = weight

    return attack_config

def get_Blended_config(W=32,H=32):
    config, _, _ = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/Blended.yaml"))
    attack_config = {
        'attack_strategy':'Blended',
        'y_target':None,
        'poisoned_rate':None,
        'pattern': None,
        'W':None,
        'H':None,
        'pieces':None,
        'mask_rate':None,
        'alpha':None,
        'train_schedule':None,
        'work_dir':None
    }
    
    attack_config['y_target'] = config['Blended']['y_target']
    attack_config['poisoned_rate'] = config['Blended']['poisoned_rate']
    attack_config['pattern'] = config['Blended']['pattern']
    attack_config['W'] = W
    attack_config['H'] = H   
    attack_config['pieces'] = config['Blended']['pieces']
    attack_config['mask_rate'] = config['Blended']['mask_rate']
    attack_config['alpha'] = config['Blended']['alpha']

    return attack_config

def get_IAD_config(W=32,H=32):
    config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/IAD.yaml"))
    attack_config = {
        'attack_strategy':'IAD',
        "dataset_name":None,
        "train_dataset1": None,
        "test_dataset1": None,
        "y_target": 0,
        "poisoned_rate": 0.1,   # follow the default configure in the original paper
        "cross_rate": 0.1,      # follow the default configure in the original paper
        "lambda_div": 1,
        "lambda_norm": 100,
        "mask_density": 0.032,
        "EPSILON": 0.0000001,
        
        "model": None,
        "modelG": None,
        "modelM": None,

        'train_schedule':None,
        'work_dir':None
    }

    attack_config['dataset_name'] = config['IAD']['dataset_name']
    attack_config['y_target'] = config['IAD']['y_target']
    attack_config['poisoned_rate'] = config['IAD']['poisoned_rate']
    attack_config['cross_rate'] = config['IAD']['cross_rate']
    attack_config['lambda_div'] = config['IAD']['lambda_div']
    attack_config['lambda_norm'] = config['IAD']['lambda_norm']
    attack_config['mask_density'] = config['IAD']['mask_density']
    attack_config['EPSILON'] = config['IAD']['EPSILON']

    return attack_config

def get_WaNet_config(W=28,H=28):
    config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/WaNet.yaml"))
    def gen_grid(height, k):
        """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
        according to the input height ``height`` and the uniform grid size ``k``.
        """
        ins = mindspore.ops.UniformReal()((1, 2, k, k)) * 2 - 1
        ins = ins / mindspore.ops.ReduceMean()(mindspore.ops.Abs()(ins))  # a uniform grid
        noise_grid = mindspore.ops.ResizeBilinear(size=height)(ins)
        
        noise_grid = mindspore.ops.Transpose()(noise_grid, (0, 2, 3, 1))  # 1*height*height*2
        array1d = mindspore.ops.LinSpace()(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
        x, y = mindspore.ops.Meshgrid()(array1d, array1d)  # 2D coordinates height*height
        identity_grid = mindspore.ops.Stack(axis=2)([y, x])[None, ...]  # 1*height*height*2

        return identity_grid, noise_grid

    attack_config = {
        'attack_strategy':'WaNet',
        "y_target": 1,
        "poisoned_rate": 0.1,   
        "identity_grid":None,
        "height": 32,
        "k": 10,
        "s": 1.0,
        "noise_grid":None,
        "noise": True,
        'train_schedule':None,
        'work_dir':None
    }
    attack_config['y_target'] = config['WaNet']['y_target']
    attack_config['poisoned_rate'] = config['WaNet']['poisoned_rate']
    attack_config['identity_grid'] = config['WaNet']['identity_grid']
    attack_config['height'] = config['WaNet']['height']
    attack_config['k'] = config['WaNet']['k']
    attack_config['s'] = config['WaNet']['s']
    attack_config['noise_grid'] = config['WaNet']['noise_grid']
    attack_config['noise'] = config['WaNet']['noise']

    identity_grid, noise_grid = gen_grid(attack_config['height'], attack_config['k'])
    attack_config["identity_grid"], attack_config["noise_grid"] = identity_grid, noise_grid

    return attack_config


def get_Refool_config(W=28,H=28):
    config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/Refool.yaml"))
    attack_config = {
        "attack_strategy":'Refool',
        "y_target": 1,
        "poisoned_rate": 0.1,   
        "max_image_size": 560,
        "ghost_rate": 0.49,
        "alpha_b": -1.,     
        "offset": (0, 0),
        "sigma": -1,
        "ghost_alpha": -1.,
        "reflection_candidates":None,
        'train_schedule':None,
        'work_dir':None
    }
    attack_config = config['Refool']
    attack_config["attack_strategy"] = 'Refool'
    return attack_config

def get_SIG_config(W=28,H=28):
    config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config/SIG.yaml"))
    attack_config = {
        "attack_strategy":"SIG",
        "y_target": 0,
        "poisoned_rate": 0.1,
        "delta": 20,
        "frequency": 6,
        "train_schedule": None,
        "work_dir": None
    }
    attack_config = config["SIG"]
    attack_config["attack_strategy"] = "SIG"
    return attack_config
   

def get_attack_config(attack_strategy = None, dataset = None):
    assert attack_strategy in support_attack_strategies, f"{attack_strategy} is not in support_datasets:{support_attack_strategies}"
    if dataset == "MNIST":
        W,H = 28,28
    elif dataset == "CIFAR-10":
        W,H = 32,32
    
    elif dataset == "GTSRB":
        W,H = 32,32

    if attack_strategy == "BadNets":
        attack_config = get_BadNets_config(W=W,H=H)
    elif attack_strategy == "Blended":
        attack_config = get_Blended_config(W=W,H=H)
    elif attack_strategy == "WaNet":
        attack_config = get_WaNet_config()
    elif attack_strategy == "IAD":
        attack_config = get_IAD_config()
    elif attack_strategy == "Refool":
        attack_config = get_Refool_config()
    elif attack_strategy == "SIG":
        attack_config = get_SIG_config()
    
   
    return attack_config

if __name__ == "__main__":  
    print(f"config:{config}, inner_dir:{inner_dir}, config_name:{config_name}")
    attack_strategy = "BadNets"
    attack_config = get_attack_config(attack_strategy, dataset="MNIST")
    print(config)
    print(attack_config)