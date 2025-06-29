import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from  config.load_config import load_config

config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"DAU.yaml"))

def get_defense_config(defense_strategy = None):
    defense_config = get_DAU_config()
    return defense_config

def get_DAU_config():
    defense_config = {
        'defense_strategy':"DAU",
        'init_filter':{},
        'anti_learning':{},
        'filter':{},
        'unlearning':{},
        'repair':{},
        'ssl':{},
        'latents':{},
        'work_dir': None,
        'schedule': None
    }

    defense_config['defense_strategy'] = config['DAU']['defense_strategy']
    # init_filter
    defense_config['init_filter'] = config['DAU']['init_filter']
    # anti_learning
    defense_config['anti_learning'] = config['DAU']['anti_learning']
    # filter
    defense_config["filter"] = config['DAU']['filter']
    # unlearning
    defense_config['unlearning'] = config['DAU']['unlearning']
    #repair
    defense_config['repair'] = config['DAU']['repair']
    # semi-supervised learning
    defense_config['ssl'] = config['DAU']['ssl']  
    # latents
    defense_config['latents'] = config['DAU']['latents']
    return defense_config


if __name__ == "__main__":
    print(f"inner_dir:{inner_dir}, config_name:{config_name}\n")
    defense_strategy = "DAU"
    defense_config = get_defense_config(defense_strategy)
    print(config)
    print(defense_config)
