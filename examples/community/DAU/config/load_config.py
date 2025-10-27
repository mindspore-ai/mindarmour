import yaml
import os
def load_config(config_path):
    """Load config file from ``config_path``.
    Args:
        config_path (str): Configuration file path, which must be in ``config`` dir, e.g.,
            ``./config/inner_dir/example.yaml`` and ``config/inner_dir/example``.
    
    Returns:
        config (dict): Configuration dict.
        inner_dir (str): Directory between ``config/`` and configuration file. If ``config_path``
            doesn't contain ``inner_dir``, return empty string.
        config_name (str): Configuration filename.
    """
    assert os.path.exists(config_path)
    config_hierarchy = config_path.split("/")
    if len(config_hierarchy) > 2:
        inner_dir = os.path.join(*config_hierarchy[1:-1])
    else:
        inner_dir = ""
    if len(config_hierarchy) > 3:
        inner_dir = os.path.join(*config_hierarchy[2:-1])
    else:
        inner_dir = ""
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config_name = config_hierarchy[-1].split(".yaml")[0]
    return config, inner_dir, config_name