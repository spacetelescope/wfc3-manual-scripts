"""
This module enables the information in the configuration
file (`config.yaml`) to be easily handled by scripts,
without having to re-load the data multiple times.
"""

import os
import yaml

__location__ = os.path.realpath(
                os.path.join(os.getcwd(), os.path.dirname(__file__)))

def get_config():
    """Gets the setting information from a configuration file in the $HOME
    directory.
    """
    with open(os.path.join(__location__, 'config.yaml'), 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

CONFIG = get_config()
