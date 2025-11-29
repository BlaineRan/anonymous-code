"""
TinyML configuration module

Provides unified accessors for these configuration files:
- llm_config.yaml       : LLM service settings
- llm_prompts.yaml      : Prompt templates
- search_space.yaml     : Architecture search space definitions
- training.yaml         : Training hyperparameters
"""

from pathlib import Path
import yaml
from typing import Dict, Any

# Path to the configs directory
_CONFIG_DIR = Path(__file__).parent

# Cache loaded configurations
_config_cache: Dict[str, Any] = {}


def _load_config(file_name: str) -> Dict[str, Any]:
    """Load the specified YAML configuration file."""
    if file_name not in _config_cache:
        config_path = _CONFIG_DIR / file_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            _config_cache[file_name] = yaml.safe_load(f)
    return _config_cache[file_name]


# Create accessors for each configuration file
def get_llm_config() -> Dict[str, Any]:
    """Return the LLM service configuration."""
    return _load_config("llm_config.yaml")


def get_llm_prompts() -> Dict[str, Any]:
    """Return the prompt template configuration."""
    return _load_config("llm_prompts.yaml")


def get_search_space() -> Dict[str, Any]:
    """Return the architecture search space definition."""
    return _load_config("search_space.yaml")


def get_tnas_search_space() -> Dict[str, Any]:
    """Return the TANS architecture search space definition."""
    return _load_config("tnas_search_space.yaml")


def get_noquant_search_space() -> Dict[str, Any]:
    """Return the no-quantization architecture search space definition."""
    return _load_config("noquant_search_space.yaml")


def get_training_config() -> Dict[str, Any]:
    """Return the training parameter configuration."""
    return _load_config("training.yaml")


def get_simple_search_space() -> Dict[str, Any]:
    """Return the simplified architecture search space definition."""
    return _load_config("simple_search.yaml")


# Explicit export list
__all__ = [
    'get_llm_config',
    'get_llm_prompts',
    'get_search_space',
    'get_training_config',
    'get_tnas_search_space',
    'get_noquant_search_space',
    'get_simple_search_space',
]

# Version info
__version__ = '0.1.0'


# Example usage
# from tinyml.configs import (
#     get_llm_config,
#     get_search_space,
# )
#
# llm_config = get_llm_config()
# search_space = get_search_space()
# print("LLM config:", llm_config['model_name'])
# print("Search space constraints:", search_space['constraints'])
