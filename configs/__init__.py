"""
TinyML 配置文件模块

提供以下配置文件的统一访问接口:
- llm_config.yaml       : LLM服务配置
- llm_prompts.yaml      : 提示词模板
- search_space.yaml     : 架构搜索空间定义
- training.yaml         : 训练参数配置
"""

from pathlib import Path
import yaml
from typing import Dict, Any

# 获取configs目录路径
_CONFIG_DIR = Path(__file__).parent

# 缓存加载的配置
_config_cache: Dict[str, Any] = {}

def _load_config(file_name: str) -> Dict[str, Any]:
    """加载指定的YAML配置文件"""
    if file_name not in _config_cache:
        config_path = _CONFIG_DIR / file_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            _config_cache[file_name] = yaml.safe_load(f)
    return _config_cache[file_name]

# 为每个配置文件创建访问函数
def get_llm_config() -> Dict[str, Any]:
    """获取LLM服务配置"""
    return _load_config("llm_config.yaml")

def get_llm_prompts() -> Dict[str, Any]:
    """获取提示词模板配置"""
    return _load_config("llm_prompts.yaml")

def get_search_space() -> Dict[str, Any]:
    """获取架构搜索空间定义"""
    return _load_config("search_space.yaml")

def get_tnas_search_space() -> Dict[str, Any]:
    """获取TANS架构搜索空间定义"""
    return _load_config("tnas_search_space.yaml")

def get_noquant_search_space() -> Dict[str, Any]:
    """获取TANS架构搜索空间定义"""
    return _load_config("noquant_search_space.yaml")

def get_training_config() -> Dict[str, Any]:
    """获取训练参数配置"""
    return _load_config("training.yaml")

def get_simple_search_space() -> Dict[str, Any]:
    """获取简化的架构搜索空间定义"""
    return _load_config("simple_search.yaml")

# 显式导出列表
__all__ = [
    'get_llm_config',
    'get_llm_prompts',
    'get_search_space',
    'get_training_config',
    'get_tnas_search_space',
    'get_noquant_search_space',
    'get_simple_search_space'
]

# 版本信息
__version__ = '0.1.0'


# 示例用法
# from tinyml.configs import (
#     get_llm_config,
#     get_search_space
# )

# # 获取配置
# llm_config = get_llm_config()
# search_space = get_search_space()

# # 在代码中使用
# print("LLM配置:", llm_config['model_name'])
# print("搜索空间约束:", search_space['constraints'])