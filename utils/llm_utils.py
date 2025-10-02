# tinyml/utils/llm_utils.py
import yaml
from pathlib import Path
from typing import Optional
from langchain_openai import ChatOpenAI
# from langchain.memory import ChatMessageHistory
# from langchain.schema import HumanMessage, AIMessage

class LLMInitializer:
    """
    集中管理 LLM 初始化的工具类，确保所有模块使用统一的LLM配置
    """
    _instance = None
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # 从YAML文件加载配置（如果未提供参数）
        if not kwargs and config_path:
            config = self._load_config(config_path)
            kwargs = config['llm']  # 提取llm配置部分
        
        self.llm = ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.7),
            base_url=kwargs["base_url"],
            api_key=kwargs["api_key"]
        )
    
    @classmethod
    def _load_config(cls, config_path: str) -> dict:
        """加载YAML配置文件"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return yaml.safe_load(path.read_text())
    
    @classmethod
    def initialize(cls, config_path: str = None, **kwargs):
        """初始化LLM单例"""
        if cls._instance is None:
            cls._instance = cls(config_path, **kwargs)
        return cls._instance
    
    @classmethod
    def get_llm(cls):
        """获取已初始化的LLM实例"""
        if cls._instance is None:
            default_config = str(Path(__file__).parent.parent / "configs" / "llm_config.yaml")
            cls.initialize(config_path=default_config)
        return cls._instance.llm


def initialize_llm(llm_config: dict = None):
    """
    初始化LLM的工厂函数
    :param llm_config: 可选，如果不提供则从默认配置文件读取
    """
    if llm_config is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "llm_config.yaml")
        return LLMInitializer.initialize(config_path=config_path).get_llm()
    return LLMInitializer.initialize(**llm_config).get_llm()


# if __name__ == "__main__":
#     # 示例用法（自动从configs/llm_config.yaml加载配置）
#     llm = LLMInitializer.get_llm()
    
#     # 对话演示
#     history = ChatMessageHistory()
#     history.add_user_message("请推荐一个轻量级CNN架构")
#     response = llm.invoke(history.messages)
#     print("AI回复:", response.content)