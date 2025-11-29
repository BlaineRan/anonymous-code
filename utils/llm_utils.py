# tinyml/utils/llm_utils.py
import yaml
from pathlib import Path
from typing import Optional
from langchain_openai import ChatOpenAI
# from langchain.memory import ChatMessageHistory
# from langchain.schema import HumanMessage, AIMessage

class LLMInitializer:
    """
    Centralized helper for initializing the LLM and ensuring all modules share the same configuration.
    """
    _instance = None
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # Load configuration from a YAML file (if no parameters are provided)
        if not kwargs and config_path:
            config = self._load_config(config_path)
            kwargs = config['llm']  # Get the llm configuration section
        
        self.llm = ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.7),
            base_url=kwargs["base_url"],
            api_key=kwargs["api_key"]
        )
    
    @classmethod
    def _load_config(cls, config_path: str) -> dict:
        """Load the YAML configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return yaml.safe_load(path.read_text())
    
    @classmethod
    def initialize(cls, config_path: str = None, **kwargs):
        """Initialize the LLM singleton."""
        if cls._instance is None:
            cls._instance = cls(config_path, **kwargs)
        return cls._instance
    
    @classmethod
    def get_llm(cls):
        """Get the initialized LLM instance."""
        if cls._instance is None:
            default_config = str(Path(__file__).parent.parent / "configs" / "llm_config.yaml")
            cls.initialize(config_path=default_config)
        return cls._instance.llm


def initialize_llm(llm_config: dict = None):
    """
    Factory function for initializing the LLM.
    :param llm_config: Optional parameter; if not provided, read from the default config file.
    """
    if llm_config is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "llm_config.yaml")
        return LLMInitializer.initialize(config_path=config_path).get_llm()
    return LLMInitializer.initialize(**llm_config).get_llm()


# if __name__ == "__main__":
#     # Example usage (automatically loads config from configs/llm_config.yaml)
#     llm = LLMInitializer.get_llm()
    
#     # Conversation demo
#     history = ChatMessageHistory()
#     history.add_user_message("Please recommend a lightweight CNN architecture")
#     response = llm.invoke(history.messages)
#     print("AI response:", response.content)
