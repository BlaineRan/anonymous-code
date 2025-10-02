import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到路径
# 从项目导入
# 修改后的导入语句
from utils import initialize_llm
from models.candidate_models import CandidateModel
from utils import visualization
import logging

logger = logging.getLogger(__name__)

@dataclass
class ArchitectureExplanation:
    """存储LLM生成的架构解释及其置信度"""
    kernel_choice: str
    expansion_reason: str
    se_block_usage: str 
    skip_connection_strategy: str
    overall_strategy: str
    confidence_score: float = 0.8  # 默认置信度

class ExplainabilityModule:
    """处理模型架构的可解释性，与LLM引导搜索紧密集成"""
    
    def __init__(self, llm_config: Optional[Dict] = None):
        """
        初始化解释模块
        
        参数:
            llm_config: 可选LLM配置， 默认使用项目配置
        """
        self.llm = initialize_llm(llm_config)
        self.explanation_history = []
        
    def explain_architecture(self, 
                           candidate: CandidateModel,
                           feedback: Optional[str] = None) -> ArchitectureExplanation:
        """
        为候选架构生成详细解释
        
        参数:
            candidate: 要解释的候选模型
            feedback: 可选的Pareto前沿反馈信息
            
        返回:
            ArchitectureExplanation: 结构化解释
        """
        prompt = self._build_explanation_prompt(candidate.config, feedback)
        
        try:
            response = self.llm.invoke(prompt).content
            logger.debug(f"Explanation LLM response: {response}")
            
            explanation = self._parse_explanation(response, candidate.config)
            self.explanation_history.append(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return self._get_default_explanation(candidate.config)

    # def generate_explanation(self, config: Dict[str, Any]) -> ArchitectureExplanation:
    #     """使用LLM生成架构设计决策的解释"""
    #     prompt = self._build_explanation_prompt(config)
    #     response = self.llm.generate(prompt)
    #     return self._parse_explanation(response)
    
    def _build_explanation_prompt(self, 
                                config: Dict[str, Any], 
                                feedback: Optional[str]) -> str:
        """
        构建解释生成提示，包含架构配置和上下文信息
        
        参数:
            config: 模型配置字典
            feedback: Pareto前沿反馈信息
            
        返回:
            str: 完整的提示文本
        """
        # 先计算kernel_sizes
        kernel_sizes = list(set(
            block['kernel_size'] 
            for stage in config['stages'] 
            for block in stage['blocks']
        ))
        kernel_sizes_str = ", ".join(map(str, sorted(kernel_sizes)))
        prompt_template = f"""As a neural architecture design expert, analyze this TinyML model configuration:

        **Configuration:**
        {json.dumps(config, indent=2)}

        **Context:**
        {feedback or "No Pareto feedback available"}

        **Analysis Tasks:**
        1. Kernel Sizes: Explain the choice of kernel sizes ({kernel_sizes_str}) across layers
        2. Expansion Factors: Justify the expansion ratios used in MBConv blocks
        3. SE Blocks: Analyze the Squeeze-Excitation block placement strategy
        4. Skip Connections: Explain the residual connection usage pattern
        5. Overall Strategy: Describe how this architecture balances:
        - Accuracy vs Efficiency
        - Memory vs Compute
        - Model Depth vs Width

        **Response Format (JSON):**
        {{
        "kernel_choice": "analysis of kernel size selection",
        "expansion_reason": "rationale for expansion factors", 
        "se_block_usage": "SE block strategy explanation",
        "skip_connection_strategy": "residual connection analysis",
        "overall_strategy": "global design philosophy",
        "confidence": 0.0-1.0
        }}"""
        

        
        return prompt_template
    
    
    def _parse_explanation(self, response: str, config: Dict) -> ArchitectureExplanation:
        """
        解析LLM的响应为结构化解释
        
        参数:
            response: LLM的原始响应
            config: 原始配置(用于验证)
            
        返回:
            ArchitectureExplanation: 解析后的解释
        """
        try:
            data = json.loads(response.strip())
            
            # 验证关键字段存在
            required_fields = {
                'kernel_choice', 'expansion_reason', 
                'se_block_usage', 'overall_strategy',
                'skip_connection_strategy', 'confidence'
            }
            if not all(field in data for field in required_fields):
                raise ValueError("Missing required explanation fields")
                
            return ArchitectureExplanation(
                kernel_choice=data['kernel_choice'],
                expansion_reason=data['expansion_reason'],
                se_block_usage=data['se_block_usage'],
                skip_connection_strategy=data['skip_connection_strategy'],
                overall_strategy=data['overall_strategy'],
                confidence_score=min(max(float(data['confidence']), 0), 1)  # 限制在0-1范围
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse explanation: {str(e)}")
            return self._get_default_explanation(config)
    
    
    def _get_default_explanation(self, config: Dict) -> ArchitectureExplanation:
        """生成默认解释(当LLM解析失败时使用)"""
        return ArchitectureExplanation(
            kernel_choice="[failed] Balancing receptive field size and computational cost",
            expansion_reason="[failed] Optimizing model capacity within resource constraints",
            se_block_usage="[failed] Selective channel attention for critical feature maps",
            skip_connection_strategy="[failed] Facilitating gradient flow in deep layers",
            overall_strategy=(
                "[failed] Accuracy-efficiency trade-off for TinyML deployment "
                f"(Target SRAM: {config.get('constraints', {}).get('max_sram', 320)/1024:.0f}KB)"
            ),
            confidence_score=0.5  # 低置信度
        )
    
    def visualize(self, 
                 explanation: ArchitectureExplanation,
                 save_path: Optional[str] = None):
        """
        可视化解释结果
        
        参数:
            explanation: 要可视化的解释
            save_path: 可选的文件保存路径
        """
        # 使用项目中的可视化工具
        visualization.plot_architecture_explanation(
            explanation=explanation,
            save_path=save_path
        )
        
    def save_explanation_history(self, file_path: str):
        """保存解释历史到JSON文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump([e.__dict__ for e in self.explanation_history], f, indent=2)
        
        logger.info(f"Saved {len(self.explanation_history)} explanations to {path}")

# 示例用法
if __name__ == "__main__":
    # 测试解释模块
    from configs import get_search_space
    

    search_space = get_search_space()
    
    # 创建候选配置 - 现在使用CandidateModel包装
    test_config = CandidateModel(
        config={
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "MBConv",
                            "kernel_size": 3,
                            "expansion": 4,
                            "has_se": True,
                            "skip_connection": False
                        }
                    ],
                    "channels": 32
                }
            ],
            "constraints": {
                "max_sram": 320 * 1024,
                "max_macs": 350 * 1e6
            }
        }
    )
    
    # 初始化解释模块
    explainer = ExplainabilityModule()
    
    # 生成解释 - 现在传递的是CandidateModel实例
    explanation = explainer.explain_architecture(
        candidate=test_config,  # 传递对象而不是字典
        feedback="Current Pareto front shows average accuracy of 68% with 250M MACs"
    )
    
    # 打印结果
    print("\n=== Architecture Explanation ===")
    print(f"Kernel Strategy: {explanation.kernel_choice}")
    print(f"Expansion Logic: {explanation.expansion_reason}")
    print(f"SE Block Usage: {explanation.se_block_usage}")
    print(f"Skip Connections: {explanation.skip_connection_strategy}")
    print(f"Overall Approach: {explanation.overall_strategy}")
    print(f"Confidence: {explanation.confidence_score:.0%}")