import json
from typing import Dict, Any
from models.candidate_models import CandidateModel

class MemoryEstimator:
    @staticmethod
    def calc_layer_mem(layer_config: Dict[str, Any], H: int, W: int, 
                       C_in: int, C_out: int) -> float:
        """计算单层内存占用，适配stages/blocks结构"""
        # 基础激活内存（字节）
        stride = layer_config.get('stride', 1)
        has_se = layer_config.get('has_se', False)
        
        if stride == 1:
            act_mem = 4 * H * W * C_in  # float32占4字节
        else:  # stride=2时空间减半
            act_mem = 4 * (H//2) * (W//2) * C_out
        
        # SE模块额外开销
        if has_se:
            se_mem = 4 * (C_in + C_in//2)  # 两个1x1卷积
            act_mem += se_mem
        
        return act_mem

    @staticmethod
    def calc_model_sram(model: CandidateModel) -> float:
        """总SRAM占用计算，适配stages/blocks结构"""
        # 参数内存（从Flash加载到SRAM）
        param_mem = 4 * model.estimate_params()  # 假设float32量化
        
        # 计算激活内存峰值
        max_act_mem = 0
        H, W = 224, 224  # 假设输入尺寸，根据实际情况调整
        C_in = 3  # 初始输入通道数
        
        for stage in model.config['stages']:
            C_out = stage['channels']
            for block in stage['blocks']:
                current_mem = MemoryEstimator.calc_layer_mem(
                    block, H, W, C_in, C_out
                )
                max_act_mem = max(max_act_mem, current_mem)
                
                # 更新尺寸和通道数
                if block.get('stride', 1) == 2:
                    H, W = H // 2, W // 2
                C_in = C_out
        
        return param_mem + max_act_mem + 20e3  # 20KB系统开销

class ConstraintValidator:
    """验证生成的模型架构是否满足硬件约束"""
    
    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints
        
    def validate(self, model: CandidateModel) -> bool:
        """验证模型是否满足所有约束条件"""
        return all([
            self._validate_macs(model),
            self._validate_sram(model),
            self._validate_params(model)
        ])
    
    def _validate_macs(self, model: CandidateModel) -> bool:
        macs = model.estimate_macs()
        return float(self.constraints['min_macs'])/1e6 <= float(macs) <= float(self.constraints['max_macs'])/1e6
    
    def _validate_sram(self, model: CandidateModel) -> bool:
        sram = MemoryEstimator.calc_model_sram(model)
        if float(sram) > float(self.constraints['max_sram']):
            print(f"SRAM超标: {float(sram)/1e3:.1f}KB > {float(self.constraints['max_sram'])/1e3}KB")
            return False
        return True
    
    def _validate_params(self, model: CandidateModel) -> bool:
        params = model.estimate_params()
        return float(params) <= float(self.constraints['max_params'])

def validate_constraints(model: CandidateModel, constraints: Dict[str, Any]) -> bool:
    """快速验证函数"""
    validator = ConstraintValidator(constraints)
    return validator.validate(model)
   