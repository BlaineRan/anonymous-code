import json
from typing import Dict, Any
from models.candidate_models import CandidateModel

class MemoryEstimator:
    @staticmethod
    def calc_layer_mem(layer_config: Dict[str, Any], H: int, W: int,
                       C_in: int, C_out: int) -> float:
        """Calculate the layer memory usage compatible with the stages/blocks layout."""
        # Base activation memory (bytes)
        stride = layer_config.get('stride', 1)
        has_se = layer_config.get('has_se', False)

        if stride == 1:
            act_mem = 4 * H * W * C_in  # float32 elements
        else:  # stride=2 halves the spatial dimensions
            act_mem = 4 * (H // 2) * (W // 2) * C_out

        # Additional overhead for the SE module
        if has_se:
            se_mem = 4 * (C_in + C_in // 2)  # two 1x1 convolutions
            act_mem += se_mem

        return act_mem

    @staticmethod
    def calc_model_sram(model: CandidateModel) -> float:
        """Estimate the total SRAM usage for the stages/blocks layout."""
        # Parameter memory loaded from Flash into SRAM
        param_mem = 4 * model.estimate_params()  # assume float32 representation

        # Estimate activation memory peak
        max_act_mem = 0
        H, W = 224, 224  # default input spatial size; adjust if needed
        C_in = 3  # initial input channel count

        for stage in model.config['stages']:
            C_out = stage['channels']
            for block in stage['blocks']:
                current_mem = MemoryEstimator.calc_layer_mem(
                    block, H, W, C_in, C_out
                )
                max_act_mem = max(max_act_mem, current_mem)

                # Update spatial dimensions and channel count
                if block.get('stride', 1) == 2:
                    H, W = H // 2, W // 2
                C_in = C_out

        return param_mem + max_act_mem + 20e3  # include 20KB system overhead

class ConstraintValidator:
    """Ensure the generated architecture meets the defined hardware constraints."""

    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints

    def validate(self, model: CandidateModel) -> bool:
        """Check whether the model satisfies all constraint requirements."""
        return all([
            self._validate_macs(model),
            self._validate_sram(model),
            self._validate_params(model)
        ])

    def _validate_macs(self, model: CandidateModel) -> bool:
        macs = model.estimate_macs()
        return float(self.constraints['min_macs']) / 1e6 <= float(macs) <= float(self.constraints['max_macs']) / 1e6

    def _validate_sram(self, model: CandidateModel) -> bool:
        sram = MemoryEstimator.calc_model_sram(model)
        if float(sram) > float(self.constraints['max_sram']):
            print(f"SRAM budget exceeded: {float(sram)/1e3:.1f}KB > {float(self.constraints['max_sram'])/1e3}KB")
            return False
        return True

    def _validate_params(self, model: CandidateModel) -> bool:
        params = model.estimate_params()
        return float(params) <= float(self.constraints['max_params'])

def validate_constraints(model: CandidateModel, constraints: Dict[str, Any]) -> bool:
    """Quick helper to validate generated models against constraints."""
    validator = ConstraintValidator(constraints)
    return validator.validate(model)
