import torch
from collections import OrderedDict
import torch.nn as nn
from typing import Tuple

def calculate_memory_usage(model: nn.Module, input_size: Tuple[int], device: torch.device = torch.device('cpu')) -> dict:
    """
    Robustly calculate model activation and parameter memory by directly querying tensor properties.
    This version accurately distinguishes between different quantization modes.
    """
    model = model.to(device)
    model.eval()
    activation_memory = 0
    parameter_memory = 0
    dummy_input = torch.randn(*input_size, device=device)

    hooks = []
    def forward_hook(module, input, output):
        nonlocal activation_memory
        # Directly query the element size of the output tensor, which is the most accurate method
        output_tensor = output[0] if isinstance(output, (tuple, list)) else output
        activation_memory += output_tensor.numel() * output_tensor.element_size()

    # Only hook on leaf modules to avoid double counting
    for layer in model.modules():
        if not list(layer.children()): 
            hooks.append(layer.register_forward_hook(forward_hook))

    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    # Also use element_size() to calculate parameter memory
    for param in model.parameters():
        parameter_memory += param.numel() * param.element_size()

    activation_memory_MB = activation_memory / (1024 ** 2)
    parameter_memory_MB = parameter_memory / (1024 ** 2)

    return {
        "activation_memory_MB": activation_memory_MB,
        "parameter_memory_MB": parameter_memory_MB,
        "total_memory_MB": activation_memory_MB + parameter_memory_MB,
    }

# def calculate_memory_usage(model: nn.Module, input_size: Tuple[int], device: torch.device = torch.device('cpu')) -> dict:
#     """
#     Robustly calculate model activation and parameter memory by directly querying tensor properties.
#     """
#     model = model.to(device)
#     model.eval()
#     activation_memory = 0
#     parameter_memory = 0
#     dummy_input = torch.randn(*input_size, device=device)

#     hooks = []
#     def forward_hook(module, input, output):
#         nonlocal activation_memory
#         # Count memory of input tensors
#         for inp in input:
#             if isinstance(inp, torch.Tensor):
#                 activation_memory += inp.numel() * inp.element_size()
#         # Count memory of output tensors
#         if isinstance(output, torch.Tensor):
#             activation_memory += output.numel() * output.element_size()
#         elif isinstance(output, (tuple, list)):
#             for out in output:
#                 if isinstance(out, torch.Tensor):
#                     activation_memory += out.numel() * out.element_size()

#     # Only hook on leaf modules to avoid double counting
#     for layer in model.modules():
#         if not list(layer.children()): 
#             hooks.append(layer.register_forward_hook(forward_hook))

#     with torch.no_grad():
#         model(dummy_input)

#     for hook in hooks:
#         hook.remove()

#     # Also use element_size() to calculate parameter memory
#     for param in model.parameters():
#         parameter_memory += param.numel() * param.element_size()

#     activation_memory_MB = activation_memory / (1024 ** 2)
#     parameter_memory_MB = parameter_memory / (1024 ** 2)

#     return {
#         "activation_memory_MB": activation_memory_MB,
#         "parameter_memory_MB": parameter_memory_MB,
#         "total_memory_MB": activation_memory_MB + parameter_memory_MB,
#     }
