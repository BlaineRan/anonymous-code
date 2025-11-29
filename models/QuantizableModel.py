import torch
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
from .conv_blocks import MBConvBlock, DWSepConvBlock, SeDpConvBlock, DpConvBlock, SeSepConvBlock
from data import create_calibration_loader

def get_static_quantization_config(precision='int8'):
    """Retrieve static quantization configs for different precisions (FBGEMM friendly)"""
    
    configs = {
        # ===== FBGEMM-compatible INT8 configs =====
        'int8': {
            'qconfig': quantization.get_default_qconfig('fbgemm'),
            'description': 'INT8 Default Quantization'
        },
        
        'int8_per_channel': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # 🔑 Fix: FBGEMM needs unsigned INT8 activations
                    qscheme=torch.per_tensor_affine  # 🔑 Fix: unsigned INT8 uses affine
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,   # Weights use signed INT8
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'INT8 Per-channel Quantization'
        },
        
        'int8_reduce_range': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # 🔑 Fix: FBGEMM needs unsigned INT8 activations
                    qscheme=torch.per_tensor_affine,
                    reduce_range=True
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=True
                )
            ),
            'description': 'INT8 Reduced Range Quantization (More Conservative)'
        },
        
        'int8_symmetric': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,   # Signed INT8 activations
                    qscheme=torch.per_tensor_symmetric
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'INT8 Symmetric Quantization (Signed Activations)'
        },
        
        'int8_fbgemm_optimized': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # FBGEMM optimization: unsigned activations
                    qscheme=torch.per_tensor_affine,
                    reduce_range=False
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,   # Signed weights
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'FBGEMM Optimized Configuration'
        },
        
        # ===== QNNPACK configs (mobile) =====
        'qnnpack': {
            'qconfig': quantization.get_default_qconfig('qnnpack'),
            'description': 'QNNPACK INT8 Quantization (Mobile Optimization)'
        },
        
        'int8_qnnpack_custom': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # QNNPACK also uses unsigned activations
                    qscheme=torch.per_tensor_affine
                ),
                weight=quantization.MinMaxObserver.with_args(  # QNNPACK uses per-tensor weights
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric
                )
            ),
            'description': 'QNNPACK Custom Configuration'
        },
        
        # ===== High-compatibility configs =====
        'int8_simple': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # 🔑 Unified use of unsigned activations
                    qscheme=torch.per_tensor_affine
                ),
                weight=quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric
                )
            ),
            'description': 'INT8 Simple Configuration (Highest Compatibility)'
        },
        
        # ===== Histogram and moving-average observers =====
        'histogram': {
            'qconfig': quantization.QConfig(
                activation=quantization.HistogramObserver.with_args(
                    dtype=torch.quint8,  # 🔑 Fix
                    qscheme=torch.per_tensor_affine
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'INT8 Histogram Observer'
        },

        'moving_average': {
            'qconfig': quantization.QConfig(
                activation=quantization.MovingAverageMinMaxObserver.with_args(
                    dtype=torch.quint8,  # 🔑 Fix
                    qscheme=torch.per_tensor_affine
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'INT8 Moving Average Observer'
        },
    }
    
    if precision not in configs:
        print(f"⚠️ Unknown quantization precision: {precision}, using default int8")
        precision = 'int8'
    
    return configs[precision]

# Updated static quantization options
STATIC_QUANTIZATION_OPTIONS = {
    # INT8 family
    'int8_default': {
        'precision': 'int8',
        'backend': 'fbgemm',
        'description': 'Default INT8 Quantization',
        'memory_saving': '~75%',
        'precision_loss': 'Medium'
    },
    'int8_per_channel': {
        'precision': 'int8_per_channel',
        'backend': 'fbgemm',
        'description': 'INT8 Per-channel Quantization (Higher Precision)',
        'memory_saving': '~75%',
        'precision_loss': 'Small'
    },
    'int8_reduce_range': {
        'precision': 'int8_reduce_range',
        'backend': 'fbgemm',
        'description': 'INT8 Reduced Range (More Conservative)',
        'memory_saving': '~75%',
        'precision_loss': 'Very Small'
    },
    'int8_asymmetric': {
        'precision': 'int8_asymmetric',
        'backend': 'fbgemm',
        'description': 'INT8 Asymmetric Quantization',
        'memory_saving': '~75%',
        'precision_loss': 'Medium'
    },
    'int8_mobile': {
        'precision': 'qnnpack',
        'backend': 'qnnpack',
        'description': 'QNNPACK Mobile Optimization',
        'memory_saving': '~75%',
        'precision_loss': 'Medium'
    },
    'int8_histogram': {
        'precision': 'histogram',
        'backend': 'fbgemm',
        'description': 'INT8 Histogram Calibration',
        'memory_saving': '~75%',
        'precision_loss': 'Small'
    },
    'int8_moving_avg': {
        'precision': 'moving_average',
        'backend': 'fbgemm',
        'description': 'INT8 Moving Average Calibration',
        'memory_saving': '~75%',
        'precision_loss': 'Small'
    }
}

def get_quantization_option(option_name):
    """Fetch a predefined quantization option"""
    return STATIC_QUANTIZATION_OPTIONS.get(option_name, STATIC_QUANTIZATION_OPTIONS['int8_default'])


def print_available_quantization_options():
    """Print all available quantization options"""
    print("\n=== Available Quantization Configuration Options ===")
    print(f"{'Option Name':<20} {'Description':<30} {'Memory Saving':<10} {'Precision Loss':<10}")
    print("-" * 70)
    
    for name, config in STATIC_QUANTIZATION_OPTIONS.items():
        print(f"{name:<20} {config['description']:<30} {config['memory_saving']:<10} {config['precision_loss']:<10}")
    print()

def fuse_model_modules(model):
    print("⚙️ Starting operator fusion...")
    model.eval()
    for module in model.modules():
        if isinstance(module, MBConvBlock):
            if hasattr(module, 'expand_conv'):
                torch.quantization.fuse_modules(module.expand_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'pw_conv'):
                torch.quantization.fuse_modules(module.pw_conv, ['0', '1'], inplace=True)
        elif isinstance(module, DWSepConvBlock):
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'pw_conv'):
                torch.quantization.fuse_modules(module.pw_conv, ['0', '1'], inplace=True)
        # Fuse SeDpConvBlock modules
        elif isinstance(module, SeDpConvBlock):
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
        # Fuse DpConvBlock modules
        elif isinstance(module, DpConvBlock):
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'pw_conv'):
                torch.quantization.fuse_modules(module.pw_conv, ['0', '1'], inplace=True) 
        # Fuse SeSepConvBlock modules
        elif isinstance(module, SeSepConvBlock):
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'pw_conv'):
                torch.quantization.fuse_modules(module.pw_conv, ['0', '1'], inplace=True)
    print("✅ Operator fusion complete.")

def fuse_QATmodel_modules(model):
    print("⚙️ Starting operator fusion...")
    # QAT preparation must run in train mode
    model.eval()
    # Gather all modules that need fusion
    modules_to_fuse = []
    for name, module in model.named_modules():
        if isinstance(module, MBConvBlock):
            # Fuse MBConvBlock
            if hasattr(module, 'expand_conv') and module.expand_conv is not None:
                expand_conv_0 = f"{name}.expand_conv.0"
                expand_conv_1 = f"{name}.expand_conv.1"
                if expand_conv_0 in dict(model.named_modules()) and expand_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([expand_conv_0, expand_conv_1])
            if hasattr(module, 'dw_conv') and module.dw_conv is not None:
                dw_conv_0 = f"{name}.dw_conv.0"
                dw_conv_1 = f"{name}.dw_conv.1"
                if dw_conv_0 in dict(model.named_modules()) and dw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([dw_conv_0, dw_conv_1])
            if hasattr(module, 'pw_conv') and module.pw_conv is not None:
                pw_conv_0 = f"{name}.pw_conv.0"
                pw_conv_1 = f"{name}.pw_conv.1"
                if pw_conv_0 in dict(model.named_modules()) and pw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([pw_conv_0, pw_conv_1])
                
        elif isinstance(module, DWSepConvBlock):
            # Fuse DWSepConvBlock
            if hasattr(module, 'dw_conv') and module.dw_conv is not None:
                dw_conv_0 = f"{name}.dw_conv.0"
                dw_conv_1 = f"{name}.dw_conv.1"
                if dw_conv_0 in dict(model.named_modules()) and dw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([dw_conv_0, dw_conv_1])
            if hasattr(module, 'pw_conv') and module.pw_conv is not None:
                pw_conv_0 = f"{name}.pw_conv.0"
                pw_conv_1 = f"{name}.pw_conv.1"
                if pw_conv_0 in dict(model.named_modules()) and pw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([pw_conv_0, pw_conv_1])
        # Fuse SeDpConvBlock
        elif isinstance(module, SeDpConvBlock):
            if hasattr(module, 'dw_conv') and module.dw_conv is not None:
                dw_conv_0 = f"{name}.dw_conv.0"
                dw_conv_1 = f"{name}.dw_conv.1"
                if dw_conv_0 in dict(model.named_modules()) and dw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([dw_conv_0, dw_conv_1])
        
        # Fuse DpConvBlock
        elif isinstance(module, DpConvBlock):
            if hasattr(module, 'dw_conv') and module.dw_conv is not None:
                dw_conv_0 = f"{name}.dw_conv.0"
                dw_conv_1 = f"{name}.dw_conv.1"
                if dw_conv_0 in dict(model.named_modules()) and dw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([dw_conv_0, dw_conv_1])
            if hasattr(module, 'pw_conv') and module.pw_conv is not None:
                pw_conv_0 = f"{name}.pw_conv.0"
                pw_conv_1 = f"{name}.pw_conv.1"
                if pw_conv_0 in dict(model.named_modules()) and pw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([pw_conv_0, pw_conv_1])
        
        # Fuse SeSepConvBlock
        elif isinstance(module, SeSepConvBlock):
            if hasattr(module, 'dw_conv') and module.dw_conv is not None:
                dw_conv_0 = f"{name}.dw_conv.0"
                dw_conv_1 = f"{name}.dw_conv.1"
                if dw_conv_0 in dict(model.named_modules()) and dw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([dw_conv_0, dw_conv_1])
            if hasattr(module, 'pw_conv') and module.pw_conv is not None:
                pw_conv_0 = f"{name}.pw_conv.0"
                pw_conv_1 = f"{name}.pw_conv.1"
                if pw_conv_0 in dict(model.named_modules()) and pw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([pw_conv_0, pw_conv_1])
    
    # Perform the fusion
    try:
        if modules_to_fuse:
            torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
            print(f"✅ Fused {len(modules_to_fuse)} groups of modules")
        else:
            print("⚠️ No modules found for fusion")
    except Exception as e:
        print(f"❌ Module fusion failed: {str(e)}")

def apply_configurable_static_quantization(trained_model, dataloader, precision='int8', backend='fbgemm'):
    """Apply configurable static quantization"""
    print(f"🔧 Starting static quantization - Precision: {precision}, Backend: {backend}")
    
    # Configure the quantization backend
    torch.backends.quantized.engine = backend
    trained_model.to('cpu').eval()
    
    # Fuse modules
    print("⚙️ Fusing modules...")
    fuse_model_modules(trained_model)
    
    # Retrieve the quantization configuration
    quant_config = get_static_quantization_config(precision)
    print(f"📋 Using configuration: {quant_config['description']}")
    
    # Apply the quantization configuration
    trained_model.qconfig = quant_config['qconfig']
    
    # Prepare for quantization
    print("⚙️ Preparing quantization...")
    quantization.prepare(trained_model, inplace=True)
    
    # Calibration phase
    print("⚙️ Starting calibration...")
    calibration_loader = create_calibration_loader(dataloader['train'], num_batches=12)
    
    trained_model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(calibration_loader):
            inputs = inputs.to('cpu')
            if inputs.dtype != torch.float32:
                inputs = inputs.float()
            
            try:
                _ = trained_model(inputs)
            except Exception as e:
                print(f"⚠️ Calibration batch {batch_idx} failed: {e}")
                continue
                
            if (batch_idx + 1) % 4 == 0:
                print(f"  Calibration progress: {batch_idx + 1}/12")
    
    print("✅ Calibration complete")
    
    # Convert to a quantized model
    print("⚙️ Converting to quantized model...")
    try:
        quantized_model = quantization.convert(trained_model, inplace=True)
        
        # Analyze the quantization result
        # analyze_quantization_result(quantized_model, precision)
        
        print(f"✅ {precision} static quantization complete")
        return quantized_model
        
    except Exception as e:
        print(f"❌ {precision} quantization failed: {e}")
        print("🔄 Falling back to default INT8 quantization...")
        
        # Fallback to the default configuration
        trained_model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(trained_model, inplace=True)
        
        # Recalibrate
        with torch.no_grad():
            for inputs, _ in calibration_loader:
                inputs = inputs.to('cpu')
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                _ = trained_model(inputs)
        
        quantized_model = quantization.convert(trained_model, inplace=True)
        print("✅ Default quantization complete")
        return quantized_model


def analyze_quantization_result(model, precision):
    """Analyze quantization results"""
    print(f"\n === {precision} Quantization Result Analysis === ")
    
    total_params = 0
    quantized_params = 0
    total_memory = 0
    quantized_memory = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_memory = param_count * param.element_size()
        
        total_params += param_count
        total_memory += param_memory
        
        if 'qint' in str(param.dtype):
            quantized_params += param_count
            quantized_memory += param_memory
            status = "✅ Quantized"
        else:
            status = "❌ Not Quantized"
        
        print(f"{name}:")
        print(f"  Parameter Count: {param_count:,}")
        print(f"  Type: {param.dtype}")
        print(f"  Memory: {param_memory/1024/1024:.3f} MB")
        print(f"  Status: {status}")
        print()
    
    # Aggregate statistics
    quantization_ratio = quantized_params / total_params * 100 if total_params > 0 else 0
    memory_ratio = quantized_memory / total_memory * 100 if total_memory > 0 else 0
    
    print("=== Quantization Statistics ===")
    print(f"Total Parameters: {total_params:,}")
    print(f"Quantized Parameters: {quantized_params:,}")
    print(f"Quantization Ratio: {quantization_ratio:.1f}%")
    print(f"Total Memory: {total_memory/1024/1024:.2f} MB")
    print(f"Quantized Memory: {quantized_memory/1024/1024:.2f} MB") 
    print(f"Memory Quantization Ratio: {memory_ratio:.1f}%")
    
    # Estimate memory savings
    if total_params > 0:
        fp32_memory = total_params * 4 / 1024 / 1024  # FP32 baseline
        current_memory = total_memory / 1024 / 1024
        memory_saving = (1 - current_memory / fp32_memory) * 100
        print(f"Memory Saving vs FP32: {memory_saving:.1f}%")
    
    return {
        'quantization_ratio': quantization_ratio,
        'memory_saving': memory_saving if total_params > 0 else 0
    }


class QuantizableModel(torch.nn.Module):
    """
    Quantizable model wrapper that injects quantization stubs
    """
    def __init__(self, model):
        super(QuantizableModel, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.quant_mode = None  # Keep track of the quantization mode
        self.quant_enabled = True
        
    def enable_quant(self):
        self.quant_enabled = True
        
    def disable_quant(self):
        self.quant_enabled = False

    def set_quant_mode(self, mode):
        self.quant_mode = mode
    
    def forward(self, x):
        if self.quant_mode in ['dynamic', 'static', 'qat']:  # Only execute in quantization modes
            x = self.quant(x)
            x = self.model(x)
            x = self.dequant(x)
        else:
            x = self.model(x)
        return x


    # Additional helpers to remain compatible with the original model
    @property
    def output_dim(self):
        return self.model.output_dim

    # Extend state-dict handling
    def load_state_dict(self, state_dict, strict=True):
        # Try loading directly first
        try:
            super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            if "Missing key(s)" in str(e):
                # Fix state-dict key names
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                super().load_state_dict(new_state_dict, strict=False)
            else:
                raise e
            
    # Provide static quantization support
    def prepare_static_quantization(self):
        """
        Prepare for static quantization by inserting observers.
        """
        self.eval()  # Ensure the model is in eval mode
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # fbgemm or 'qnnpack'
        # Special handling for GroupNorm layers
        torch.quantization.quantize_default_mappings[torch.nn.GroupNorm] = torch.quantization.default_float_to_quantized_operator_mappings[torch.nn.GroupNorm]
        torch.quantization.prepare(self, inplace=True)  # Insert observers
        print("Model structure after quantization preparation:")
        print(self)  # Display the structure to verify observers

    def convert_static_quantization(self):
        """
        Convert into a statically quantized model
        """
        quantization.convert(self, inplace=True)
