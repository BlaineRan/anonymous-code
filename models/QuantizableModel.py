import torch
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
from .conv_blocks import MBConvBlock, DWSepConvBlock, SeDpConvBlock, DpConvBlock, SeSepConvBlock
from data import create_calibration_loader

def get_static_quantization_config(precision='int8'):
    """获取不同精度的静态量化配置（FBGEMM兼容版）"""
    
    configs = {
        # ===== FBGEMM兼容的INT8配置 =====
        'int8': {
            'qconfig': quantization.get_default_qconfig('fbgemm'),
            'description': 'INT8 默认量化'
        },
        
        'int8_per_channel': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # 🔑 修正：FBGEMM需要无符号INT8激活
                    qscheme=torch.per_tensor_affine  # 🔑 修正：无符号INT8使用affine
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,   # 权重使用有符号INT8
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'INT8 逐通道量化'
        },
        
        'int8_reduce_range': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # 🔑 修正：FBGEMM需要无符号INT8激活
                    qscheme=torch.per_tensor_affine,
                    reduce_range=True
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=True
                )
            ),
            'description': 'INT8 减少范围量化 (更保守)'
        },
        
        'int8_symmetric': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,   # 有符号INT8激活
                    qscheme=torch.per_tensor_symmetric
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'INT8 对称量化 (有符号激活)'
        },
        
        'int8_fbgemm_optimized': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # FBGEMM优化：无符号激活
                    qscheme=torch.per_tensor_affine,
                    reduce_range=False
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,   # 有符号权重
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'FBGEMM优化配置'
        },
        
        # ===== QNNPACK配置（移动端） =====
        'qnnpack': {
            'qconfig': quantization.get_default_qconfig('qnnpack'),
            'description': 'QNNPACK INT8量化 (移动端优化)'
        },
        
        'int8_qnnpack_custom': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # QNNPACK也使用无符号激活
                    qscheme=torch.per_tensor_affine
                ),
                weight=quantization.MinMaxObserver.with_args(  # QNNPACK使用per-tensor权重
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric
                )
            ),
            'description': 'QNNPACK自定义配置'
        },
        
        # ===== 高兼容性配置 =====
        'int8_simple': {
            'qconfig': quantization.QConfig(
                activation=quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,  # 🔑 统一使用无符号激活
                    qscheme=torch.per_tensor_affine
                ),
                weight=quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric
                )
            ),
            'description': 'INT8 简化配置 (最高兼容性)'
        },
        
        # ===== 直方图和移动平均观察器 =====
        'histogram': {
            'qconfig': quantization.QConfig(
                activation=quantization.HistogramObserver.with_args(
                    dtype=torch.quint8,  # 🔑 修正
                    qscheme=torch.per_tensor_affine
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'INT8 直方图观察器'
        },

        'moving_average': {
            'qconfig': quantization.QConfig(
                activation=quantization.MovingAverageMinMaxObserver.with_args(
                    dtype=torch.quint8,  # 🔑 修正
                    qscheme=torch.per_tensor_affine
                ),
                weight=quantization.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric
                )
            ),
            'description': 'INT8 移动平均观察器'
        },
    }
    
    if precision not in configs:
        print(f"⚠️ 未知的量化精度: {precision}, 使用默认的 int8")
        precision = 'int8'
    
    return configs[precision]

# 更新的静态量化配置选项
STATIC_QUANTIZATION_OPTIONS = {
    # INT8 系列
    'int8_default': {
        'precision': 'int8',
        'backend': 'fbgemm',
        'description': '默认INT8量化',
        'memory_saving': '~75%',
        'precision_loss': '中等'
    },
    'int8_per_channel': {
        'precision': 'int8_per_channel',
        'backend': 'fbgemm',
        'description': 'INT8逐通道量化 (更高精度)',
        'memory_saving': '~75%',
        'precision_loss': '较小'
    },
    'int8_reduce_range': {
        'precision': 'int8_reduce_range',
        'backend': 'fbgemm',
        'description': 'INT8减少范围 (更保守)',
        'memory_saving': '~75%',
        'precision_loss': '很小'
    },
    'int8_asymmetric': {
        'precision': 'int8_asymmetric',
        'backend': 'fbgemm',
        'description': 'INT8非对称量化',
        'memory_saving': '~75%',
        'precision_loss': '中等'
    },
    'int8_mobile': {
        'precision': 'qnnpack',
        'backend': 'qnnpack',
        'description': 'QNNPACK移动端优化',
        'memory_saving': '~75%',
        'precision_loss': '中等'
    },
    'int8_histogram': {
        'precision': 'histogram',
        'backend': 'fbgemm',
        'description': 'INT8直方图校准',
        'memory_saving': '~75%',
        'precision_loss': '较小'
    },
    'int8_moving_avg': {
        'precision': 'moving_average',
        'backend': 'fbgemm',
        'description': 'INT8移动平均校准',
        'memory_saving': '~75%',
        'precision_loss': '较小'
    }
}

def get_quantization_option(option_name):
    """获取预定义的量化选项"""
    return STATIC_QUANTIZATION_OPTIONS.get(option_name, STATIC_QUANTIZATION_OPTIONS['int8_default'])


def print_available_quantization_options():
    """打印所有可用的量化选项"""
    print("\n=== 可用的量化配置选项 ===")
    print(f"{'选项名称':<20} {'描述':<30} {'内存节省':<10} {'精度损失':<10}")
    print("-" * 70)
    
    for name, config in STATIC_QUANTIZATION_OPTIONS.items():
        print(f"{name:<20} {config['description']:<30} {config['memory_saving']:<10} {config['precision_loss']:<10}")
    print()

def fuse_model_modules(model):
    print("⚙️ 开始算子融合...")
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
        # 对 SeDpConvBlock 进行融合
        elif isinstance(module, SeDpConvBlock):
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
        # 对 DpConvBlock 进行融合
        elif isinstance(module, DpConvBlock):
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'pw_conv'):
                torch.quantization.fuse_modules(module.pw_conv, ['0', '1'], inplace=True) 
        # 对 SeSepConvBlock 进行融合
        elif isinstance(module, SeSepConvBlock):
            if hasattr(module, 'dw_conv'):
                torch.quantization.fuse_modules(module.dw_conv, ['0', '1'], inplace=True)
            if hasattr(module, 'pw_conv'):
                torch.quantization.fuse_modules(module.pw_conv, ['0', '1'], inplace=True)
    print("✅ 算子融合完成。")

def fuse_QATmodel_modules(model):
    print("⚙️ 开始算子融合...")
    # 必须在训练模式下进行QAT准备
    model.eval()
    # 获取所有需要融合的模块
    modules_to_fuse = []
    for name, module in model.named_modules():
        if isinstance(module, MBConvBlock):
            # MBConvBlock的融合
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
            # DWSepConvBlock的融合
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
        # SeDpConvBlock的融合
        elif isinstance(module, SeDpConvBlock):
            if hasattr(module, 'dw_conv') and module.dw_conv is not None:
                dw_conv_0 = f"{name}.dw_conv.0"
                dw_conv_1 = f"{name}.dw_conv.1"
                if dw_conv_0 in dict(model.named_modules()) and dw_conv_1 in dict(model.named_modules()):
                    modules_to_fuse.append([dw_conv_0, dw_conv_1])
        
        # DpConvBlock的融合
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
        
        # SeSepConvBlock的融合
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
    
    # 执行融合
    try:
        if modules_to_fuse:
            torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
            print(f"✅ 融合了 {len(modules_to_fuse)} 组模块")
        else:
            print("⚠️ 未找到需要融合的模块")
    except Exception as e:
        print(f"❌ 模块融合失败: {str(e)}")

def apply_configurable_static_quantization(trained_model, dataloader, precision='int8', backend='fbgemm'):
    """应用可配置的静态量化"""
    print(f"🔧 开始静态量化 - 精度: {precision}, 后端: {backend}")
    
    # 设置量化后端
    torch.backends.quantized.engine = backend
    trained_model.to('cpu').eval()
    
    # 融合模块
    print("⚙️ 融合模块...")
    fuse_model_modules(trained_model)
    
    # 获取量化配置
    quant_config = get_static_quantization_config(precision)
    print(f"📋 使用配置: {quant_config['description']}")
    
    # 应用量化配置
    trained_model.qconfig = quant_config['qconfig']
    
    # 准备量化
    print("⚙️ 准备量化...")
    quantization.prepare(trained_model, inplace=True)
    
    # 校准阶段
    print("⚙️ 开始校准...")
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
                print(f"⚠️ 校准批次 {batch_idx} 失败: {e}")
                continue
                
            if (batch_idx + 1) % 4 == 0:
                print(f"  校准进度: {batch_idx + 1}/12")
    
    print("✅ 校准完成")
    
    # 转换为量化模型
    print("⚙️ 转换为量化模型...")
    try:
        quantized_model = quantization.convert(trained_model, inplace=True)
        
        # 分析量化结果
        # analyze_quantization_result(quantized_model, precision)
        
        print(f"✅ {precision} 静态量化完成")
        return quantized_model
        
    except Exception as e:
        print(f"❌ {precision} 量化失败: {e}")
        print("🔄 回退到默认INT8量化...")
        
        # 回退到默认配置
        trained_model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(trained_model, inplace=True)
        
        # 重新校准
        with torch.no_grad():
            for inputs, _ in calibration_loader:
                inputs = inputs.to('cpu')
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                _ = trained_model(inputs)
        
        quantized_model = quantization.convert(trained_model, inplace=True)
        print("✅ 默认量化完成")
        return quantized_model


def analyze_quantization_result(model, precision):
    """分析量化结果"""
    print(f"\n === {precision} 量化结果分析 === ")
    
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
            status = "✅ 已量化"
        else:
            status = "❌ 未量化"
        
        print(f"{name}:")
        print(f"  参数数: {param_count:,}")
        print(f"  类型: {param.dtype}")
        print(f"  内存: {param_memory/1024/1024:.3f} MB")
        print(f"  状态: {status}")
        print()
    
    # 统计结果
    quantization_ratio = quantized_params / total_params * 100 if total_params > 0 else 0
    memory_ratio = quantized_memory / total_memory * 100 if total_memory > 0 else 0
    
    print("=== 量化统计 ===")
    print(f"总参数数: {total_params:,}")
    print(f"量化参数数: {quantized_params:,}")
    print(f"量化比例: {quantization_ratio:.1f}%")
    print(f"总内存: {total_memory/1024/1024:.2f} MB")
    print(f"量化后内存: {quantized_memory/1024/1024:.2f} MB") 
    print(f"内存量化比例: {memory_ratio:.1f}%")
    
    # 估算内存节省
    if total_params > 0:
        fp32_memory = total_params * 4 / 1024 / 1024  # FP32基准
        current_memory = total_memory / 1024 / 1024
        memory_saving = (1 - current_memory / fp32_memory) * 100
        print(f"相比FP32内存节省: {memory_saving:.1f}%")
    
    return {
        'quantization_ratio': quantization_ratio,
        'memory_saving': memory_saving if total_params > 0 else 0
    }


class QuantizableModel(torch.nn.Module):
    """
    可量化模型包装器，添加量化stub
    """
    def __init__(self, model):
        super(QuantizableModel, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.quant_mode = None  # 添加量化模式标志
        self.quant_enabled = True
        
    def enable_quant(self):
        self.quant_enabled = True
        
    def disable_quant(self):
        self.quant_enabled = False

    def set_quant_mode(self, mode):
        self.quant_mode = mode
    
    def forward(self, x):
        if self.quant_mode in ['dynamic', 'static', 'qat']:  # 仅在量化模式下执行
            x = self.quant(x)
            x = self.model(x)
            x = self.dequant(x)
        else:
            x = self.model(x)
        return x


    # 添加以下方法以保持与原始模型的兼容性
    @property
    def output_dim(self):
        return self.model.output_dim

    # 添加状态字典处理
    def load_state_dict(self, state_dict, strict=True):
        # 先尝试直接加载
        try:
            super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            if "Missing key(s)" in str(e):
                # 修复状态字典键名
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                super().load_state_dict(new_state_dict, strict=False)
            else:
                raise e
            
    # 添加静态量化支持
    def prepare_static_quantization(self):
        """
        准备静态量化：为模型插入观察器。
        """
        self.eval()  # 确保模型处于评估模式
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')  #  fbgemm 或 'qnnpack'
        # 特别处理GroupNorm层
        torch.quantization.quantize_default_mappings[torch.nn.GroupNorm] = torch.quantization.default_float_to_quantized_operator_mappings[torch.nn.GroupNorm]
        torch.quantization.prepare(self, inplace=True)  # 插入观察器
        print("量化准备后的模型结构:")
        print(self)  # 打印模型结构，检查观察器是否正确插入

    def convert_static_quantization(self):
        """
        转换为静态量化模型
        """
        quantization.convert(self, inplace=True)