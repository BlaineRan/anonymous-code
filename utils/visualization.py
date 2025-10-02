import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from matplotlib.gridspec import GridSpec

def plot_architecture_explanation(
    explanation,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    dpi: int = 100
):
    """
    可视化架构解释结果
    
    参数:
        explanation: ArchitectureExplanation 实例
        save_path: 可选，图片保存路径
        figsize: 图表大小
        dpi: 图片分辨率
    """
    # 创建图表
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(3, 2, figure=fig)
    
    # 1. 总体策略展示
    ax_overall = fig.add_subplot(gs[0, :])
    ax_overall.axis('off')
    ax_overall.set_title('Overall Design Strategy', fontsize=14, pad=20)
    ax_overall.text(0.05, 0.5, 
                   explanation.overall_strategy,
                   ha='left', va='center', wrap=True, fontsize=12)
    
    # 2. 内核选择解释
    ax_kernel = fig.add_subplot(gs[1, 0])
    ax_kernel.axis('off')
    ax_kernel.set_title('Kernel Size Selection', fontsize=12, pad=10)
    ax_kernel.text(0.05, 0.5, 
                  explanation.kernel_choice,
                  ha='left', va='center', wrap=True, fontsize=10)
    
    # 3. 扩展因子解释
    ax_expansion = fig.add_subplot(gs[1, 1])
    ax_expansion.axis('off')
    ax_expansion.set_title('Expansion Factor Strategy', fontsize=12, pad=10)
    ax_expansion.text(0.05, 0.5, 
                     explanation.expansion_reason,
                     ha='left', va='center', wrap=True, fontsize=10)
    
    # 4. SE模块解释
    ax_se = fig.add_subplot(gs[2, 0])
    ax_se.axis('off')
    ax_se.set_title('SE Block Usage', fontsize=12, pad=10)
    ax_se.text(0.05, 0.5, 
              explanation.se_block_usage,
              ha='left', va='center', wrap=True, fontsize=10)
    
    # 5. 跳跃连接解释
    ax_skip = fig.add_subplot(gs[2, 1])
    ax_skip.axis('off')
    ax_skip.set_title('Skip Connection Strategy', fontsize=12, pad=10)
    ax_skip.text(0.05, 0.5, 
                explanation.skip_connection_strategy,
                ha='left', va='center', wrap=True, fontsize=10)
    
    # 添加置信度指示器
    fig.text(0.95, 0.05, 
            f"Confidence: {explanation.confidence_score:.0%}",
            ha='right', va='bottom', fontsize=10,
            bbox=dict(facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_explanation_json(explanation, file_path: str):
    """
    将解释保存为JSON文件
    
    参数:
        explanation: ArchitectureExplanation 实例
        file_path: 保存路径
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(asdict(explanation), f, indent=2)

def print_explanation(explanation):
    """
    在控制台打印解释结果
    
    参数:
        explanation: ArchitectureExplanation 实例
    """
    print("\n" + "="*40)
    print("ARCHITECTURE EXPLANATION".center(40))
    print("="*40)
    print(f"\n🔹 Overall Strategy:\n{explanation.overall_strategy}")
    print(f"\n🔹 Kernel Selection:\n{explanation.kernel_choice}")
    print(f"\n🔹 Expansion Factors:\n{explanation.expansion_reason}")
    print(f"\n🔹 SE Blocks:\n{explanation.se_block_usage}")
    print(f"\n🔹 Skip Connections:\n{explanation.skip_connection_strategy}")
    print(f"\nConfidence: {explanation.confidence_score:.0%}")
    print("="*40 + "\n")

# 导出函数
__all__ = ['plot_architecture_explanation', 'save_explanation_json', 'print_explanation']