import os
import json
import random

def reduce_stage_files(input_dir, output_dir, keep_ratio=0.45):
    """
    将每个stage文件随机保留指定比例的数据
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        keep_ratio: 保留比例 (0-1之间)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 四个stage文件
    stage_files = [
        'stage1_architecture_performance.jsonl',
        'stage2_architecture_performance.jsonl', 
        'stage3_architecture_performance.jsonl',
        'stage4_architecture_performance.jsonl'
    ]
    
    total_original_count = 0
    total_reduced_count = 0
    
    for stage_file in stage_files:
        input_path = os.path.join(input_dir, stage_file)
        output_path = os.path.join(output_dir, stage_file)
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping")
            continue
        
        # 读取原始数据
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        original_count = len(lines)
        total_original_count += original_count
        
        # 随机打乱并选择指定比例的数据
        random.shuffle(lines)
        keep_count = int(original_count * keep_ratio)
        selected_lines = lines[:keep_count]
        total_reduced_count += keep_count
        
        # 写入新的文件
        with open(output_path, 'w') as f:
            f.writelines(selected_lines)
        
        print(f"{stage_file}: {original_count} -> {keep_count} models ({(keep_count/original_count)*100:.1f}%)")
    
    print(f"\nTotal: {total_original_count} -> {total_reduced_count} models ({(total_reduced_count/total_original_count)*100:.1f}%)")
    print(f"Data saved to: {output_dir}")

def main():
    input_dir = '/root/tinyml/GNNPredictor/arch_data/UTD-MHAD'
    output_dir = '/root/tinyml/GNNPredictor/arch_data/UTD-MHAD(1)'
    
    # 首先统计原始文件的数量
    stage_files = ['stage1', 'stage2', 'stage3', 'stage4']
    
    print("Original file statistics:")
    total_count = 0
    for stage in stage_files:
        file_path = os.path.join(input_dir, f"{stage}_architecture_performance.jsonl")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                count = len(f.readlines())
            print(f"{stage}: {count} models")
            total_count += count
        else:
            print(f"{stage}: file not found")
    
    print(f"Total models: {total_count}")
    print("\n" + "="*50)
    
    # 减少数据量
    reduce_stage_files(input_dir, output_dir, keep_ratio=0.45)

if __name__ == "__main__":
    main()