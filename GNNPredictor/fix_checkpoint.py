import torch
import os
from pathlib import Path

def fix_checkpoints():
    """修复所有包含encoder的checkpoint文件"""
    model_dirs = [
        '/root/tinyml/GNNPredictor/model/UTD-MHAD',
        '/root/tinyml/GNNPredictor/model/Wharf', 
        '/root/tinyml/GNNPredictor/model/Mhealth',
        '/root/tinyml/GNNPredictor/model/USCHAD',
        '/root/tinyml/GNNPredictor/model/MMAct'
    ]
    
    for model_dir in model_dirs:
        checkpoint_path = os.path.join(model_dir, 'trained_predictor.pth')
        if os.path.exists(checkpoint_path):
            try:
                print(f"处理: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                
                # 创建新的checkpoint，只包含必要的部分
                new_checkpoint = {
                    'model_state_dict': checkpoint['model_state_dict'],
                    'train_losses': checkpoint.get('train_losses', []),
                    'val_losses': checkpoint.get('val_losses', []),
                    'best_val_loss': checkpoint.get('best_val_loss', 0),
                    'total_training_time': checkpoint.get('total_training_time', 0)
                }
                
                # 保存修复后的checkpoint
                torch.save(new_checkpoint, checkpoint_path)
                print(f"✅ 修复完成: {checkpoint_path}")
                
            except Exception as e:
                print(f"❌ 修复失败 {checkpoint_path}: {e}")

if __name__ == "__main__":
    fix_checkpoints()