import torch
import os
from pathlib import Path

def fix_checkpoints():
    """Repair checkpoint files that contain encoder state."""
    model_dirs = [
        '/root/tinyml/GNNPredictor/model/UTD-MHAD',
        '/root/tinyml/GNNPredictor/model/Wharf',
        '/root/tinyml/GNNPredictor/model/Mhealth',
        '/root/tinyml/GNNPredictor/model/USCHAD',
        '/root/tinyml/GNNPredictor/model/MMAct',
    ]
    
    for model_dir in model_dirs:
        checkpoint_path = os.path.join(model_dir, 'trained_predictor.pth')
        if os.path.exists(checkpoint_path):
            try:
                print(f"Processing: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                
                # Build a new checkpoint with only the required parts
                new_checkpoint = {
                    'model_state_dict': checkpoint['model_state_dict'],
                    'train_losses': checkpoint.get('train_losses', []),
                    'val_losses': checkpoint.get('val_losses', []),
                    'best_val_loss': checkpoint.get('best_val_loss', 0),
                    'total_training_time': checkpoint.get('total_training_time', 0),
                }
                
                # Save the repaired checkpoint
                torch.save(new_checkpoint, checkpoint_path)
                print(f"✅ Repair complete: {checkpoint_path}")
                
            except Exception as e:
                print(f"❌ Repair failed {checkpoint_path}: {e}")

if __name__ == "__main__":
    fix_checkpoints()
