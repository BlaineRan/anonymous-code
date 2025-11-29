import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch_geometric.loader import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from GNNEncoder import ArchitectureEncoder
from GNNdataloader import ArchitectureDataset
from Predictor import GNNPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm  # Import tqdm
import time

def train_predictor():
    # Initialize encoder
    encoder = ArchitectureEncoder()
    # Create dataset - using new split method
    # /root/tinyml/GNNPredictor/arch_data/MMAct
    base_dir = '/root/tinyml/GNNPredictor/arch_data/MMAct'
    train_dataset = ArchitectureDataset(base_dir, encoder, subset="train")
    val_dataset = ArchitectureDataset(base_dir, encoder, subset="val")
    test_dataset = ArchitectureDataset(base_dir, encoder, subset="test")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = GNNPredictor(input_dim=encoder.base_feature_dim + 1, output_dim=3)  # +1 for channels
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training parameter settings
    max_epochs = 150  # Maximum training epochs
    patience = 15     # Early stopping patience
    min_delta = 0.001  # Minimum improvement threshold

    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        # Progress bar for training phase
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} - Training", leave=False)
        for batch in train_pbar:
            optimizer.zero_grad()
            pred = model(batch) # shape: [batch_size, 3]
            # print(batch.y)
            # Ensure batch.y is 2D [batch_size, 2]
            if batch.y.dim() == 1:
                # If 1D, reshape to 2D
                # batch_y = batch.y.view(-1, 2)
                batch_y = batch.y.view(-1, 3)
            else:
                batch_y = batch.y

            # loss = criterion(pred.squeeze(), batch.y)
            # Calculate loss for two tasks
            loss_original = criterion(pred[:, 0], batch_y[:, 0])
            loss_quantized = criterion(pred[:, 1], batch_y[:, 1])
            loss_qat = criterion(pred[:, 2], batch_y[:, 2])
            
            # Weighted combination of losses
            loss = (loss_original + loss_quantized + loss_qat) / 3
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Update training progress bar
            train_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        # Progress bar for validation phase
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} - Validation", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                pred = model(batch)
                if batch.y.dim() == 1:
                    # If 1D, reshape to 2D
                    # batch_y = batch.y.view(-1, 2)
                    batch_y = batch.y.view(-1, 3)
                else:
                    batch_y = batch.y

                loss_original = criterion(pred[:, 0], batch_y[:, 0])
                loss_quantized = criterion(pred[:, 1], batch_y[:, 1])
                loss_qat = criterion(pred[:, 2], batch_y[:, 2])
                batch_val_loss = (loss_original + loss_quantized + loss_qat) / 3
                # val_loss += criterion(pred.squeeze(), batch.y).item()
                val_loss += batch_val_loss.item()
                # Update validation progress bar
                val_pbar.set_postfix({"Val Loss": f"{batch_val_loss.item():.4f}"})
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Early stopping mechanism
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f'Epoch {epoch+1}: Best validation loss updated to {best_val_loss:.4f}')
        else:
            epochs_no_improve += 1
            print(f'Epoch {epoch+1}: Validation loss not improved ({epochs_no_improve}/{patience})')
        
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = max_epochs - epoch - 1
            estimated_remaining_time = avg_time_per_epoch * remaining_epochs

            print(f'Epoch {epoch+1}/{max_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Best Val: {best_val_loss:.4f}, '
                  f'ETA: {estimated_remaining_time/60:.1f} minutes')
            
        # Check early stopping condition
        if epochs_no_improve >= patience:
            print(f'\nEarly stopping triggered! Stopping training after {epoch+1} epochs, '
                  f'validation loss not improved for {patience} consecutive epochs')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model weights (Validation Loss: {best_val_loss:.4f})')
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f'Total training time: {total_time/60:.1f} minutes')

    # Plot loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {best_val_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Return important information during training
    training_info = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time
    }

    return model, encoder, test_loader, training_info

def test_predictor_performance(model, test_loader):
    """Evaluate predictor performance using the full test set"""
    model.eval()
    
    print("=" * 60)
    print("Model Performance Evaluation on Test Set")
    print("=" * 60)
    
    total_error_original = 0
    total_error_quantized = 0
    total_error_qat = 0
    total_samples = 0
    
    predictions_qat = []
    predictions_original = []
    predictions_quantized = []
    ground_truths_original = []
    ground_truths_quantized = []
    ground_truths_qat = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in test_pbar:
            pred = model(batch)
            
            # Ensure labels are 2D [batch_size, 3]
            if batch.y.dim() == 1:
                batch_y = batch.y.view(-1, 3)
            else:
                batch_y = batch.y
            
            # Calculate error for each sample
            batch_size = batch_y.size(0)
            total_samples += batch_size
            
            for i in range(batch_size):
                pred_original = pred[i, 0].item()
                pred_quantized = pred[i, 1].item()
                pred_qat = pred[i, 2].item()
                
                true_original = batch_y[i, 0].item()
                true_quantized = batch_y[i, 1].item()
                true_qat = batch_y[i, 2].item()
                
                # Calculate error
                error_original = abs(pred_original - true_original)
                error_quantized = abs(pred_quantized - true_quantized)
                error_qat = abs(pred_qat - true_qat)
                
                total_error_original += error_original
                total_error_quantized += error_quantized
                total_error_qat += error_qat
                
                predictions_original.append(pred_original)
                predictions_quantized.append(pred_quantized)
                predictions_qat.append(pred_qat)
                ground_truths_original.append(true_original)
                ground_truths_quantized.append(true_quantized)
                ground_truths_qat.append(true_qat)
            
            # Update progress bar
            current_mae_original = total_error_original / total_samples
            current_mae_quantized = total_error_quantized / total_samples
            current_mae_qat = total_error_qat / total_samples
            test_pbar.set_postfix({
                "MAE_orig": f"{current_mae_original:.4f}",
                "MAE_quant": f"{current_mae_quantized:.4f}",
                "MAE_qat": f"{current_mae_qat:.4f}"
            })
    
    # Statistical results
    mae_original = total_error_original / total_samples
    mae_quantized = total_error_quantized / total_samples
    mae_qat = total_error_qat / total_samples

    rmse_original = np.sqrt(np.mean((np.array(predictions_original) - np.array(ground_truths_original)) ** 2))
    rmse_quantized = np.sqrt(np.mean((np.array(predictions_quantized) - np.array(ground_truths_quantized)) ** 2))
    rmse_qat = np.sqrt(np.mean((np.array(predictions_qat) - np.array(ground_truths_qat)) ** 2))
    
    print("\n" + "=" * 60)
    print("Overall Performance Metrics on Test Set:")
    print(f"Number of Test Samples: {total_samples}")
    print(f"Original Accuracy - MAE: {mae_original:.4f}%, RMSE: {rmse_original:.4f}%")
    print(f"Quantized Accuracy - MAE: {mae_quantized:.4f}%, RMSE: {rmse_quantized:.4f}%")
    print(f"QAT Accuracy - MAE: {mae_qat:.4f}%, RMSE: {rmse_qat:.4f}%")
    print(f"Original Accuracy - Prediction Range: {min(predictions_original):.2f}% - {max(predictions_original):.2f}%")
    print(f"Original Accuracy - True Range: {min(ground_truths_original):.2f}% - {max(ground_truths_original):.2f}%")
    print(f"Quantized Accuracy - Prediction Range: {min(predictions_quantized):.2f}% - {max(predictions_quantized):.2f}%")
    print(f"Quantized Accuracy - True Range: {min(ground_truths_quantized):.2f}% - {max(ground_truths_quantized):.2f}%")
    print(f"QAT Accuracy - Prediction Range: {min(predictions_qat):.2f}% - {max(predictions_qat):.2f}%")
    print(f"QAT Accuracy - True Range: {min(ground_truths_qat):.2f}% - {max(ground_truths_qat):.2f}%")
    print("=" * 60)

    return (mae_original, rmse_original), (mae_quantized, rmse_quantized), (mae_qat, rmse_qat)

# Run training
predictor_model, encoder, test_loader, training_info = train_predictor()
# Test model performance
# mae, rmse = test_predictor_performance(predictor_model, test_dataset, num_samples=5)
# Test model performance
(mae_orig, rmse_orig), (mae_quant, rmse_quant), (mae_qat, rmse_qat) = test_predictor_performance(predictor_model, test_loader)

# Optional: Save trained model
torch.save({
    'model_state_dict': predictor_model.state_dict(),
    'encoder': encoder,
    'train_losses': training_info['train_losses'],
    'val_losses': training_info['val_losses'],
    'best_val_loss': training_info['best_val_loss'],
    'total_training_time': training_info['total_time']
}, '/root/tinyml/GNNPredictor/model/MMAct/trained_predictor.pth')

# print(f"Model saved, test performance: MAE={mae:.2f}%, RMSE={rmse:.2f}%")
print(f"Model saved, test performance:")
print(f"Original Accuracy - MAE={mae_orig:.2f}%, RMSE={rmse_orig:.2f}%")
print(f"Quantized Accuracy - MAE={mae_quant:.2f}%, RMSE={rmse_quant:.2f}%")
print(f"QAT Accuracy - MAE={mae_qat:.2f}%, RMSE={rmse_qat:.2f}%")
print(f"Best Validation Loss: {training_info['best_val_loss']:.4f}")
print(f"Total Training Time: {training_info['total_time']/60:.1f} minutes")