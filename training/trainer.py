import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Set random seed
SEED = 42  # You can choose any integer as seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class SingleTaskTrainer:
    """
    Trainer for single dataset
    """
    def __init__(self, model, dataloaders, device='cuda', logger=None):
        """
        Initialize trainer

        Args:
            model: Model to train
            dataloaders: Dictionary of dataloaders, containing 'train' and 'test' keys
            device: Training device ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.logger = logger

        # If no logger provided, create a simple logger to simulate print behavior

        # Ensure model has output_dim attribute
        if not hasattr(model, 'output_dim'):
            raise AttributeError("Model must have 'output_dim' attribute")

        # Get number of classes
        self.num_classes = len(dataloaders['train'].dataset.classes)
        print(f"Number of classes: {self.num_classes}")

        # Create task head
        self.task_head = nn.Linear(model.output_dim, self.num_classes).to(device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            list(model.parameters()) + list(self.task_head.parameters()),
            lr=1e-3
        )

    def _output(self, message):
        """Unified output method: use logger if available, otherwise use print"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def train_epoch(self):
        """
        Single training epoch
        """
        self.model.train()
        self.task_head.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.dataloaders['train'], desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            features = self.model(inputs)
            outputs = self.task_head(features)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        metrics = {
            'loss': running_loss / len(self.dataloaders['train']),
            'accuracy': 100. * correct / total
        }
        return metrics

    def evaluate(self):
        """
        Model evaluation
        """
        self.model.eval()
        self.task_head.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['test'], desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                features = self.model(inputs)
                outputs = self.task_head(features)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        metrics = {
            'loss': running_loss / len(self.dataloaders['test']),
            'accuracy': 100. * correct / total
        }
        return metrics

    def train(self, epochs=10, save_path='best_model.pth'):
        """
        Train model and save best weights

        Args:
            epochs: Number of training epochs
            save_path: Path to save best model weights
        Returns:
            best_accuracy: Best validation accuracy
            best_val_metrics: Best validation metrics
            history: Training history
            best_model_state: Best model state dictionary
        """
        best_accuracy = 0.0
        best_val_metrics = None  # Save best validation metrics
        history = []
        best_model_state = None  # Save best model state

        for epoch in range(epochs):
            # print(f"\nEpoch {epoch + 1}/{epochs}")
            self._output(f"\nEpoch {epoch + 1}/{epochs}")

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.evaluate()

            # Save history
            history.append({
                'train': train_metrics,
                'val': val_metrics
            })

            # print(f"\nValidation Accuracy: {val_metrics['accuracy']:.2f}%")
            self._output(f"\nValidation Accuracy: {val_metrics['accuracy']:.2f}%")

            # Update best model
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                best_val_metrics = val_metrics
                best_model_state = {
                    'model': self.model.state_dict(),
                    'head': self.task_head.state_dict()
                }

            # Save best model weights to file
            torch.save(best_model_state, save_path)
            # print(f"✅ Best model saved with accuracy: {best_accuracy:.2f}%")
            self._output(f"✅ Best model saved with accuracy: {best_accuracy:.2f}%")

        return best_accuracy, best_val_metrics, history, best_model_state

class MultiTaskTrainer:
    def __init__(self, model, dataloaders, device='cuda'):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        
        self.device = device

        # Ensure model has output_dim attribute
        if not hasattr(model, 'output_dim'):
            raise AttributeError("Model must have 'output_dim' attribute")

        
        # Dynamically create multi-task heads (based on dataset class counts)
        self.task_heads = nn.ModuleDict()
        for task_name, loaders in dataloaders.items():
            if loaders['train'] is None:
                continue
                
            # Get number of classes from dataset
            num_classes = len(loaders['train'].dataset.classes)
            print(f"Creating task head for {task_name}: input_dim={model.output_dim}, output_dim={num_classes}")
            self.task_heads[task_name] = nn.Linear(model.output_dim, num_classes).to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            list(model.parameters()) + list(self.task_heads.parameters()),
            lr=1e-3
        )
        
    def train_epoch(self):
        self.model.train()
        for head in self.task_heads.values():
            head.train()
            
        task_metrics = defaultdict(dict)
        
        # Alternately train different datasets
        for task_name, loaders in self.dataloaders.items():
            if loaders['train'] is None:
                continue
                
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(loaders['train'], desc=f"Training {task_name}"):
                inputs = inputs.to(self.device)  # [B, C, T]
                labels = labels.to(self.device)

                # Add debug print to check shapes
                # print(f"Input shape: {inputs.shape}")  # Debug
                
                self.optimizer.zero_grad()
                features = self.model(inputs)

                # Add debug print for features shape
                # print(f"Features shape: {features.shape}")  # Debug

                outputs = self.task_heads[task_name](features)

                # Add debug print for outputs shape
                # print(f"Outputs shape: {outputs.shape}")  # Debug
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            task_metrics[task_name] = {
                'loss': running_loss / len(loaders['train']),
                'accuracy': 100. * correct / total
            }
        
        return task_metrics
    
    def evaluate(self):
        self.model.eval()
        for head in self.task_heads.values():
            head.eval()
            
        task_metrics = defaultdict(dict)
        
        with torch.no_grad():
            for task_name, loaders in self.dataloaders.items():
                if loaders['test'] is None:
                    continue
                    
                running_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in tqdm(loaders['test'], desc=f"Evaluating {task_name}"):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    features = self.model(inputs)
                    outputs = self.task_heads[task_name](features)
                    
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                task_metrics[task_name] = {
                    'loss': running_loss / len(loaders['test']),
                    'accuracy': 100. * correct / total
                }
        
        return task_metrics
    
    def train(self, epochs=10, save_path='best_model.pth', patience=5, min_delta=0.01):
        """
        Train model and save best weights, supports early stopping.

        Args:
            epochs: Maximum number of training epochs
            save_path: Path to save best model weights
            patience: Tolerance epochs for early stopping
            min_delta: Minimum change in validation metric
        Returns:
            best_avg_acc: Best average validation accuracy
            best_val_metrics: Best validation metrics
            history: Training history
            best_model_state: Best model state dictionary
        """
        best_avg_acc = 0.0
        best_val_metrics = None  # To save validation accuracy corresponding to best weights
        history = []
        best_model_state = None  # To save best model state

        # Early stopping variables
        no_improvement_epochs = 0  # Record consecutive epochs with no improvement in validation metric
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.evaluate()
            
            # Calculate average accuracy
            avg_acc = np.mean([m['accuracy'] for m in val_metrics.values()])
            # print("avg_acc:", avg_acc)
            # Save history
            history.append({
                'train': train_metrics,
                'val': val_metrics,
                'avg_acc': avg_acc
            })
            
            print(f"\nValidation Accuracy:")
            for task, metrics in val_metrics.items():
                print(f"{task}: {metrics['accuracy']:.2f}%")
            print(f"Average Accuracy: {avg_acc:.2f}%")
            
            # # Update best model
            # if avg_acc > best_avg_acc:
            #     best_avg_acc = avg_acc
            #     best_val_metrics = val_metrics  # Save metrics for each task corresponding to best val accuracy
            #     best_model_state = {
            #         'model': self.model.state_dict(),
            #         'heads': self.task_heads.state_dict()
            #     }
            
            # # Save best model weights to file
            # torch.save(best_model_state, save_path)
            # print(f"✅ Best model saved with accuracy: {best_avg_acc:.2f}%")
            
            # Check if it is the best model
            if avg_acc > best_avg_acc + min_delta:  # Significant improvement in validation metric
                best_avg_acc = avg_acc
                best_val_metrics = val_metrics  # Save metrics for each task corresponding to best val accuracy
                best_model_state = {
                    'model': self.model.state_dict(),
                    'heads': self.task_heads.state_dict()
                }
                
                # Save best model weights to file
                torch.save(best_model_state, save_path)
                print(f"✅ Best model saved with accuracy: {best_avg_acc:.2f}%")

                no_improvement_epochs = 0  # Reset early stopping counter
            else:
                no_improvement_epochs += 1  # No improvement in validation metric, increment counter
                print(f"⚠️ No improvement for {no_improvement_epochs} epoch(s).")

            # Check if early stopping is triggered
            if no_improvement_epochs >= patience:
                print(f"⏹️ Early stopping triggered. No improvement for {patience} consecutive epochs.")
                break
        
        return best_avg_acc, best_val_metrics, history, best_model_state