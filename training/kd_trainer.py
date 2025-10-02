import torch
import torch.nn as nn
from training.trainer import BaseTrainer

class KDTrainer(BaseTrainer):
    def __init__(self, student, teacher, train_loader, val_loader, config):
        super().__init__(student, train_loader, val_loader, config)
        self.teacher = teacher
        self.teacher.eval()
        self.temperature = config['temperature']
        self.alpha = config['alpha']
        
    def compute_loss(self, data, target):
        # 教师模型预测
        with torch.no_grad():
            teacher_logits = self.teacher(data)
        
        # 学生模型预测
        student_logits = self.model(data)
        
        # 计算蒸馏损失
        soft_teacher = torch.softmax(teacher_logits/self.temperature, dim=1)
        soft_student = torch.log_softmax(student_logits/self.temperature, dim=1)
        kd_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher)
        
        # 计算交叉熵损失
        ce_loss = nn.CrossEntropyLoss()(student_logits, target)
        
        # 组合损失
        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return loss