import torch
import torch.nn as nn
from torchvision import models

class GlobalHintNet(nn.Module):
    def __init__(self):
        super(GlobalHintNet, self).__init__()
        
        # Завантажуємо ResNet18, навчену на ImageNet
        # weights='DEFAULT' завантажує найсучасніші ваги
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Нам потрібна вся мережа КРІМ останнього шару (який каже "це кіт" чи "це пес")
        # Ми хочемо отримати вектор ознак розміром 512
        # list(resnet.children())[:-1] бере всі шари до Average Pooling включно
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Заморожуємо ваги! Ми не хочемо псувати ResNet, він вже розумний.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x має розмір (Batch, 1, 256, 256) - це наш канал L
        # ResNet очікує 3 канали (RGB). Тому ми дублюємо L тричі.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # (Batch, 3, 256, 256)
        
        features = self.backbone(x) 
        # Вихід ResNet: (Batch, 512, 1, 1)
        
        # Перетворюємо в плоский вектор (Batch, 512)
        return features.view(features.size(0), -1)