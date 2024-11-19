#
import torch
import torch.nn as nn


class CNNModel(nn.Module):
    
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # لایه پیچشی 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # کاهش ابعاد
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # لایه پیچشی 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # لایه پیچشی 3
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # تطبیق با ابعاد خروجی لایه‌های پیچشی
            nn.ReLU(),
            nn.Linear(128, 5),  # 5 کلاس برای دسته‌بندی
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
    


class ResidualCNN(nn.Module):
    
    def __init__(self):
        super(ResidualCNN, self).__init__()
        self.layer1 = ResidualBlock(3, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 64, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 5)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    


class InceptionBlock(nn.Module):
    
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        # مسیر 1x1
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        # مسیر 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1)
        )
        # مسیر 5x5
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        # مسیر Max Pooling
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 24, kernel_size=1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = torch.cat([branch1, branch3, branch5, branch_pool], dim=1)
        return outputs
    


class InceptionCNN(nn.Module):
    
    def __init__(self):
        super(InceptionCNN, self).__init__()
        self.layer1 = InceptionBlock(3)
        self.layer2 = InceptionBlock(88)  # 16+24+24+24=88
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(88, 5)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    

#cloner174