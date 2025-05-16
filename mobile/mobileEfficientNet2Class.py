import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

original_model = models.efficientnet_v2_s(pretrained=True)
original_model.eval()

if torch.cuda.is_available():
    original_model.to('cuda')

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class EfficientNetConv0(nn.Module):
    def __init__(self):
        super(EfficientNetConv0, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[0:1])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetConv1(nn.Module):
    def __init__(self):
        super(EfficientNetConv1, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[0:2])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetConv2(nn.Module):
    def __init__(self):
        super(EfficientNetConv2, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[0:3])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetConv3(nn.Module):
    def __init__(self):
        super(EfficientNetConv3, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[0:4])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetConv4(nn.Module):
    def __init__(self):
        super(EfficientNetConv4, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[0:5])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetConv5(nn.Module):
    def __init__(self):
        super(EfficientNetConv5, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[0:6])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetConv6(nn.Module):
    def __init__(self):
        super(EfficientNetConv6, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[0:7])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetConv7(nn.Module):
    def __init__(self):
        super(EfficientNetConv7, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[0:8])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetPool(nn.Module):
    def __init__(self):
        super(EfficientNetPool, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children()) + [nn.AdaptiveAvgPool2d(1), Flatten()]) 
    def forward(self, x):
        x = self.features(x)
        return x

class EfficientNetClassfier0(nn.Module):
    def __init__(self):
        super(EfficientNetClassfier0, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children()) + [nn.AdaptiveAvgPool2d(1), Flatten()] + list(original_model.classifier.children())[0:1])
    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetAll(nn.Module):
    def __init__(self):
        super(EfficientNetAll, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children()) + [nn.AdaptiveAvgPool2d(1), Flatten()] + list(original_model.classifier.children()))
    def forward(self, x):
        x = self.features(x)
        return x

    
model0 = EfficientNetConv0()
model1 = EfficientNetConv2()
model2 = EfficientNetConv3()
model3 = EfficientNetConv4()
model4 = EfficientNetConv5()
model5 = EfficientNetConv6()
model6 = EfficientNetConv7()
model7 = EfficientNetPool()
model8 = EfficientNetAll()


def classify(i):
    if i == 1:
        return model0
    elif i == 2: 
        return model1
    elif i == 3:
        return model2
    elif i == 4:
        return model3
    elif i == 5:
        return model4 
    elif i == 6:
        return model5
    elif i == 7:
        return model6
    elif i == 8:
        return model7
    elif i == 9:
        return model8




