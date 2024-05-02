import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

# Constants and configurations
NUM_CLASSES = 10
BATCH_SIZE = 100
NUM_WORKERS = 8

def create_tiny_resnet():
    weights = ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    resnet.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
    resnet_ = list(resnet.children())[:-2]
    resnet_[3] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    classifier = nn.Conv2d(512, NUM_CLASSES, 1)
    torch.nn.init.kaiming_normal_(classifier.weight)
    resnet_.append(classifier)
    resnet_.append(nn.Upsample(size=32, mode='bilinear', align_corners=False))
    tiny_resnet = nn.Sequential(*resnet_)
    return nn.DataParallel(tiny_resnet).cuda()

def attention(x):
    return torch.sigmoid(torch.logsumexp(x, 1, keepdim=True))
