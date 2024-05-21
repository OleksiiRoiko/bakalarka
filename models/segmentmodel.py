import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
from torchinfo import summary
NUM_CLASSES = 10

def create_tiny_resnet():
    weights = ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)

    # Modify the initial convolution layer
    resnet.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

    # Extract layers except the last two layers
    resnet_layers = list(resnet.children())[:-2]

    # Replace the fourth layer with an upsampling layer
    resnet_layers[3] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    # Add a final classifier layer
    classifier = nn.Conv2d(512, NUM_CLASSES, 1)
    torch.nn.init.kaiming_normal_(classifier.weight)

    # Append classifier and another upsampling layer
    resnet_layers.append(classifier)
    resnet_layers.append(nn.Upsample(size=32, mode='bilinear', align_corners=False))

    tiny_resnet = nn.Sequential(*resnet_layers)
    return nn.DataParallel(tiny_resnet).cuda()

model = create_tiny_resnet()

x = [1,3,32,32]

summary(model,x)