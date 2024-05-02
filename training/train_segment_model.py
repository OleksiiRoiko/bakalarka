from torch import optim
from models import create_tiny_resnet, BATCH_SIZE, NUM_WORKERS
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Constants and configurations
NUM_EPOCHS = 10
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_dataloader(train=True):
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=train, num_workers=NUM_WORKERS)

def train_model(model, loader, criterion, optimizer, scheduler):
    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f'Starting Epoch {epoch + 1}/{NUM_EPOCHS}')
        for x, _label in loader:
            x, _label = x.cuda(), _label.cuda()
            label = F.one_hot(_label, NUM_CLASSES).float()
            seg_out = model(x)
            logit = torch.log(torch.exp(seg_out * 0.5).mean((-2, -1))) * 2
            loss = criterion(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

if __name__ == '__main__':
    model = create_tiny_resnet()
    train_loader = get_dataloader(train=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 78, 0.001)
    train_model(model, train_loader, criterion, optimizer, scheduler)
