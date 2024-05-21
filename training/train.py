import os
from torch.utils.data import DataLoader

from models import ResNet, ResidualBlock, create_tiny_resnet
from training import save_model, plot_learning_curves
from data.dataset_factory import DatasetFactory
from evaluation.metrics import evaluate_model

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR


def train_model(model, train_loader, val_loader, device, num_epochs, num_classes):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001, nesterov=True)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = BCEWithLogitsLoss()

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            if labels.dtype == torch.int64:
                labels = F.one_hot(labels, num_classes=num_classes).float()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            preds = outputs.sigmoid().max(1)[1]
            correct += preds.eq(labels.max(1)[1]).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        val_loss, val_accuracy = evaluate_model(model, val_loader, device, num_classes)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        # Print training and validation resultsm
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, '
              f'Train Acc: {epoch_train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    return train_losses, train_accuracies, val_losses, val_accuracies


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    segmentation_model = create_tiny_resnet()
    segmentation_model_path = '../segment.pth'
    segmentation_model.load_state_dict(torch.load(segmentation_model_path))
    segmentation_model.eval()
    segmentation_model.to(device)

    configurations = [

        {'num_epochs': 7, 'num_classes': 10,'dataset_name': 'SegmentedCIFAR10WithObject','model': segmentation_model, 'mode': 4, 'fill_background': True,'crop_size': None, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackground','model': segmentation_model, 'mode': 4, 'fill_background': False,'crop_size': None, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackgroundSoftLabel','model': segmentation_model, 'mode': 4, 'fill_background': False,'crop_size': None, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels','model': segmentation_model, 'mode': 4, 'fill_background': False,'crop_size': None, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackground','model': segmentation_model, 'mode': 4, 'fill_background': False,'crop_size': None, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel','model': segmentation_model, 'mode': 4, 'fill_background': False,'crop_size': None, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels','model': segmentation_model, 'mode': 4, 'fill_background': False,'crop_size': None, 'batch_size': 32 },

    ]


    for config in configurations:
        train_dataset = DatasetFactory.create_dataset(
            root='./data',
            name=config['dataset_name'],
            train=True,
            download=True,
            model=config.get('model'),
            mode=config.get('mode'),
            fill_background=config.get('fill_background'),
            crop_size=config.get('crop_size')
        )
        val_dataset = DatasetFactory.create_dataset(
            root='./data',
            name=config['dataset_name'],
            train=False,
            download=True,
            model=config.get('model'),
            mode=config.get('mode'),
            fill_background=config.get('fill_background'),
            crop_size=config.get('crop_size')
        )

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=config['num_classes'])
        model.to(device)

        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, device,
            config['num_epochs'], config['num_classes']
        )

        plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies)

        save_directory = os.path.join(os.path.dirname(__file__), 'sgd(lr0,01-mo0.9-wd0.001-nest)steplr(ss10-gamma0.1)-batch32-7ep-BCElogit-5time')
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, f"{config['dataset_name']}-mode{config['mode']}-cr.size{config['crop_size']}.pth")
        save_model(model, model_path)
        print(f"Model trained and saved: {model_path}")

if __name__ == "__main__":
    main()


"""
        {'num_epochs': 7, 'num_classes': 10,'dataset_name': 'DefaultCIFAR10','model': None, 'mode': None, 'fill_background': None,'crop_size': None, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackground','model': None, 'mode': 1, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackground','model': None, 'mode': 1, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackground','model': None, 'mode': 2, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackground','model': None, 'mode': 2, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackground','model': None, 'mode': 3, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackground','model': None, 'mode': 3, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackgroundSoftLabel','model': None, 'mode': 1, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackgroundSoftLabel','model': None, 'mode': 1, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackgroundSoftLabel','model': None, 'mode': 2, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackgroundSoftLabel','model': None, 'mode': 2, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackgroundSoftLabel','model': None, 'mode': 3, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithBackgroundSoftLabel','model': None, 'mode': 3, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels','model': None, 'mode': 1, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels','model': None, 'mode': 1, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels','model': None, 'mode': 2, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels','model': None, 'mode': 2, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels','model': None, 'mode': 3, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 11,'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels','model': None, 'mode': 3, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackground','model': None, 'mode': 1, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackground','model': None, 'mode': 1, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackground','model': None, 'mode': 2, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackground','model': None, 'mode': 2, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackground','model': None, 'mode': 3, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackground','model': None, 'mode': 3, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel','model': None, 'mode': 1, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel','model': None, 'mode': 1, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel','model': None, 'mode': 2, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel','model': None, 'mode': 2, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel','model': None, 'mode': 3, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel','model': None, 'mode': 3, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },

        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels','model': None, 'mode': 1, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels','model': None, 'mode': 1, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels','model': None, 'mode': 2, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels','model': None, 'mode': 2, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels','model': None, 'mode': 3, 'fill_background': None,'crop_size': 4, 'batch_size': 32 },
        {'num_epochs': 7, 'num_classes': 20,'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels','model': None, 'mode': 3, 'fill_background': None,'crop_size': 8, 'batch_size': 32 },
        
"""
