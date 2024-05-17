import os
import optuna
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

# Function to create data loaders
def create_dataloaders(config, batch_size):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Function to train the model
def train_model(model, train_loader, val_loader, device, num_epochs, num_classes, lr, weight_decay, step_size, gamma):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = BCEWithLogitsLoss()

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
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

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, '
              f'Train Acc: {epoch_train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    return train_losses, train_accuracies, val_losses, val_accuracies

# Function to set up the environment (device and model)
def setup_environment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    segmentation_model = create_tiny_resnet()
    segmentation_model_path = '../segment.pth'
    segmentation_model.load_state_dict(torch.load(segmentation_model_path))
    segmentation_model.eval()
    segmentation_model.to(device)

    return device, segmentation_model

# Optuna objective function for hyperparameter optimization
def objective(trial):
    device, segmentation_model = setup_environment()

    config = {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackground', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 4}


    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    train_loader, val_loader = create_dataloaders(config, batch_size)

    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=config['num_classes'])
    model.to(device)

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    step_size = trial.suggest_int('step_size', 1, 10)
    gamma = trial.suggest_float('gamma', 0.1, 0.9)

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, device,
        config['num_epochs'], config['num_classes'], lr, weight_decay, step_size, gamma
    )

    return max(val_losses)

# Main function to run the entire process
def main():
    # Step 1: Optimize hyperparameters with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)

    best_params = study.best_params

    # Step 2: Train and save 44 models with the best hyperparameters
    device, segmentation_model = setup_environment()

    configurations = [
        {'num_epochs': 5, 'num_classes': 10, 'dataset_name': 'SegmentedCIFAR10WithObject', 'model': segmentation_model, 'mode': 4, 'fill_background': True, 'crop_size': None},
        {'num_epochs': 5, 'num_classes': 10, 'dataset_name': 'DefaultCIFAR10', 'model': None, 'mode': None, 'fill_background': None, 'crop_size': None},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackground', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackground', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackground', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackground', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackground', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackground', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackground', 'model': segmentation_model, 'mode': 4, 'fill_background': False, 'crop_size': None},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackgroundSoftLabel', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackgroundSoftLabel', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackgroundSoftLabel', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackgroundSoftLabel', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackgroundSoftLabel', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackgroundSoftLabel', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithBackgroundSoftLabel', 'model': segmentation_model, 'mode': 4, 'fill_background': False, 'crop_size': None},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 11, 'dataset_name': 'CIFAR10WithDistributedSoftBackgroundLabels', 'model': segmentation_model, 'mode': 4, 'fill_background': False, 'crop_size': None},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackground', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackground', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackground', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackground', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackground', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackground', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackground', 'model': segmentation_model, 'mode': 4, 'fill_background': False, 'crop_size': None},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithClassSpecificBackgroundSoftLabel', 'model': segmentation_model, 'mode': 4, 'fill_background': False, 'crop_size': None},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels', 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels', 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 4},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels', 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 8},
        {'num_epochs': 5, 'num_classes': 20, 'dataset_name': 'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels', 'model': segmentation_model, 'mode': 4, 'fill_background': False, 'crop_size': None},
    ]

    for config in configurations:
        batch_size = best_params['batch_size']
        train_loader, val_loader = create_dataloaders(config, batch_size)

        model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=config['num_classes'])
        model.to(device)

        lr = best_params['lr']
        weight_decay = best_params['weight_decay']
        step_size = best_params['step_size']
        gamma = best_params['gamma']

        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, device,
            config['num_epochs'], config['num_classes'], lr, weight_decay, step_size, gamma
        )

        plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies)

        save_directory = os.path.join(os.path.dirname(__file__), f'adam(lr{lr})steplr(ss{step_size}-gamma{gamma})-batch{batch_size}-5ep-BCElogit-1time')
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, f"{config['dataset_name']}-mode{config['mode']}-cr.size{config['crop_size']}.pth")
        save_model(model, model_path)
        print(f"Model trained and saved: {model_path}")

if __name__ == "__main__":
    main()
