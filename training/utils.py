import torch
import matplotlib.pyplot as plt
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Strata tréningu')
    plt.plot(epochs, val_losses, 'r+-', label='Strata pri overovaní')
    plt.title('Straty pri tréningu a overovaní')
    plt.xlabel('Epochy')
    plt.ylabel('Straty')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Presnosť tréningu')
    plt.plot(epochs, val_accuracies, 'r+-', label='Presnosť overovania')
    plt.title('Presnosť tréningu a validácie')
    plt.xlabel('Epochy')
    plt.ylabel('Presnosť')
    plt.legend()

    plt.tight_layout()
    plt.show()