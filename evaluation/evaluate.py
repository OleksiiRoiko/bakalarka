import numpy as np
import torch

from models import ResNet, ResidualBlock
from training import load_model
from torch.utils.data import DataLoader
from data.dataset_factory import DatasetFactory
from evaluation import plot_histograms, plot_prah_results,store_model_evaluation
from metrics import (evaluate_model,
                     evaluate_model_with_data_collection,
                     collect_probabilities_and_labels,
                     compute_ece,
                     compute_ace,
                     find_optimal_prah)
import io
import sys
import os

def get_num_classes_from_filename(filename):
    if ('CIFAR10WithBackground' in filename or
        'CIFAR10WithBackgroundSoftLabel' in filename or
        'CIFAR10WithDistributedSoftBackgroundLabels' in filename):
        return 11
    elif ('CIFAR10WithClassSpecificBackground' in filename or
          'CIFAR10WithClassSpecificBackgroundSoftLabel' in filename or
          'CIFAR10WithDistributedClassSpecificSoftBackgroundLabels' in filename):
        return 20
    elif ('DefaultCIFAR10' in filename or
          'SegmentedCIFAR10' in filename):
        return 10
    else:
        raise ValueError(f"Unknown model type in filename: {filename}")

def load_and_evaluate(model_dir,specific_model_filename):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    configurations = {'dataset_class': "DefaultCIFAR10", 'model': None, 'mode': None, 'fill_background': None, 'crop_size': None, 'batch_size': 32}


    test_dataset = DatasetFactory.create_dataset(
        root='./data',
        name=configurations['dataset_class'],
        train=False,
        download=True,
        model=configurations.get('model'),
        mode=configurations.get('mode'),
        fill_background=configurations.get('fill_background'),
        crop_size=configurations.get('crop_size')
    )
    test_loader = DataLoader(test_dataset, batch_size=configurations['batch_size'], shuffle=False)

    if specific_model_filename:
        model_files = [specific_model_filename]
    else:
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    for model_file in model_files:
        num_classes = get_num_classes_from_filename(model_file)

        model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)
        model.to(device)

        model_path = os.path.join(model_dir, model_file)
        model = load_model(model, model_path, device)

        old_stdout = sys.stdout
        sys.stdout = new_stdout = io.StringIO()


        _, default_accuracy = evaluate_model(model, test_loader, device,10)
        print(f'Default Accuracy: {default_accuracy}%')

        prah_values = np.linspace(0, 0.9, 10)
        classified_counts = []
        accuracies = []
        results_data = []

        for prah in prah_values:
            result = evaluate_model_with_data_collection(model, test_loader, device,10,prah)
            results_data.append(result)
            classified_counts.append(result["total_classified"])
            accuracies.append(result["accuracy"])

            print(f'PRAH: {prah:.2f}')
            print(f'Accuracy: {result["accuracy"]:.2f}%')
            print(f'Classified: {result["total_classified"]}, Correct Classified: {result["correct_classified"]}, Incorrect Classified: {result["incorrect_classified"]}')
            print(f'Not Classified: {result["not_classified"]}, Correct Not Classified: {result["correct_not_classified"]}, Incorrect Not Classified: {result["incorrect_not_classified"]}')

        if prah_values[0] == 0:
            zero_prah_data = results_data[0]
            plot_histograms(zero_prah_data['diff_correct'], zero_prah_data['diff_incorrect'], zero_prah_data['diff_all'])

        plot_prah_results(prah_values, classified_counts, accuracies)

        probabilities, true_labels = collect_probabilities_and_labels(model, test_loader, device)
        ece = compute_ece(probabilities, true_labels)
        print(f'Expected Calibration Error (ECE): {ece}')

        ace = compute_ace(probabilities, true_labels)
        print(f'Adaptive Calibration Error (ACE): {ace}')

        PRAH = find_optimal_prah(results_data)
        print(f'Optimal Prah (PRAH): {PRAH:.4f}')

        sys.stdout = old_stdout

        evaluation_results = new_stdout.getvalue()

        model_name = os.path.basename(model_path)
        store_model_evaluation(model_name, evaluation_results, '../csv/new.csv')



if __name__ == "__main__":
    model_dir = '../training/sgd(lr0,01-mo0.9-wd0.001-nest)steplr(ss10-gamma0.1)-batch32-7ep-BCElogit-5time'
    specific_model = None
    load_and_evaluate(model_dir, specific_model_filename=specific_model)

