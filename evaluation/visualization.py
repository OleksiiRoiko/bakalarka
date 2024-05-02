# evaluation/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import csv
import re


def store_model_evaluation(model_name, evaluation_output, file_path):
    # Extract default accuracy and ECE
    default_accuracy_match = re.search(r'Default Accuracy: ([0-9.]+)%', evaluation_output)
    ece_match = re.search(r'Expected Calibration Error \(ECE\): ([0-9.]+)', evaluation_output)
    ace_match = re.search(r'Adaptive Calibration Error \(ACE\): ([0-9.]+)', evaluation_output)
    optimal_prah_match = re.search(r'Optimal Prah \(PRAH\): ([0-9.]+)', evaluation_output)


    default_accuracy = default_accuracy_match.group(1) if default_accuracy_match else 'N/A'
    ece = ece_match.group(1) if ece_match else 'N/A'
    ace = ace_match.group(1) if ace_match else 'N/A'
    optimal_prah = optimal_prah_match.group(1) if optimal_prah_match else 'N/A'

    # Use a refined regular expression to extract PRAH-related information
    prah_pattern = r'PRAH: ([0-9.]+)\s+Accuracy: ([0-9.]+)%\s+Classified: (\d+), Correct Classified: (\d+), Incorrect Classified: (\d+)\s+Not Classified: (\d+), Correct Not Classified: (\d+), Incorrect Not Classified: (\d+)'
    prah_matches = re.findall(prah_pattern, evaluation_output)

    with open(file_path, mode='a', newline='') as file:
        fieldnames = ["Model Name", "Default Accuracy", "ECE", "ACE", "Optim. PRAH", "PRAH", "PRAH Accuracy", "Classified", "Correct Classified", "Incorrect Classified", "Not Classified", "Correct Not Classified", "Incorrect Not Classified"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Check if file is empty to write headers
        if file.tell() == 0:
            writer.writeheader()

        # Write a summary line for the model
        summary_data = {
            "Model Name": model_name,
            "Default Accuracy": default_accuracy,
            "ECE": ece,
            "ACE": ace,
            "Optim. PRAH": optimal_prah
        }
        writer.writerow(summary_data)

        # Write detailed PRAH entries
        for prah, prah_acc, classified, correct_classified, incorrect_classified, not_classified, correct_not_classified, incorrect_not_classified in prah_matches:
            prah_data = {
                "PRAH": prah,
                "PRAH Accuracy": prah_acc,
                "Classified": classified,
                "Correct Classified": correct_classified,
                "Incorrect Classified": incorrect_classified,
                "Not Classified": not_classified,
                "Correct Not Classified": correct_not_classified,
                "Incorrect Not Classified": incorrect_not_classified
            }
            writer.writerow(prah_data)

def plot_histograms(diff_correct, diff_incorrect, diff_all):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].hist(diff_correct, bins=30, alpha=0.5, color='green')
    axs[0].set_title('Correctly Classified')
    axs[0].set_xlabel('Diff in Top Two Probabilities')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(diff_incorrect, bins=30, alpha=0.5, color='red')
    axs[1].set_title('Incorrectly Classified')
    axs[1].set_xlabel('Diff in Top Two Probabilities')

    axs[2].hist(diff_all, bins=30, alpha=0.5, color='blue')
    axs[2].set_title('All Images')
    axs[2].set_xlabel('Diff in Top Two Probabilities')

    plt.tight_layout()
    plt.show()

def plot_prah_results(prah_values, classified_counts, accuracies):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # "Classified Examples vs PRAH" on the left
    ax1.plot(prah_values, classified_counts, 'o-', color='blue')
    ax1.set_title('Classified Examples vs PRAH')
    ax1.set_xlabel('PRAH Value')
    ax1.set_ylabel('Number of Classified Examples')

    # "Accuracy vs PRAH" on the right
    ax2.plot(prah_values, accuracies, 'o-', color='red')
    ax2.set_title('Accuracy vs PRAH')
    ax2.set_xlabel('PRAH Value')
    ax2.set_ylabel('Accuracy (%)')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()
