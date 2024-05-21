# evaluation/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import csv
import re
import pandas as pd
import seaborn as sns
from tabulate import tabulate

plt.rcParams.update({'font.size': 14})
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
    axs[0].set_title('Správne klasifikované')
    axs[0].set_xlabel('Rozdiel v dvoch max pravdepod.')
    axs[0].set_ylabel('Frekvencia')

    axs[1].hist(diff_incorrect, bins=30, alpha=0.5, color='red')
    axs[1].set_title('Nesprávne klasifikované')
    axs[1].set_xlabel('Rozdiel v dvoch max pravdepod.')

    axs[2].hist(diff_all, bins=30, alpha=0.5, color='blue')
    axs[2].set_title('Všetky obrázky')
    axs[2].set_xlabel('Rozdiel v dvoch max pravdepod.')

    plt.tight_layout()
    plt.show()

def plot_prah_results(prah_values, classified_counts, accuracies):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # "Classified Examples vs PRAH" on the left
    ax1.plot(prah_values, classified_counts, 'o-', color='blue')
    ax1.set_title('Klasifikované príklady vs PRAH')
    ax1.set_xlabel('Hodnota PRAH')
    ax1.set_ylabel('Počet klasifikovaných príkladov')

    # "Accuracy vs PRAH" on the right
    ax2.plot(prah_values, accuracies, 'o-', color='red')
    ax2.set_title('Presnosť vs PRAH')
    ax2.set_xlabel('Hodnota PRAH')
    ax2.set_ylabel('Presnosť (%)')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()

def calculate_statistics(input_file_path, output_file_path):
    data = pd.read_csv(input_file_path)

    filtered_data = data[['Model Name', 'Default Accuracy', 'ECE', 'ACE', 'Optim. PRAH']].dropna(subset=['Model Name'])

    stats = filtered_data.groupby('Model Name').agg({
        'Default Accuracy': ['mean', 'std'],
        'ECE': ['mean', 'std'],
        'ACE': ['mean', 'std'],
        'Optim. PRAH': ['mean', 'std']
    }).reset_index()

    stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in stats.columns.values]

    stats = stats.round(4)

    stats.to_csv(output_file_path, index=False)

def aggregate_model_data_and_save(file_path, output_file_path):
    data = pd.read_csv(file_path)

    data['Model Name'].fillna(method='ffill', inplace=True)

    grouped_data = data.groupby(['Model Name','PRAH']).agg({
        'PRAH Accuracy': 'mean',
        'Classified': 'mean',
        'Correct Classified': 'mean',
        'Incorrect Classified': 'mean',
        'Not Classified': 'mean',
        'Correct Not Classified': 'mean',
        'Incorrect Not Classified': 'mean'
    }).reset_index()

    grouped_data = grouped_data.round(2)

    grouped_data.to_csv(output_file_path, index=False)

def create_plots(data):
    plt.figure(figsize=(15, 5))

    # Plot 1: Presnosť vs Hodnota PRAH
    plt.subplot(1, 3, 1)
    sns.lineplot(x='PRAH', y='PRAH Accuracy', data=data, marker='o')
    plt.title('Presnosť vs Hodnota PRAH')
    plt.xlabel('Hodnota PRAH')
    plt.ylabel('Presnosť (%)')

    # Plot 2: Správne klas. vs. Nesprávne klas.
    melted_data_1 = data.melt(id_vars=['PRAH'], value_vars=['Correct Classified', 'Incorrect Classified'], var_name='Kategória', value_name='Počet')
    melted_data_1['Kategória'] = melted_data_1['Kategória'].replace({
        'Correct Classified': 'Správne klasifikované',
        'Incorrect Classified': 'Nesprávne klasifikované'
    })
    plt.subplot(1, 3, 2)
    sns.barplot(x='PRAH', y='Počet', hue='Kategória', data=melted_data_1)
    plt.title('Správne klas. vs. Nesprávne klas.')
    plt.xlabel('Hodnota PRAH')
    plt.ylabel('Počet')

    # Plot 3: Klasifikované vs. Neklasifikované
    melted_data_2 = data.melt(id_vars=['PRAH'], value_vars=['Classified', 'Not Classified'], var_name='Kategória', value_name='Počet')
    melted_data_2['Kategória'] = melted_data_2['Kategória'].replace({
        'Classified': 'Klasifikované',
        'Not Classified': 'Neklasifikované'
    })
    plt.subplot(1, 3, 3)
    sns.barplot(x='PRAH', y='Počet', hue='Kategória', data=melted_data_2)
    plt.title('Klasifikované vs. Neklasifikované')
    plt.xlabel('Hodnota PRAH')
    plt.ylabel('Počet')

    plt.tight_layout()
    plt.show()

def split_csv(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Select and save ECE results
    ece_df = df[['Model Name', 'ECE_mean', 'ECE_std']]
    ece_df.to_csv('../csv/ECE_results.csv', index=False)

    # Select and save ACE results
    ace_df = df[['Model Name', 'ACE_mean', 'ACE_std']]
    ace_df.to_csv('../csv/ACE_results.csv', index=False)

    # Select and save accuracy results
    accuracy_df = df[['Model Name', 'Default Accuracy_mean', 'Default Accuracy_std']]
    accuracy_df.columns = ['Model Name', 'accuracy_mean', 'accuracy_std']  # Renaming columns for consistency
    accuracy_df.to_csv('../csv/accuracy_results.csv', index=False)

    # Select and save optim.prah results
    optim_prah_df = df[['Model Name', 'Optim. PRAH_mean', 'Optim. PRAH_std']]
    optim_prah_df.columns = ['Model Name', 'optim.prah_mean', 'optim.prah_std']  # Renaming columns for consistency
    optim_prah_df.to_csv('../csv/optim_prah_results.csv', index=False)

def split_csv2(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    accuracy_df = df[['Model Name', 'Default Accuracy_mean', 'Optim. PRAH_mean']]
    accuracy_df.columns = ['Model Name', 'accuracy_mean', 'optim.prah_mean']  # Renaming columns for consistency
    accuracy_df.to_csv('../csv/accuracy-prah_results.csv', index=False)


#calculate_statistics('../csv/new.csv','../csv/new-2method.csv')
#aggregate_model_data_and_save('../csv/new.csv','../csv/new-1method.csv')

#split_csv('../csv/2method.csv') #ECE, ACE, accuraccy
#split_csv2('../csv/2method.csv') #accuray_optimalprah

#file_path = '../csv/1method.csv'
#data = pd.read_csv(file_path)

#data_special_model = data[data['Model Name'] == "CIFAR10WithBackground-mode2-cr.size4.pth"]
#data_standard_model = data[data['Model Name'] == "DefaultCIFAR10-modeNone-cr.sizeNone.pth"]

#create_plots(data_special_model)
#create_plots(data_standard_model)


#accuracy_prah_df = pd.read_csv('../csv/accuracy_results.csv')

#top5_models = accuracy_prah_df.sort_values(by='accuracy_mean', ascending=False).head(5)

#latex_table = top5_models[['Model Name', 'accuracy_mean', 'accuracy_std']].to_latex(index=False,float_format="%.4f")

#print(latex_table)

accuracy_prah_df = pd.read_csv('../csv/accuracy_results.csv')

# Select the default model's data
default_model = accuracy_prah_df[accuracy_prah_df['Model Name'] == 'SegmentedCIFAR10WithObject-mode4-white.pth']
def2model = accuracy_prah_df[accuracy_prah_df['Model Name'] == 'SegmentedCIFAR10WithObject-mode4-cr.sizeNone.pth']

# Combine the default model with the top 5 models
combined_models = pd.concat([default_model, def2model])
combined_models = combined_models.sort_values(by='accuracy_mean', ascending=False)

# Generate the LaTeX table
latex_table = combined_models[['Model Name', 'accuracy_mean', 'accuracy_std']].to_latex(index=False)

print(latex_table)
