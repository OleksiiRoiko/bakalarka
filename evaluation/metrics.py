import numpy as np
from scipy.interpolate import interp1d
from torch.nn import BCEWithLogitsLoss
import torch
from torch.nn.functional import one_hot
def evaluate_model(model, loader, device, num_classes):
    model.eval()
    criterion = BCEWithLogitsLoss()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs[:,:num_classes]

            if labels.dtype == torch.int64:
                labels = one_hot(labels, num_classes=num_classes).float()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.sigmoid().max(1)[1]
            correct += preds.eq(labels.max(1)[1]).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def evaluate_model_with_data_collection(model, testloader, device, num_classes, prah_threshold=0.0):
    model.eval()
    total_images = len(testloader.dataset)
    total_classified, correct_classified, incorrect_classified = 0, 0, 0
    not_classified, correct_not_classified, incorrect_not_classified = 0, 0, 0
    diff_correct, diff_incorrect, diff_all = [], [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs[:,:num_classes]

            if labels.dtype == torch.int64:
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

            probabilities = torch.sigmoid(outputs)
            top_two_probs = probabilities.topk(2, dim=1).values
            diff = top_two_probs[:, 0] - top_two_probs[:, 1]
            diff_all.extend(diff.cpu().numpy())

            _, predicted = probabilities.max(1)
            _, labels_max = labels.max(1) if labels.ndim > 1 else (None, labels)
            classified_mask = diff > prah_threshold
            total_classified += classified_mask.sum().item()
            not_classified += (~classified_mask).sum().item()

            correct_mask = predicted == labels_max
            correct_classified += (correct_mask & classified_mask).sum().item()
            incorrect_classified += (~correct_mask & classified_mask).sum().item()
            correct_not_classified += (correct_mask & ~classified_mask).sum().item()
            incorrect_not_classified += (~correct_mask & ~classified_mask).sum().item()

            if correct_mask.any():
                diff_correct.extend(diff[correct_mask & classified_mask].cpu().numpy())
            if (~correct_mask).any():
                diff_incorrect.extend(diff[~correct_mask & classified_mask].cpu().numpy())

    accuracy = 100.0 * correct_classified / total_classified if total_classified > 0 else 0

    result = {
        "prah": prah_threshold,
        "total_images": total_images,
        "total_classified": total_classified,
        "correct_classified": correct_classified,
        "incorrect_classified": incorrect_classified,
        "not_classified": not_classified,
        "correct_not_classified": correct_not_classified,
        "incorrect_not_classified": incorrect_not_classified,
        "accuracy": accuracy,
        "diff_correct": diff_correct,
        "diff_incorrect": diff_incorrect,
        "diff_all": diff_all
    }

    return result

def collect_probabilities_and_labels(model, testloader, device):
    model.eval()
    probabilities = []
    true_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs[:,:10]

            if labels.dtype == torch.int64:
                labels = torch.nn.functional.one_hot(labels, num_classes=10).float()

            probs = torch.sigmoid(outputs)

            probabilities.append(probs.cpu())
            true_labels.append(labels.cpu())

    probabilities = torch.cat(probabilities, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    return probabilities, true_labels

def compute_ece(probabilities, true_labels, n_bins=15):
    bin_bounds = torch.linspace(0, 1, n_bins + 1)
    ece = torch.tensor(0.0)
    max_probs, predicted_indices = probabilities.max(dim=1)
    true_indices = true_labels.max(dim=1)[1]  # Индекс максимального значения в мягких метках
    bin_indices = torch.bucketize(max_probs, bin_bounds) - 1

    for i in range(n_bins):
        in_bin = bin_indices == i
        if in_bin.any():
            true_labels_in_bin = true_indices[in_bin]
            predicted_in_bin = predicted_indices[in_bin]
            bin_accuracy = (predicted_in_bin == true_labels_in_bin).float().mean()
            bin_confidence = max_probs[in_bin].mean()
            bin_weight = in_bin.float().mean().clamp(min=1e-8)
            ece += torch.abs(bin_confidence - bin_accuracy) * bin_weight
    return ece


def compute_ace(probabilities, true_labels, n_bins=15):
    ace = torch.tensor(0.0)
    max_probs, predicted_indices = probabilities.max(dim=1)
    true_indices = true_labels.max(dim=1)[1]  # Получение индексов максимальных значений из мягких меток
    quantiles = torch.linspace(0, 1, n_bins + 1)
    bin_edges = torch.quantile(max_probs, quantiles)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        in_bin = (max_probs >= bin_lower) & (max_probs < bin_upper)
        if in_bin.any():
            true_labels_in_bin = true_indices[in_bin]
            predicted_in_bin = predicted_indices[in_bin]
            bin_accuracy = (predicted_in_bin == true_labels_in_bin).float().mean()
            bin_confidence = max_probs[in_bin].mean()
            bin_weight = in_bin.float().mean().clamp(min=1e-8)
            ace += torch.abs(bin_confidence - bin_accuracy) * bin_weight

    return ace


def calculate_score(accuracy, total_classified, total_images, alpha=0.5, beta=0.5):
    return (alpha * accuracy) + (beta * (total_classified / total_images * 100))

def find_optimal_prah(results):
    prah_values = np.array([data['prah'] for data in results])
    accuracies = np.array([data['accuracy'] for data in results])
    total_classifieds = np.array([data['correct_classified'] for data in results])
    total_images = results[0]['total_images']

    min_classified_percentage = 0.2

    accuracy_interp = interp1d(prah_values, accuracies, kind='cubic', fill_value='extrapolate')
    classified_interp = interp1d(prah_values, total_classifieds, kind='cubic', fill_value='extrapolate')

    fine_prah_values = np.linspace(min(prah_values), max(prah_values), 10000)

    scores = []
    for prah in fine_prah_values:
        interpolated_classified = classified_interp(prah)
        if interpolated_classified / total_images >= min_classified_percentage:
            score = calculate_score(accuracy_interp(prah), interpolated_classified, total_images, alpha=0.5, beta=0.5)
        else:
            score = -np.inf
        scores.append(score)

    scores = np.array(scores)
    best_index = np.argmax(scores)
    best_prah = fine_prah_values[best_index]
    return best_prah
