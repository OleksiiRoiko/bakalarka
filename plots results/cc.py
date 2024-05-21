import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from data.dataset_factory import DatasetFactory
from models import create_tiny_resnet
from PIL import ImageDraw
from data.cifar10_dataset import CIFAR10Utils  # Import CIFAR10Utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

segmentation_model = create_tiny_resnet()
segmentation_model_path = '../segment.pth'
segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
segmentation_model.eval()
segmentation_model.to(device)

configurations = [
    {'dataset_class': "SegmentedCIFAR10WithObject", 'model': segmentation_model, 'mode': 4, 'fill_background': True, 'crop_size': None},
    {'dataset_class': "CIFAR10WithBackground", 'model': segmentation_model, 'mode': 4, 'fill_background': False, 'crop_size': None},
    {'dataset_class': "CIFAR10WithBackground", 'model': None, 'mode': 1, 'fill_background': None, 'crop_size': 8},
    {'dataset_class': "CIFAR10WithBackground", 'model': None, 'mode': 2, 'fill_background': None, 'crop_size': 8},
    {'dataset_class': "CIFAR10WithBackground", 'model': None, 'mode': 3, 'fill_background': None, 'crop_size': 8},
]

def plot_images(original_img, processed_img, crop_coords, title):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image with crop rectangle
    if crop_coords is not None:
        draw = ImageDraw.Draw(original_img)
        if isinstance(crop_coords[0], tuple):
            for coords in crop_coords:
                draw.rectangle(coords, outline="red", width=1)
        else:
            draw.rectangle(crop_coords, outline="red", width=1)
    axs[0].imshow(original_img)
    axs[0].set_title('Originálny obrázok')
    axs[0].axis('off')

    # Plot processed image
    axs[1].imshow(processed_img)
    axs[1].set_title(title)
    axs[1].axis('off')

    plt.show()

for config in configurations:
    dataset = DatasetFactory.create_dataset(
        root='./data',
        name=config['dataset_class'],
        train=True,
        download=True,
        model=config.get('model'),
        mode=config.get('mode'),
        fill_background=config.get('fill_background'),
        crop_size=config.get('crop_size', 8)  # Default crop size if not provided
    )

    # Retrieve an image and label
    original_img, label = dataset[0]

    # Convert to PIL image if necessary
    original_img = to_pil_image(original_img) if isinstance(original_img, torch.Tensor) else original_img

    # Apply transformations based on the mode
    processed_img, crop_coords = CIFAR10Utils.create_background_image(
        original_img,
        mode=config['mode'],
        crop_size=config.get('crop_size'),
        model=config.get('model'),
        fill_background=config.get('fill_background')
    )

    title = f"Mode {config['mode']}" if config['mode'] != 4 else ("Segmentovaný obrázok" if config['fill_background'] else "Segmentovaný obrázok")
    plot_images(original_img, processed_img, crop_coords, title)
