import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from data.dataset_factory import DatasetFactory
from models import create_tiny_resnet
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

segmentation_model = create_tiny_resnet()
segmentation_model_path = '../segment.pth'
segmentation_model.load_state_dict(torch.load(segmentation_model_path))
segmentation_model.eval()
segmentation_model.to(device)


configurations = [
    {'dataset_class': "SegmentedCIFAR10WithObject",'model': segmentation_model, 'mode': 4, 'fill_background': True, 'crop_size': None },

    {'dataset_class': "CIFAR10WithBackground", 'model': segmentation_model, 'mode': 4,'fill_background': False, 'crop_size': None },
    {'dataset_class': "CIFAR10WithBackground", 'model': None, 'mode': 2,'fill_background': None, 'crop_size': 16 },
    {'dataset_class': "CIFAR10WithBackgroundSoftLabel", 'model': segmentation_model, 'mode': 4,'fill_background': False, 'crop_size': None },
    {'dataset_class': "CIFAR10WithBackgroundSoftLabel", 'model': None, 'mode': 2,'fill_background': None, 'crop_size': 16 },
    {'dataset_class': "CIFAR10WithDistributedSoftBackgroundLabels", 'model': segmentation_model, 'mode': 4,'fill_background': False, 'crop_size': None },
    {'dataset_class': "CIFAR10WithDistributedSoftBackgroundLabels", 'model': None, 'mode': 2,'fill_background': None, 'crop_size': 16 },
    {'dataset_class': "CIFAR10WithClassSpecificBackground", 'model': segmentation_model, 'mode': 4,'fill_background': False, 'crop_size': None },
    {'dataset_class': "CIFAR10WithClassSpecificBackground", 'model': None, 'mode': 2,'fill_background': None, 'crop_size': 16 },
    {'dataset_class': "CIFAR10WithClassSpecificBackgroundSoftLabel", 'model': segmentation_model, 'mode': 4,'fill_background': False, 'crop_size': None },
    {'dataset_class': "CIFAR10WithClassSpecificBackgroundSoftLabel", 'model': None, 'mode': 2,'fill_background': None, 'crop_size': 16 },
    {'dataset_class': "CIFAR10WithDistributedClassSpecificSoftBackgroundLabels", 'model': segmentation_model, 'mode': 4,'fill_background': False, 'crop_size': None },
    {'dataset_class': "CIFAR10WithDistributedClassSpecificSoftBackgroundLabels", 'model': None, 'mode': 2,'fill_background': None, 'crop_size': 16 },

    {'dataset_class': "DefaultCIFAR10", 'model': None, 'mode': None, 'fill_background': None, 'crop_size': None}

]


for config in configurations:
    dataset = DatasetFactory.create_dataset(
        root='./data',
        name=config['dataset_class'],
        train=True,
        download=True,
        model=config.get('model'),
        mode=config.get('mode'),
        fill_background=config.get('fill_background'),
        crop_size=config.get('crop_size')
    )
    img, label = dataset[0]
    print(dataset.__len__())

    img = to_pil_image(img) if isinstance(img, torch.Tensor) else img

    print(f'Label before one_hot: {label}, {label.dtype}')

    if isinstance(label, torch.Tensor) and label.dtype == torch.float32:
        formatted_label = [f'{number:.5f}' for number in label.numpy()]
        print(f'Soft label: [{', '.join(formatted_label)}] {label.dtype}')
        plt.figure()
        plt.imshow(img)
        plt.title(f'Label: {formatted_label}')
        plt.show()
    else:
        label = torch.tensor(label, dtype=torch.int64) if not isinstance(label, torch.Tensor) else label.long()
        if label.ndim == 0 or (label.ndim == 1 and label.size(0) == 1):
            label = label.view(-1)
            label = F.one_hot(label, num_classes=20).float()
            formatted_label_one_hot = [f'{number:.5f}' for number in label.numpy().ravel()]
            print(f'Hard label after one_hot: [{', '.join(formatted_label_one_hot)}] {label.dtype}')
            plt.figure()
            plt.imshow(img)
            plt.title(f'Label: {label}')
            plt.show()