from torchvision.transforms.functional import crop
import random
import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
from PIL import Image
from torchvision import transforms


transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class CIFAR10Utils:
    @staticmethod
    def crop_single_corner(image, corner, crop_size):
        width, height = image.size
        corners = {
            'top_left': (0, 0, crop_size, crop_size),
            'top_right': (width - crop_size, 0, width, crop_size),
            'bottom_left': (0, height - crop_size, crop_size, height),
            'bottom_right': (width - crop_size, height - crop_size, width, height)
        }
        return image.crop(corners[corner]), corners[corner]

    @staticmethod
    def crop_resize_combine_corners(image, corner_crop_size):
        resized_corner_size = (16, 16)
        width, height = image.size
        corners = {
            'top_left': (0, 0, corner_crop_size[0], corner_crop_size[1]),
            'top_right': (width - corner_crop_size[0], 0, width, corner_crop_size[1]),
            'bottom_left': (0, height - corner_crop_size[1], corner_crop_size[0], height),
            'bottom_right': (width - corner_crop_size[0], height - corner_crop_size[1], width, height)
        }
        final_image = Image.new('RGB', (32, 32))
        for i, (corner, coordinates) in enumerate(corners.items()):
            crop = image.crop(coordinates).resize(resized_corner_size)
            final_image.paste(crop, ((i % 2) * 16, (i // 2) * 16))
        return final_image, list(corners.values())

    @staticmethod
    def segment_image(img, model, fill_background=None):
        device = next(model.parameters()).device
        img_tensor = to_tensor(img).float().unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            seg_out = model(img_tensor)
            attn = torch.sigmoid(torch.logsumexp(seg_out, 1, keepdim=True))

        mask = attn.squeeze().cpu().numpy() > 0.005
        original_img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

        if fill_background:
            background_color = CIFAR10Utils.calculate_mean_color(original_img, ~mask)
            white_color = np.ones_like(original_img) * 1.0
            segmented_img = np.where(mask[..., None], original_img, white_color)
        else:
            background_color_distribution = CIFAR10Utils.get_color_distribution(original_img, ~mask)
            background_color = CIFAR10Utils.calculate_mean_color(original_img, ~mask)
            segmented_img = CIFAR10Utils.apply_color_distribution(original_img, mask, background_color)

        return segmented_img, None

    @staticmethod
    def calculate_mean_color(img, mask):
        masked_pixels = img[mask]
        mean_color = np.mean(masked_pixels, axis=0)
        return mean_color

    @staticmethod
    def get_color_distribution(img, mask):
        masked_pixels = img[mask]
        return masked_pixels

    @staticmethod
    def apply_color_distribution(img, mask, color_distribution):
        h, w, c = img.shape
        flat_color_distribution = color_distribution.reshape(-1, c)
        sampled_colors = flat_color_distribution[np.random.choice(flat_color_distribution.shape[0], mask.sum(), replace=True)]
        segmented_img = img.copy()
        segmented_img[mask] = sampled_colors

        return segmented_img

    @staticmethod
    def create_background_image(image, mode, crop_size=None, model=None, fill_background=None):
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)

        if mode == 1:
            x, y = random.randint(0, 24), random.randint(0, 24)
            crop_coords = (x, y, x + crop_size, y + crop_size)
            cropped_image = crop(image, x, y, crop_size, crop_size)
            resized_image = resize(cropped_image, [32, 32])
            return resized_image, crop_coords
        elif mode == 2:
            corner = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
            cropped_image, crop_coords = CIFAR10Utils.crop_single_corner(image, corner, crop_size)
            resized_image = resize(cropped_image, [32, 32])
            return resized_image, crop_coords
        elif mode == 3:
            resized_image, crop_coords = CIFAR10Utils.crop_resize_combine_corners(image, (crop_size, crop_size))
            return resized_image, crop_coords
        elif mode == 4 and model is not None:
            return CIFAR10Utils.segment_image(image, model, fill_background=fill_background)
        else:
            raise ValueError("Invalid mode or missing model for segmentation.")

class CIFAR10WithBackground(torchvision.datasets.CIFAR10):
    def __init__(self, *args, model=None, mode=None, fill_background=None, crop_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.mode = mode
        self.fill_background = fill_background
        self.crop_size = crop_size

    def __getitem__(self, index):
        original_index = index % len(self.data)
        image, _ = super().__getitem__(original_index)
        if index >= len(self.data):
            image = CIFAR10Utils.create_background_image(image=image,
                                                         mode=self.mode,
                                                         crop_size=self.crop_size,
                                                         model=self.model,
                                                         fill_background=self.fill_background)
            label = 10
        else:
            label = self.targets[original_index]

        #image = transform_to_tensor(image)

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return 2 * len(self.data)

class CIFAR10WithClassSpecificBackground(torchvision.datasets.CIFAR10):
    def __init__(self, *args, model=None, mode=None, fill_background=None, crop_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.mode = mode
        self.fill_background = fill_background
        self.crop_size = crop_size

    def __getitem__(self, index):
        is_background = index >= len(self.data)
        original_index = index % len(self.data)

        image, label = super().__getitem__(original_index)

        if is_background:
            image = CIFAR10Utils.create_background_image(image=image,
                                                         mode=self.mode,
                                                         crop_size=self.crop_size,
                                                         model=self.model,
                                                         fill_background=self.fill_background)
            label += 10

        image = transform_to_tensor(image)

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return 2 * len(self.data)

class CIFAR10WithBackgroundSoftLabel(torchvision.datasets.CIFAR10):
    def __init__(self, *args, model=None, mode=None, fill_background=None, crop_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.mode = mode
        self.fill_background = fill_background
        self.crop_size = crop_size

    def __getitem__(self, index):
        original_index = index % len(self.data)
        image, _ = super().__getitem__(original_index)

        if index >= len(self.data):
            image = CIFAR10Utils.create_background_image(image=image,
                                                         mode=self.mode,
                                                         crop_size=self.crop_size,
                                                         model=self.model,
                                                         fill_background=self.fill_background)
            soft_label = np.zeros(11)
            soft_label[-1] = 0.8
            soft_label[self.targets[original_index]] = 0.2
        else:
            soft_label = np.zeros(11)
            soft_label[self.targets[original_index]] = 1.0

        image = transform_to_tensor(image)

        return image, torch.tensor(soft_label, dtype=torch.float32)

    def __len__(self):
        return 2 * len(self.data)

class CIFAR10WithClassSpecificBackgroundSoftLabel(torchvision.datasets.CIFAR10):
    def __init__(self, *args, model=None, mode=None, fill_background=None, crop_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.mode = mode
        self.fill_background = fill_background
        self.crop_size = crop_size

    def __getitem__(self, index):
        original_index = index % len(self.data)
        image, label = super().__getitem__(original_index)

        if index >= len(self.data):
            image = CIFAR10Utils.create_background_image(image=image,
                                                         mode=self.mode,
                                                         crop_size=self.crop_size,
                                                         model=self.model,
                                                         fill_background=self.fill_background)
            soft_label = np.zeros(20)
            soft_label[label + 10] = 0.8
            soft_label[label] = 0.2
        else:
            soft_label = np.zeros(20)
            soft_label[label] = 1.0

        image = transform_to_tensor(image)

        return image, torch.tensor(soft_label, dtype=torch.float32)

    def __len__(self):
        return 2 * super().__len__()

class CIFAR10WithDistributedSoftBackgroundLabels(torchvision.datasets.CIFAR10):
    def __init__(self, *args, model=None, mode=None, fill_background=None, crop_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.mode = mode
        self.fill_background = fill_background
        self.crop_size = crop_size

    def __getitem__(self, index):
        original_index = index % len(self.data)
        image, label = super().__getitem__(original_index)

        if index >= len(self.data):
            image = CIFAR10Utils.create_background_image(image=image,
                                                         mode=self.mode,
                                                         crop_size=self.crop_size,
                                                         model=self.model,
                                                         fill_background=self.fill_background)
            soft_label = np.full(11, 0.02)
            soft_label[-1] = 0.8
        else:
            soft_label = np.zeros(11)
            soft_label[label] = 1.0

        image = transform_to_tensor(image)

        return image, torch.tensor(soft_label, dtype=torch.float32)

    def __len__(self):
        return 2 * len(self.data)

class CIFAR10WithDistributedClassSpecificSoftBackgroundLabels(torchvision.datasets.CIFAR10):
    def __init__(self, *args, model=None, mode=None, fill_background=None, crop_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.mode = mode
        self.fill_background = fill_background
        self.crop_size = crop_size

    def __getitem__(self, index):
        original_index = index % len(self.data)
        image, label = super().__getitem__(original_index)

        if index >= len(self.data):
            image = CIFAR10Utils.create_background_image(image=image,
                                                         mode=self.mode,
                                                         crop_size=self.crop_size,
                                                         model=self.model,
                                                         fill_background=self.fill_background)
            soft_label = np.full(20,0.2 / 19)
            soft_label[label + 10] = 0.8
        else:
            soft_label = np.zeros(20)
            soft_label[label] = 1.0

        image = transform_to_tensor(image)

        return image, torch.tensor(soft_label, dtype=torch.float32)

    def __len__(self):
        return 2 * super().__len__()

class SegmentedCIFAR10WithObject(torchvision.datasets.CIFAR10):
    def __init__(self, *args, model=None, mode=None, fill_background=None, crop_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.mode = mode
        self.fill_background = fill_background
        self.crop_size = crop_size

    def __getitem__(self, index):
        original_index = index % len(self.data)
        image, label = super().__getitem__(original_index)

        if index >= len(self.data):
            image = CIFAR10Utils.create_background_image(image=image,
                                                                 mode=self.mode,
                                                                 crop_size=self.crop_size,
                                                                 model=self.model,
                                                                 fill_background=self.fill_background)
            label = self.targets[original_index]
        else:
            label = self.targets[original_index]

        #image = transform_to_tensor(image)

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return 2 * len(self.data)

class DefaultCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, model=None, mode=None, fill_background=None, crop_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.mode = mode
        self.fill_background = fill_background
        self.crop_size = crop_size

    def __getitem__(self, index):
        image, label = super(DefaultCIFAR10, self).__getitem__(index)

        image = transform_to_tensor(image)

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return super(DefaultCIFAR10, self).__len__()