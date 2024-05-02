from torch.utils.data import DataLoader
from models import create_tiny_resnet, BATCH_SIZE, NUM_WORKERS, attention
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, utils
import torch
import numpy as np

BACKGROUND_COLOR = [255,0,0]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_dataloader(train=False):
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def visualize_masks(loader, model):
    model.eval()
    plt.figure(figsize=(20, 10))
    with torch.no_grad():
        for x, _ in loader:
            x = x.cuda()
            seg_out = model(x)
            attn = attention(seg_out)
            for i in range(min(len(x), 1)):  # Visualize the first 5 images
                original_img = utils.make_grid(x[i].cpu()).numpy().transpose(1, 2, 0)
                mask = attn[i].cpu().numpy()[0]

                # Apply threshold to create binary mask
                thresholded_mask = mask > 0.25

                # Prepare masked image
                background = np.ones_like(original_img) * np.array(BACKGROUND_COLOR) / 255.0
                masked_image = np.where(thresholded_mask[..., None], original_img, background)

                # Plot image with mask
                plt.subplot(2, 5, i + 6)
                plt.imshow(masked_image)
                plt.axis('off')
            plt.show()
            break  # Only show one batch for sampling


if __name__ == '__main__':
    model = create_tiny_resnet()
    model.load_state_dict(torch.load('../segment.pth'))
    test_loader = get_dataloader(train=False)
    visualize_masks(test_loader, model)
