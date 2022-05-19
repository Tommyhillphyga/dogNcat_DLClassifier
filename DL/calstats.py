import numpy as np
import torch, torchvision

def getmeanNstd(loader):
    mean, std, total_no_of_image = 0., 0., 0
    for images, _ in loader:
        batch_size = images.size(0)
        image = images.view(batch_size, images.size(1), -1)
        mean += image.mean(2).sum(0)
        std += image.std(2).sum(0)
        total_no_of_image+=batch_size

    mean /= total_no_of_image
    std /= total_no_of_image

    return mean, std

