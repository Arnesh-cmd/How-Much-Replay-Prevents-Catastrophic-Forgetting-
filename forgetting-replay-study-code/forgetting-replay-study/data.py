"""
data.py: loads the digit images and builds task A and task B.

    Task A = normal MNIST handwritten digits.
    Task B = the SAME digits, rotated by some angle.

The rotation angle is our "similarity dial": 0 degrees means task B is
identical to task A, 90 degrees means it looks very different.

Every image is stored as numbers between 0 (black) and 1 (white), in a
"tensor" (PyTorch's name for a grid of numbers) of shape:

    (number_of_images, 1, 28, 28)
     ^ how many         ^ one colour channel (greyscale), 28x28 pixels
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def load_mnist(data_dir):
    """Download MNIST (once) and return it as four tensors."""
    from torchvision import datasets  # imported here so tests can run without it

    train = datasets.MNIST(data_dir, train=True, download=True)
    test = datasets.MNIST(data_dir, train=False, download=True)

    # .data holds the raw pixel values 0-255. Divide by 255 to get 0-1, and
    # add the channel dimension with unsqueeze(1).
    x_train = train.data.float().div(255).unsqueeze(1)
    x_test = test.data.float().div(255).unsqueeze(1)
    return x_train, train.targets.clone(), x_test, test.targets.clone()


def make_fake_data(n_train=3000, n_test=1000, seed=0):
    """
    Small made-up dataset with the same shape as MNIST.

    Used ONLY for testing the code quickly without downloading anything.
    Each of the 10 "digits" is a smooth random blob pattern plus noise, so a
    network can learn to tell them apart, and rotating them changes them
    enough to cause forgetting. Never report results from fake data.
    """
    g = torch.Generator().manual_seed(seed)
    coarse = torch.rand(10, 1, 7, 7, generator=g)
    templates = F.interpolate(coarse, size=(28, 28), mode="bilinear", align_corners=False)

    def sample(n):
        labels = torch.randint(0, 10, (n,), generator=g)
        noise = torch.rand(n, 1, 28, 28, generator=g)
        images = 0.5 * templates[labels] + 0.5 * noise
        return images, labels

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    return x_train, y_train, x_test, y_test


def rotate(images, angle):
    """Rotate every image by `angle` degrees (anticlockwise)."""
    if angle == 0:
        return images.clone()
    # Bilinear interpolation blends neighbouring pixels so rotated digits
    # stay smooth instead of jagged.
    return TF.rotate(images, angle, interpolation=InterpolationMode.BILINEAR)


def build_tasks(data, angle):
    """
    Given the base data and an angle, return task A and task B.

    Both tasks use the same labels (a rotated 7 is still a 7), so the network
    keeps the same 10 outputs. Only the look of the images changes.
    """
    x_train, y_train, x_test, y_test = data
    task_a = {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
    }
    task_b = {
        "x_train": rotate(x_train, angle), "y_train": y_train,
        "x_test": rotate(x_test, angle), "y_test": y_test,
    }
    return task_a, task_b
