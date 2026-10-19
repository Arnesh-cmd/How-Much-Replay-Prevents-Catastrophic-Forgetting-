# Step 1: A small handwritten-digit classifier (MNIST)
# Paste into a Google Colab cell and run. A CPU is fine; no GPU needed.
# PyTorch and torchvision are already installed in Colab.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---- 1. Fix the random seed so you get the same results every run ----
torch.manual_seed(0)

# ---- 2. Load the data ----
# ToTensor turns each 28x28 image into numbers between 0 (black) and 1 (white).
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# A DataLoader hands the data to the network in small groups called "batches".
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

print("Training images:", len(train_data))   # expect 60000
print("Test images:", len(test_data))        # expect 10000
image, label = train_data[0]
print("One image has shape", tuple(image.shape), "and label", label)  # expect (1, 28, 28) and 5

# ---- 3. Build the network ----
model = nn.Sequential(
    nn.Flatten(),          # 28x28 grid -> one list of 784 numbers
    nn.Linear(784, 100),   # hidden layer 1: 100 artificial neurons
    nn.ReLU(),             # keeps positive values, turns negatives into 0
    nn.Linear(100, 100),   # hidden layer 2: 100 neurons
    nn.ReLU(),
    nn.Linear(100, 10),    # output: one score for each digit 0-9
)

# The loss measures how wrong the network's scores are (lower = better).
loss_fn = nn.CrossEntropyLoss()
# The optimiser nudges every weight to reduce the loss. lr = step size.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def test_accuracy(model, loader):
    """Fraction of images the network labels correctly. No learning happens here."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():                      # no weight changes during testing
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)   # pick the highest-scoring digit
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total


# ---- 4. Sanity check: an untrained network should be guessing ----
print(f"Accuracy before training: {test_accuracy(model, test_loader):.3f}")  # expect roughly 0.10

# ---- 5. Train ----
for epoch in range(3):                         # one epoch = one pass through all 60000 images
    total_loss = 0.0
    for images, labels in train_loader:
        scores = model(images)                 # forward pass: make predictions
        loss = loss_fn(scores, labels)         # how wrong were they?
        optimizer.zero_grad()                  # clear the previous step's adjustments
        loss.backward()                        # backpropagation: work out how each weight affected the error
        optimizer.step()                       # nudge every weight to reduce the error
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    acc = test_accuracy(model, test_loader)
    print(f"Epoch {epoch + 1}: average loss {avg_loss:.3f}, test accuracy {acc:.3f}")

# Expect test accuracy to climb to roughly 0.96-0.98 by epoch 3.
