"""
training.py: how the network learns, and how we measure it.

THE LEARNING LOOP (the same idea in every function below):
    1. Show the network a batch of images.
    2. It outputs 10 scores per image; the highest is its guess.
    3. The loss function measures how wrong the guesses were.
    4. loss.backward() works out, for EVERY weight, which direction to nudge
       it to make the loss smaller (this is "backpropagation").
    5. optimizer.step() nudges every weight a little in that direction.

WHY FORGETTING HAPPENS: in train_task_b, step 3 only measures mistakes on
the batch it was shown. If that batch contains only task-B images, nothing
checks task A, so the weights task A relied on get pushed around freely.

REPLAY: we keep a small "memory" of task-A images and mix some into every
task-B batch. Now step 3 also measures task-A mistakes, so the nudges try
to keep task A working too.
"""

import torch
import torch.nn as nn


def accuracy(model, x, y, batch_size=2000):
    """Fraction of images labelled correctly. No learning happens here."""
    model.eval()
    correct = 0
    with torch.no_grad():  # turn off gradient tracking: faster, no weight changes
        for start in range(0, len(x), batch_size):
            scores = model(x[start:start + batch_size])
            guesses = scores.argmax(dim=1)  # index of the highest score
            correct += (guesses == y[start:start + batch_size]).sum().item()
    model.train()
    return correct / len(x)


def _one_step(model, optimizer, loss_fn, images, labels):
    """Steps 1-5 of the learning loop, for one batch. Returns the loss."""
    scores = model(images)
    loss = loss_fn(scores, labels)
    optimizer.zero_grad()   # forget the previous step's nudges
    loss.backward()         # work out the nudge for every weight
    optimizer.step()        # apply the nudges
    return loss.item()


def train_task_a(model, task, epochs, batch_size, lr, generator):
    """
    Ordinary training on task A.

    One epoch = one pass through all training images, in a random order
    (the generator makes that order the same every time for a given seed).
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x, y = task["x_train"], task["y_train"]
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            idx = order[start:start + batch_size]
            _one_step(model, optimizer, loss_fn, x[idx], y[idx])


def train_task_b(model, task_b, memory_x, memory_y, steps, batch_size, lr,
                 gen_b, gen_extra, eval_every, evaluate):
    """
    Training on task B, optionally with replay.

    FAIRNESS RULE: every step uses 2 x batch_size images, whatever the
    condition, so all conditions do the same amount of computation:
        - first half:  always new task-B images
        - second half: old task-A images from memory (if there is a memory),
                       otherwise MORE task-B images
    Without this rule, replay runs would do twice the work, and we couldn't
    tell whether replay helped or extra computation helped.

    `evaluate` is a function that measures both tasks; we call it every
    `eval_every` steps to build learning curves.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xb, yb = task_b["x_train"], task_b["y_train"]
    use_memory = memory_x is not None and len(memory_x) > 0

    curve = [{"step": 0, **evaluate()}]
    for step in range(1, steps + 1):
        idx_b = torch.randint(0, len(xb), (batch_size,), generator=gen_b)
        if use_memory:
            idx_m = torch.randint(0, len(memory_x), (batch_size,), generator=gen_extra)
            images = torch.cat([xb[idx_b], memory_x[idx_m]])
            labels = torch.cat([yb[idx_b], memory_y[idx_m]])
        else:
            idx_extra = torch.randint(0, len(xb), (batch_size,), generator=gen_extra)
            images = torch.cat([xb[idx_b], xb[idx_extra]])
            labels = torch.cat([yb[idx_b], yb[idx_extra]])
        _one_step(model, optimizer, loss_fn, images, labels)

        if step % eval_every == 0:
            curve.append({"step": step, **evaluate()})
    return curve
