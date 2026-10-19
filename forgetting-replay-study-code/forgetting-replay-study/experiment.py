"""
experiment.py: runs a whole experiment grid and saves the results.

USAGE (in a terminal, or with ! in front in a Colab cell):
    python experiment.py pilot          # small check run - do this first
    python experiment.py main           # the core experiment
    python experiment.py part1          # does more practice on A protect it?
    python experiment.py width          # do wider networks forget less?
    python experiment.py pilot --fake   # use fake data (only for testing code)

WHAT ONE "RUN" DOES (see run_one below):
    1. Fix all random seeds.
    2. Build a fresh network.
    3. Train it on task A. Measure accuracy on A ("before").
    4. Pick `memory_size` random task-A training images as the replay memory.
    5. Train on task B for a fixed number of steps (with replay if memory > 0),
       measuring both tasks every few steps.
    6. Measure accuracy on A again ("after").  forgetting = before - after

OUTPUT FILES (in results/):
    <name>_summary.csv   one row per run: the final numbers
    <name>_curves.csv    accuracy on both tasks during task-B training
    <name>_config.json   the exact settings + software versions used

RESUMING: results are saved after every run. If Colab disconnects, just run
the same command again: finished runs are skipped.
"""

import csv
import itertools
import json
import os
import platform
import random
import sys
import time

import numpy as np
import torch

from config import EXPERIMENTS
from data import build_tasks, load_mnist, make_fake_data
from model import make_model
from training import accuracy, train_task_a, train_task_b

SUMMARY_FIELDS = [
    "angle", "memory_size", "epochs_task_a", "hidden_width", "seed",
    "acc_a_before", "acc_b_before", "acc_a_after", "acc_b_after",
    "forgetting", "seconds",
]
CURVE_FIELDS = [
    "angle", "memory_size", "epochs_task_a", "hidden_width", "seed",
    "step", "acc_a", "acc_b",
]


def set_all_seeds(seed):
    """Make every source of randomness repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_generator(seed, purpose):
    """
    A separate random-number stream for each job (task A order, memory choice,
    task B batches, extra batches). Keeping them separate means that changing
    the memory size does NOT change which task-B images are shown - so the
    conditions differ only in the thing we deliberately changed.
    """
    offsets = {"task_a": 0, "memory": 1, "task_b": 2, "extra": 3}
    return torch.Generator().manual_seed(seed * 10 + offsets[purpose])


def run_one(task_a, task_b, memory_size, epochs_a, width, seed, cfg):
    """One complete run. Returns (summary_row, curve_rows)."""
    started = time.time()
    set_all_seeds(seed)
    model = make_model(width)  # same seed + same width = identical starting weights

    # ---- Step 3: learn task A ----
    train_task_a(model, task_a, epochs_a, cfg["batch_size"], cfg["learning_rate"],
                 make_generator(seed, "task_a"))
    acc_a_before = accuracy(model, task_a["x_test"], task_a["y_test"])
    acc_b_before = accuracy(model, task_b["x_test"], task_b["y_test"])

    # ---- Step 4: choose the replay memory from task A's TRAINING images ----
    # (Never from the test images - the test set must stay unseen.)
    if memory_size > 0:
        chosen = torch.randperm(len(task_a["x_train"]),
                                generator=make_generator(seed, "memory"))[:memory_size]
        memory_x, memory_y = task_a["x_train"][chosen], task_a["y_train"][chosen]
    else:
        memory_x, memory_y = None, None

    # ---- Step 5: learn task B, measuring both tasks along the way ----
    def evaluate():
        return {
            "acc_a": accuracy(model, task_a["x_test"], task_a["y_test"]),
            "acc_b": accuracy(model, task_b["x_test"], task_b["y_test"]),
        }

    curve = train_task_b(model, task_b, memory_x, memory_y, cfg["steps_task_b"],
                         cfg["batch_size"], cfg["learning_rate"],
                         make_generator(seed, "task_b"), make_generator(seed, "extra"),
                         cfg["eval_every"], evaluate)

    # ---- Step 6: final measurements ----
    final = evaluate()
    settings = {"memory_size": memory_size, "epochs_task_a": epochs_a,
                "hidden_width": width, "seed": seed}
    summary = {
        **settings,
        "acc_a_before": acc_a_before, "acc_b_before": acc_b_before,
        "acc_a_after": final["acc_a"], "acc_b_after": final["acc_b"],
        "forgetting": acc_a_before - final["acc_a"],
        "seconds": round(time.time() - started, 1),
    }
    curve_rows = [{**settings, **point} for point in curve]
    return summary, curve_rows


def _run_key(row):
    """Identifies a run, so finished runs can be skipped when resuming."""
    return (int(row["angle"]), int(row["memory_size"]), int(row["epochs_task_a"]),
            int(row["hidden_width"]), int(row["seed"]))


def save_config(cfg, name, fake):
    """Record exactly what produced these results (reproducibility)."""
    record = {
        "experiment": name,
        "fake_data": fake,
        "settings": cfg,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(cfg["results_dir"], f"{name}_config.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def run_experiment(name, fake=False, cfg_override=None):
    cfg = dict(EXPERIMENTS[name])
    if cfg_override:
        cfg.update(cfg_override)
    if fake:
        name = name + "_fake"  # never mix fake and real results
    os.makedirs(cfg["results_dir"], exist_ok=True)
    save_config(cfg, name, fake)

    summary_path = os.path.join(cfg["results_dir"], f"{name}_summary.csv")
    curve_path = os.path.join(cfg["results_dir"], f"{name}_curves.csv")

    # Which runs are already finished? (for resuming)
    done = set()
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            done = {_run_key(r) for r in csv.DictReader(f)}

    data = make_fake_data() if fake else load_mnist(cfg["data_dir"])

    grid = list(itertools.product(cfg["angles"], cfg["memory_sizes"],
                                  cfg["epochs_task_a"], cfg["hidden_widths"], cfg["seeds"]))
    print(f"Experiment '{name}': {len(grid)} runs ({len(done)} already done)")

    tasks_cache = {}  # rotating 60,000 images is slow, so do it once per angle
    for i, (angle, mem, epochs_a, width, seed) in enumerate(grid, start=1):
        if (angle, mem, epochs_a, width, seed) in done:
            continue
        if angle not in tasks_cache:
            tasks_cache = {angle: build_tasks(data, angle)}  # keep one angle in memory
        task_a, task_b = tasks_cache[angle]

        summary, curve = run_one(task_a, task_b, mem, epochs_a, width, seed, cfg)
        summary = {"angle": angle, **summary}
        curve = [{"angle": angle, **row} for row in curve]
        _append_rows(summary_path, SUMMARY_FIELDS, [summary])
        _append_rows(curve_path, CURVE_FIELDS, curve)

        print(f"[{i}/{len(grid)}] angle={angle:>3} memory={mem:>5} epochsA={epochs_a:>2} "
              f"width={width:>3} seed={seed:>2} | A before {summary['acc_a_before']:.3f} "
              f"-> after {summary['acc_a_after']:.3f} | B {summary['acc_b_after']:.3f} "
              f"| {summary['seconds']}s")
    print(f"Done. Results in {summary_path}")


def _append_rows(path, fields, rows):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in EXPERIMENTS:
        print("Usage: python experiment.py <" + "|".join(EXPERIMENTS) + "> [--fake]")
        sys.exit(1)
    run_experiment(args[0], fake="--fake" in args)
