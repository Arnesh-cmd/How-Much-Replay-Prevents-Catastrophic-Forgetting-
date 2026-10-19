# How Much Replay Prevents Catastrophic Forgetting?

A controlled experiment on how many saved old examples a neural network needs to stop forgetting an old task, and whether that number depends on how different the new task is.

**Author:** [Your Name] · **Date:** October 2026 · **Paper:** [link to PDF]

---

## Key finding

A small neural network learned to read handwritten digits (task A), then the same digits rotated by 0°–90° (task B). Without help, it forgot up to **65 percentage points** of accuracy on the normal digits. Mixing a few saved normal digits into training ("replay") prevented most of this:

| Replay memory | Share of forgetting removed (30°–90°) |
|---|---|
| 10 images | 29–30% |
| 100 images | 62–71% |
| 1,000 images | 92–94% |
| 10,000 images | ~100% |

**A given memory size removed nearly the same share of forgetting at every angle from 30° to 90°**, even though bigger angles caused much more forgetting. Replay cost at most 2.6 percentage points of accuracy on the new task. Training ten times longer on task A protected it far less than a 100-image replay memory.

---

## The experiment

| | |
|---|---|
| **Task A** | MNIST handwritten digits (normal) |
| **Task B** | The same digits rotated by 0°, 15°, 30°, 45°, 60° or 90° |
| **Network** | Fully connected: 784 → 100 → 100 → 10, ReLU, SGD (learning rate 0.1) |
| **Training** | 1 epoch on task A, then 1,000 steps on task B |
| **Replay memory** | 0, 10, 100, 1,000 or 10,000 random task-A training images |
| **Repeats** | 10 random seeds per condition (300 runs in the main experiment) |
| **Forgetting** | Task-A test accuracy before task B minus after task B |

**Fairness:** every condition uses the same number of task-B steps and the same 128 images per step, so no condition trains longer or computes more. Each seed is shared across conditions, so they differ only in the variable being tested. Test images are never used for training.

**Side experiments:**
- **Part 1:** does training longer on task A (1, 2, 5, 10 epochs) protect it?
- **Width:** do wider networks (25, 100, 400 neurons) forget less?

---

## Run it yourself

### In Google Colab (no installation)

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. In the first cell, run:
   ```
   !git clone https://github.com/<your-username>/forgetting-replay-study.git
   %cd forgetting-replay-study
   ```
3. Then run the steps below, each in its own cell with `!` in front.

Alternatively, upload `run_in_colab.ipynb`, which walks through every step.

### On your own computer

```bash
git clone https://github.com/<your-username>/forgetting-replay-study.git
cd forgetting-replay-study
pip install -r requirements.txt
```

### Steps

```bash
python test_everything.py        # 18 automated checks (~1 min)
python experiment.py pilot       # small check run (~2-5 min)
python analyze.py pilot
python experiment.py main        # main experiment, 300 runs (~20-40 min on CPU)
python analyze.py main
python experiment.py part1       # longer practice on task A (~10-15 min)
python analyze.py part1
python experiment.py width       # network width (~5 min)
python analyze.py width
```

MNIST downloads automatically on the first run. Results are saved after every run, so an interrupted experiment resumes where it stopped.

**Note:** this repository already contains the original results in `results/`, so `experiment.py` will skip runs that are already done. To repeat the experiments from scratch, delete or rename the `results/` folder first. To only re-create the tables and figures from the original data, run `analyze.py` directly.

---

## Repository contents

| File | Purpose |
|---|---|
| `config.py` | Every setting for every experiment, in one place |
| `data.py` | Loads MNIST and builds the rotated task B |
| `model.py` | The neural network |
| `training.py` | Training on task A, training on task B with replay, accuracy |
| `experiment.py` | Runs every combination of settings and seeds, saves results |
| `analyze.py` | Means, 95% bootstrap confidence intervals, paired tests, figures |
| `test_everything.py` | Automated checks that the code is correct |
| `00_warmup_digit_classifier.py` | A minimal standalone digit classifier |
| `run_in_colab.ipynb` | Step-by-step notebook for Google Colab |
| `GUIDE.md` | Detailed explanation of every design decision |
| `results/` | Raw results (CSV) and the settings and software versions used (JSON) |
| `results/figures/` | Figures for each experiment, created by `analyze.py` |

---

## Reproducibility

- Same settings + same seed give identical results. All 27 pilot runs were reproduced exactly inside the main experiment.
- Each experiment saves its exact settings and software versions in `results/<name>_config.json`.
- Original runs: Google Colab CPU, Python 3.13.16, PyTorch 2.11.0. Other versions or hardware may give slightly different numbers.

## Limitations

- Runs without replay saw 128 new task-B images per step, replay runs 64; part of replay's small cost on task B may come from this.
- Task A was trained for only 1 epoch (93% accuracy).
- One dataset, one small network, one optimiser; results may not hold for large models or real-world data.
- Rotation is a simple, artificial kind of task difference, and only two tasks were tested.

## Related work

- McCloskey & Cohen (1989). [Catastrophic interference in connectionist networks](https://doi.org/10.1016/S0079-7421(08)60536-8).
- Goodfellow et al. (2014). [An empirical investigation of catastrophic forgetting in gradient-based neural networks](https://arxiv.org/abs/1312.6211).
- Kirkpatrick et al. (2017). [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796).
- Chaudhry et al. (2019). [On tiny episodic memories in continual learning](https://arxiv.org/abs/1902.10486).
- Rolnick et al. (2019). [Experience replay for continual learning](https://arxiv.org/abs/1811.11682).
- van de Ven & Tolias (2019). [Three scenarios for continual learning](https://arxiv.org/abs/1904.07734).
- Ramasesh et al. (2021). [Anatomy of catastrophic forgetting](https://arxiv.org/abs/2007.07400).

## License

MIT. See `LICENSE`.
