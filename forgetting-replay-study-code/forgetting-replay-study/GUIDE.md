# Catastrophic Forgetting Project: Guide

**Research question:** *How much replay of old examples is needed to stop a neural network forgetting an old task, and does that amount depend on how similar the new task is?*

**Part 1 (side question):** *Does practising the old task for longer protect it?*

Read this guide before running anything. Keep it open while you work.

---

## 1. The project in one picture

```
 Task A: normal digits          Task B: the same digits, rotated
 ┌─────┐ ┌─────┐                ┌─────┐ ┌─────┐
 │  7  │ │  3  │      then      │  ⟋  │ │  ω  │   (rotated by 0°–90°)
 └─────┘ └─────┘                └─────┘ └─────┘

 1. Train the network on A           → measure accuracy on A ("before")
 2. Train it on B, mixing in a few
    old A images from a "memory"     → measure accuracy on A again ("after")
 3. Forgetting = before − after
 4. Repeat with different memory sizes and angles, 10 seeds each
```

Two dials:
- **Angle** (0°, 15°, 30°, 45°, 60°, 90°) is how *different* task B is. At 0° task B is identical to A.
- **Memory size** (0, 10, 100, 1000, 10000 images) is how many old task-A images are kept for replay. 0 means no replay.

---

## 2. What each file does

Read the files in this order. Every file has comments explaining each part.

| File | What it does | Why it exists |
|---|---|---|
| `00_warmup_digit_classifier.py` | A tiny standalone digit classifier | Learn the basics first: loading data, a network, the training loop |
| `config.py` | **All settings for all experiments** | So every result can be traced to exact settings. Change numbers *only* here |
| `data.py` | Loads MNIST, rotates images to make task B, makes fake data for tests | Keeps all data handling in one place |
| `model.py` | Builds the neural network (784 → hidden → hidden → 10) | One definition, used everywhere |
| `training.py` | The learning loop: train on A, train on B with replay, measure accuracy | The heart of the experiment |
| `experiment.py` | Runs every combination of settings × seeds, saves results to CSV | Turns one run into a whole controlled experiment |
| `analyze.py` | Averages, confidence intervals, statistical tests, figures | Turns raw numbers into evidence |
| `test_everything.py` | 18 automatic checks that the code is correct | Catches silent bugs before they ruin your results |

### How the pieces connect

```
config.py ──► experiment.py ──► data.py      (get task A and B)
                   │       └──► model.py     (fresh network)
                   │       └──► training.py  (train A, then B + replay)
                   ▼
             results/*.csv ──► analyze.py ──► tables + figures
```

---

## 3. How to run it in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) and choose **File → Upload notebook**. Upload `run_in_colab.ipynb`.
2. Run the first cell. It asks you to upload `forgetting-replay-study.zip`, then unzips it.
3. Run the remaining cells **in order**, one at a time, and read the output of each before moving on.

The notebook runs these commands. You can also type them yourself in a cell, with `!` in front:

```
!python test_everything.py          # 1. check the code (~1 min)
!python 00_warmup_digit_classifier.py   # 2. warm-up (~1 min)
!python experiment.py pilot         # 3. small pilot (~2-5 min)
!python analyze.py pilot            # 4. look at pilot results
!python experiment.py main          # 5. main experiment (~20-40 min)
!python analyze.py main             # 6. main results
!python experiment.py part1         # 7. Part 1 (~10-15 min)
!python analyze.py part1
```

**Important Colab tips:**
- Colab deletes your files when the session ends. **Download the `results` folder** after every experiment. The notebook has a cell that zips it for you.
- If Colab disconnects mid-experiment, reconnect, re-upload and unzip, upload your partial `results` folder, then run the same command again. Finished runs are skipped automatically.
- A GPU isn't needed. The default CPU runtime is fine.

To run it on your own laptop instead, install Python, then run `pip install torch torchvision matplotlib pandas scipy` and the same commands without the `!`.

---

## 4. What you should see (and what's a warning sign)

These are **expected ranges**, not results. Your real numbers are the only results that count.

| Step | Expect | Warning sign |
|---|---|---|
| `test_everything.py` | `18/18 tests passed` | Any FAIL: stop and fix it first |
| Warm-up | Accuracy ≈ 0.10 before training, ≈ 0.96–0.98 after 3 epochs | Stays near 0.10: something is broken |
| Pilot, "A before" | Above ≈ 0.90 in every run | Below 0.90: the analysis warns you; train A longer |
| Pilot, 0° rotation | Forgetting ≈ 0 (it may even be slightly negative) | Large forgetting at 0°: a bug, since A and B are identical |
| Pilot, 90°, memory 0 | Clear forgetting | Almost no forgetting anywhere: the task is too easy, so redesign before the main run |

**Negative forgetting** means task A got *better* during task B. That's normal at small angles, because training on B is like extra practice on A.

---

## 5. Design decisions, and why they make the experiment fair

These are the choices a reviewer will ask about. You should be able to explain each one in your own words.

1. **Same number of task-B steps for every condition.** Otherwise a condition could look better just by training longer.

2. **Same amount of computation for every condition.** Every training step uses 128 images: 64 new task-B images, plus 64 more. The extra 64 come from the replay memory, or, when there's no memory, they're more task-B images. Without this rule, replay runs would do twice the work, and you couldn't tell whether replay helped or extra work helped.

3. **Memory size is counted in images, not percentages.** Earlier we talked about percentages. Counting images is clearer to report ("100 old images were enough") and easier to compare with other papers.

4. **Paired design.** Each seed gives the same starting network and the same task-A training in every condition, so conditions differ *only* in the thing being tested. `test_everything.py` checks this.

5. **Separate random streams.** Changing the memory size doesn't change which task-B images are shown (see `make_generator` in `experiment.py`).

6. **The memory comes from training images only.** The test set is never used for training. If it were, the test accuracy would be cheating.

7. **10 seeds per condition.** One run can be lucky or unlucky. Averages with confidence intervals show how reliable a result is.

8. **Settings are saved with the results.** Each `*_config.json` records the settings and software versions.

---

## 6. Understanding the statistics

- **Mean:** the average over the 10 seeds.
- **95% confidence interval (CI):** the code resamples your 10 results thousands of times (a **bootstrap**) to see how much the average jumps around. Roughly: if you redid the whole experiment, the true average would usually fall inside this range. **If two conditions' CIs overlap a lot, you can't claim a difference.**
- **Paired test:** for each seed, it computes forgetting without replay minus forgetting with replay, then asks whether those differences are consistently above zero (a Wilcoxon signed-rank test). The pilot has only 3 seeds, which is too few for a p-value, so it shows `NaN`. That's expected.
- **Holm correction:** you run about 25 tests in the main experiment. By pure luck, about 1 in 20 tests looks "significant" even when there's no real effect. Holm adjusts the p-values for this. Use `p_holm`, not `p`.

---

## 7. Hypotheses: write them down before the main run

Copy these into your research log **before** running `main`, and add your own predictions. Writing predictions down first is called *pre-registration*. It stops you from adjusting your story to fit whatever came out.

- **H1:** forgetting drops sharply with small memories, then levels off.
- **H2:** forgetting without replay is larger for bigger angles. (Also test: is the *medium* angle hardest to protect with a small memory?)
- **H3 (width experiment):** wider networks forget less.
- **Part 1:** more epochs on task A do **not** substantially reduce forgetting.

**What would refute each one:**
- **H1:** no clear levelling off, or replay doesn't help at all.
- **H2:** forgetting is the same at every angle.
- **H3:** the CIs for widths 25, 100 and 400 overlap completely.
- **Part 1:** forgetting falls clearly as the number of epochs rises.

**A refuted hypothesis is a real result. Report it honestly.**

---

## 8. What you can and cannot claim

**You can claim** what you measured. For example: *"In rotated MNIST with a 2-layer MLP, a replay memory of N images reduced forgetting from X to Y (95% CI …)."*

**You cannot claim:**
- that the same holds for large models, for other tasks, or for the human brain;
- that your method is new: replay is well known. Your contribution is the *measured relationship* between memory size and task similarity;
- anything from fake-data runs. Their results files end in `_fake`, and they are only for testing code.

---

## 9. Keep a research log

Use a notebook or document, dated, with one entry per session:
- what you ran (command and experiment name);
- what you expected;
- what happened, with key numbers;
- anything surprising, and any bugs you found and how you fixed them;
- what you'll do next.

Keep failed and abandoned experiments in the log too. Your paper's "Limitations" section comes straight from here.

---

## 10. Troubleshooting

- **`ModuleNotFoundError: torch`** (laptop only): run `pip install torch torchvision`.
- **MNIST download fails:** wait a minute and retry. The download servers are sometimes busy.
- **The analysis warns that task A was not learned well:** in `config.py`, change `"epochs_task_a": [1]` in `BASE` to `[2]`. Record the change in your log, and re-run the pilot.
- **Everything shows almost no forgetting:** increase `steps_task_b` or the learning rate in `config.py`, then re-run the pilot. Decide this **from the pilot, before** the main experiment, never after seeing main results.
- **You changed a setting but the results didn't change:** delete the old `results/<name>_summary.csv` first. The runner skips runs it thinks are already done.

---

## 11. After the main experiment

1. Look at `heatmap_forgetting.png` first. That's your central figure.
2. Check the trade-off figure `taskB_vs_memory.png`: did replay slow down learning B?
3. Write one paragraph per hypothesis: supported, refuted, or unclear, with numbers and CIs.
4. Optional extensions, *after* the main study is finished:
   - **Smarter memory:** keep the hardest examples instead of random ones.
   - **"Sleep-like" replay:** rehearse A in occasional bursts instead of every step.
   - **EWC** (Kirkpatrick et al., 2017): protect the weights that matter most for task A.
