"""
test_everything.py: checks that each part of the code does what it claims.

USAGE:
    python test_everything.py

It uses small fake data, so it runs in about a minute with no download.
Every test prints PASS or FAIL. Run this after ANY change to the code.
If a test fails, fix it before running experiments - a bug that silently
changes results is the worst thing that can happen in research.
"""

import torch
import torch.nn as nn

from config import BASE
from data import build_tasks, make_fake_data, rotate
from experiment import run_one
from model import make_model
from training import accuracy, train_task_a

results = []


def check(name, condition, detail=""):
    results.append(condition)
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


# Tiny settings so the tests are quick (3 epochs on A: the fake data is small)
CFG = dict(BASE, steps_task_b=300, eval_every=100)
data = make_fake_data()


# ---- 1. Data has the right shape and range ----
x_train, y_train, x_test, y_test = data
check("images have shape (N, 1, 28, 28)", tuple(x_train.shape[1:]) == (1, 28, 28))
check("pixel values are between 0 and 1", x_train.min() >= 0 and x_train.max() <= 1)
check("labels are digits 0-9", set(y_train.tolist()) <= set(range(10)))

# ---- 2. Rotation does what we expect ----
check("rotating by 0 degrees changes nothing", torch.equal(rotate(x_test, 0), x_test))
rotated = rotate(x_test, 45)
check("rotating keeps the shape", rotated.shape == x_test.shape)
check("rotating by 45 degrees changes the images", not torch.allclose(rotated, x_test))
task_a, task_b = build_tasks(data, 90)
check("tasks A and B share the same labels", torch.equal(task_a["y_train"], task_b["y_train"]))

# ---- 3. The accuracy function is correct on a case we can work out by hand ----
class AlwaysThree(nn.Module):
    """A fake 'network' that always guesses the digit 3."""
    def forward(self, x):
        scores = torch.zeros(len(x), 10)
        scores[:, 3] = 1.0
        return scores

expected = (y_test == 3).float().mean().item()
check("accuracy of 'always guess 3' = fraction of 3s",
      abs(accuracy(AlwaysThree(), x_test, y_test) - expected) < 1e-6,
      f"expected {expected:.3f}")

# ---- 4. An untrained network is guessing; a trained one is not ----
torch.manual_seed(0)
model = make_model(100)
before = accuracy(model, x_test, y_test)
check("untrained network is near chance (10%)", 0.0 <= before <= 0.3, f"{before:.3f}")
train_task_a(model, task_a, epochs=3, batch_size=64, lr=0.1,
             generator=torch.Generator().manual_seed(0))
after = accuracy(model, x_test, y_test)
check("training on task A raises accuracy above 80%", after > 0.8, f"{after:.3f}")

# ---- 5. Same seed -> identical results (reproducibility) ----
s1, _ = run_one(task_a, task_b, 100, 3, 100, seed=7, cfg=CFG)
s2, _ = run_one(task_a, task_b, 100, 3, 100, seed=7, cfg=CFG)
check("same seed gives identical results",
      s1["acc_a_after"] == s2["acc_a_after"] and s1["acc_b_after"] == s2["acc_b_after"])
# Different seeds must give different networks. We compare the WEIGHTS, not
# the accuracy: two different networks can both score exactly 100% on easy
# data, so equal accuracy would not prove the seeds were ignored.
def trained_weights(seed):
    torch.manual_seed(seed)
    net = make_model(100)
    start = [p.detach().clone() for p in net.parameters()]
    train_task_a(net, task_a, epochs=1, batch_size=64, lr=0.1,
                 generator=torch.Generator().manual_seed(seed))
    return start, [p.detach().clone() for p in net.parameters()]

start7, end7 = trained_weights(7)
start8, end8 = trained_weights(8)
check("different seeds give different starting weights",
      not all(torch.equal(a, b) for a, b in zip(start7, start8)))
check("different seeds give different trained networks",
      not all(torch.equal(a, b) for a, b in zip(end7, end8)))

# ---- 6. Paired design: all memory sizes share the same task-A training ----
s0, _ = run_one(task_a, task_b, 0, 3, 100, seed=7, cfg=CFG)
check("task-A accuracy before switching is identical across memory sizes",
      s0["acc_a_before"] == s1["acc_a_before"])

# ---- 7. Forgetting = before - after, and the curve has the right length ----
check("forgetting is calculated as before - after",
      abs(s1["forgetting"] - (s1["acc_a_before"] - s1["acc_a_after"])) < 1e-12)
_, curve = run_one(task_a, task_b, 0, 3, 100, seed=7, cfg=CFG)
check("learning curve has one point per evaluation (+ the start)",
      len(curve) == CFG["steps_task_b"] // CFG["eval_every"] + 1, f"{len(curve)} points")

# ---- 8. The effect we study is visible, even on fake data ----
no_replay, _ = run_one(task_a, task_b, 0, 3, 100, seed=0, cfg=CFG)
big_replay, _ = run_one(task_a, task_b, 2000, 3, 100, seed=0, cfg=CFG)
check("without replay, task A is forgotten", no_replay["forgetting"] > 0.1,
      f"forgetting {no_replay['forgetting']:.3f}")
check("with lots of replay, forgetting is smaller",
      big_replay["forgetting"] < no_replay["forgetting"],
      f"{big_replay['forgetting']:.3f} vs {no_replay['forgetting']:.3f}")

print(f"\n{sum(results)}/{len(results)} tests passed.")
if not all(results):
    raise SystemExit("Some tests FAILED - fix them before running experiments.")
