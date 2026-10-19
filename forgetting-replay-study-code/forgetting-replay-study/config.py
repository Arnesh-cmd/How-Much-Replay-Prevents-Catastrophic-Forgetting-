"""
config.py: every setting for every experiment lives in this one file.

WHY A CONFIG FILE?
    Science has to be repeatable. If a setting is hidden inside the code, you
    (or a reviewer) can't tell what produced a result. So: never change numbers
    inside the other files. Change them here, and the setting is saved
    automatically next to your results.

HOW THE EXPERIMENTS ARE DEFINED
    Each experiment is a "grid": a list of values for each of the four things
    we can vary. The runner tries EVERY combination, once per random seed.

        angles         how much task B's digits are rotated (task similarity)
        memory_sizes   how many old task-A images are kept for replay
        epochs_task_a  how many passes through task A before switching
        hidden_widths  how many neurons in each hidden layer
"""

# ---------------------------------------------------------------------------
# Settings shared by every experiment
# ---------------------------------------------------------------------------
BASE = {
    "data_dir": "data",        # where MNIST is downloaded
    "results_dir": "results",  # where CSV files and figures go

    # Network training
    "learning_rate": 0.1,      # step size for nudging the weights
    "batch_size": 64,          # images per training step

    # Task B budget. Every condition gets EXACTLY the same number of steps on
    # task B, so no condition wins just by training longer.
    "steps_task_b": 1000,

    # How often (in task-B steps) to measure accuracy on both tasks.
    # This gives the learning curves.
    "eval_every": 100,

    # Default values (each experiment below overrides some of these lists)
    "angles": [60],
    "memory_sizes": [0],
    "epochs_task_a": [1],
    "hidden_widths": [100],
    "seeds": list(range(10)),  # 10 independent repeats of every condition
}

# ---------------------------------------------------------------------------
# The experiments
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    # PILOT: a small, fast version. Run this FIRST to check that forgetting
    # actually happens and that nothing is broken. About 2-5 minutes.
    "pilot": {
        **BASE,
        "angles": [0, 45, 90],
        "memory_sizes": [0, 100, 10000],
        "seeds": [0, 1, 2],
    },

    # MAIN: the core question. How much replay is needed, and does it depend
    # on how similar the tasks are? 6 angles x 5 memory sizes x 10 seeds =
    # 300 runs. Roughly 20-40 minutes on Colab's free CPU.
    "main": {
        **BASE,
        "angles": [0, 15, 30, 45, 60, 90],
        "memory_sizes": [0, 10, 100, 1000, 10000],
    },

    # PART 1: does practising task A for longer protect it? No replay here,
    # so the only thing that changes is how long task A was trained.
    # 40 runs, about 10-15 minutes (the 10-epoch runs are the slow ones).
    "part1": {
        **BASE,
        "angles": [60],
        "memory_sizes": [0],
        "epochs_task_a": [1, 2, 5, 10],
    },

    # WIDTH: do wider networks forget less? (Hypothesis H3). About 5 minutes.
    "width": {
        **BASE,
        "angles": [60],
        "memory_sizes": [0],
        "hidden_widths": [25, 100, 400],
    },
}
