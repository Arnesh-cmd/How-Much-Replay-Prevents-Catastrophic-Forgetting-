"""
analyze.py: turns the CSV results into tables, statistics and figures.

USAGE:
    python analyze.py main          # or pilot, part1, width
    python analyze.py pilot --fake  # analyse a fake-data test run

WHAT IT PRODUCES (in results/figures/ and printed on screen):
    - A table of mean forgetting with 95% confidence intervals per condition
    - Paired statistical tests: does replay reduce forgetting vs. no replay?
    - Figures: heatmap, forgetting vs memory size, learning curves, trade-off

KEY STATISTICS IDEAS (read GUIDE.md for the longer explanation):
    Mean: the average over seeds.
    95% bootstrap confidence interval: resample the seeds thousands of times
        and see how much the mean jumps around. If you repeated the whole
        experiment, the true average would usually land inside this range.
    Paired comparison: the SAME seed is used for every condition, so we
        compare condition X vs no-replay seed by seed. This removes
        "lucky seed" noise.
    Holm correction: when you run many tests, some look significant by pure
        luck. Holm adjusts the p-values to account for that.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw figures to files (works without a screen)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

# ---- Colours: a colourblind-checked palette, used in a fixed order ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # shape as well as colour
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Diverging scale for forgetting: grey = no change, red = forgot, blue = improved.
DIVERGING = LinearSegmentedColormap.from_list(
    "forgetting", ["#184f95", "#86b6ef", "#f0efec", "#f0a39f", "#b8302f"])

VARIABLES = ["angle", "memory_size", "epochs_task_a", "hidden_width"]
LABELS = {
    "angle": "Rotation of task B (degrees)",
    "memory_size": "Replay memory (old images kept)",
    "epochs_task_a": "Epochs of practice on task A",
    "hidden_width": "Neurons per hidden layer",
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=10000, seed=0):
    """Mean and 95% bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resampled_means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(resampled_means, [2.5, 97.5])
    return values.mean(), low, high


def summarize(df, by, column):
    rows = []
    for keys, group in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mean, low, high = bootstrap_ci(group[column])
        rows.append({**dict(zip(by, keys)), "n": len(group),
                     "mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def holm(p_values):
    """Holm-Bonferroni correction for running many tests at once."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    order = np.argsort(np.where(valid, p, np.inf))
    m = valid.sum()
    running_max = 0.0
    for rank, i in enumerate(order[:m]):
        running_max = max(running_max, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running_max
    return adjusted


def paired_tests(df):
    """
    For every angle and every memory size > 0, compare against memory = 0
    using the same seeds. Positive 'reduction' = replay reduced forgetting.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    rows = []
    for angle, at_angle in df.groupby("angle"):
        baseline = at_angle[at_angle.memory_size == 0].set_index("seed")["forgetting"]
        for mem in sorted(at_angle.memory_size.unique()):
            if mem == 0:
                continue
            other = at_angle[at_angle.memory_size == mem].set_index("seed")["forgetting"]
            seeds = baseline.index.intersection(other.index)
            diffs = (baseline[seeds] - other[seeds]).values
            mean, low, high = bootstrap_ci(diffs)
            p = np.nan
            if wilcoxon is not None and len(diffs) >= 6 and np.any(diffs != 0):
                p = wilcoxon(diffs).pvalue
            rows.append({"angle": angle, "memory_size": mem, "n_pairs": len(diffs),
                         "reduction": mean, "ci_low": low, "ci_high": high, "p": p})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm(table["p"].values)
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_figure(width=8, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def plot_heatmap(df, out_dir):
    table = summarize(df, ["angle", "memory_size"], "forgetting")
    grid = table.pivot(index="angle", columns="memory_size", values="mean")
    values = grid.values
    # Centre the colours on 0 so grey always means "no change".
    limit = max(0.01, np.nanmax(np.abs(values)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, ax = _new_figure(7, 4.8)
    image = ax.imshow(values, cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{a}°" for a in grid.index])
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            colour = "white" if abs(value) > 0.6 * limit else INK
            ax.text(c, r, f"{value:+.2f}", ha="center", va="center", color=colour, fontsize=9)
    _style(ax, "Forgetting of task A (mean over seeds)",
           LABELS["memory_size"], LABELS["angle"])
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax)
    bar.set_label("Accuracy lost on A  (red = forgot, blue = improved)", color=INK2)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap_forgetting.png"), dpi=150)
    plt.close(fig)


def _categorical_axis(values):
    """Memory sizes like 0, 10, 100, 10000 are spread evenly, labelled honestly."""
    values = sorted(values)
    return values, {v: i for i, v in enumerate(values)}


def plot_lines(df, column, ylabel, title, filename, out_dir):
    """`column` vs memory size, one line per angle, with 95% CI bands."""
    table = summarize(df, ["angle", "memory_size"], column)
    sizes, pos = _categorical_axis(table.memory_size.unique())
    fig, ax = _new_figure()
    for i, (angle, rows) in enumerate(table.groupby("angle")):
        rows = rows.sort_values("memory_size")
        x = [pos[m] for m in rows.memory_size]
        colour = SERIES[i % len(SERIES)]
        ax.fill_between(x, rows.ci_low, rows.ci_high, color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, rows["mean"], color=colour, linewidth=2, marker=MARKERS[i % len(MARKERS)],
                markersize=7, label=f"{angle}°")
    ax.set_xticks(range(len(sizes)), [str(s) for s in sizes])
    _style(ax, title, LABELS["memory_size"], ylabel)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.legend(title="Rotation", frameon=False, labelcolor=INK2,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)


def plot_curves(curves, angle, out_dir):
    """Accuracy on task A DURING task-B training, one line per memory size."""
    at_angle = curves[curves.angle == angle]
    if at_angle.empty:
        return
    fig, ax = _new_figure()
    for i, (mem, rows) in enumerate(at_angle.groupby("memory_size")):
        stats = summarize(rows, ["step"], "acc_a").sort_values("step")
        colour = SERIES[i % len(SERIES)]
        ax.fill_between(stats.step, stats.ci_low, stats.ci_high, color=colour, alpha=0.15, linewidth=0)
        ax.plot(stats.step, stats["mean"], color=colour, linewidth=2, label=f"{mem} images")
    _style(ax, f"Task A accuracy while learning task B (rotation {angle}°)",
           "Training steps on task B", "Accuracy on task A")
    ax.set_ylim(0, 1)
    ax.legend(title="Replay memory", frameon=False, labelcolor=INK2,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"curves_angle{angle}.png"), dpi=150)
    plt.close(fig)


def plot_single_variable(df, variable, out_dir):
    """For part1 / width: forgetting and task-B accuracy vs one variable."""
    for column, ylabel in [("forgetting", "Forgetting (accuracy lost on A)"),
                           ("acc_b_after", "Final accuracy on task B")]:
        table = summarize(df, [variable], column).sort_values(variable)
        values, pos = _categorical_axis(table[variable].unique())
        x = [pos[v] for v in table[variable]]
        fig, ax = _new_figure(6, 4)
        ax.errorbar(x, table["mean"],
                    yerr=[table["mean"] - table.ci_low, table.ci_high - table["mean"]],
                    color=SERIES[0], linewidth=2, marker="o", markersize=8, capsize=4)
        ax.set_xticks(range(len(values)), [str(v) for v in values])
        _style(ax, f"{ylabel} vs {LABELS[variable].lower()}", LABELS[variable], ylabel)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{column}_vs_{variable}.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze(name, results_dir="results"):
    summary_path = os.path.join(results_dir, f"{name}_summary.csv")
    if not os.path.exists(summary_path):
        print(f"No results found at {summary_path}. Run experiment.py first.")
        return
    df = pd.read_csv(summary_path)
    curves = pd.read_csv(os.path.join(results_dir, f"{name}_curves.csv"))
    out_dir = os.path.join(results_dir, "figures", name)
    os.makedirs(out_dir, exist_ok=True)
    pd.set_option("display.width", 120)

    varied = [v for v in VARIABLES if df[v].nunique() > 1]
    print(f"Loaded {len(df)} runs. Variables that change: {varied}\n")

    # Sanity check first: did task A get learned at all? If not, "forgetting"
    # means nothing - you can't forget what you never learned.
    print(f"Task A accuracy before switching: mean {df.acc_a_before.mean():.3f} "
          f"(min {df.acc_a_before.min():.3f})")
    if df.acc_a_before.min() < 0.9:
        print("WARNING: task A was not learned well (below 90%) in some runs.\n"
              "         Forgetting numbers are not meaningful until this is fixed\n"
              "         (e.g. more epochs on task A). Do not report these results.")
    print()

    if not varied:
        print("Nothing varies in this experiment, so there is nothing to compare.")
        return

    print("=== Forgetting (accuracy lost on task A), mean and 95% CI ===")
    print(summarize(df, varied, "forgetting").round(3).to_string(index=False), "\n")
    print("=== Final accuracy on task B, mean and 95% CI ===")
    print(summarize(df, varied, "acc_b_after").round(3).to_string(index=False), "\n")

    if "angle" in varied and "memory_size" in varied:
        tests = paired_tests(df)
        print("=== Paired tests: forgetting reduction vs no replay (same seeds) ===")
        print("reduction > 0 means replay reduced forgetting; p_holm < 0.05 = significant")
        print(tests.round(4).to_string(index=False), "\n")
        tests.to_csv(os.path.join(results_dir, f"{name}_paired_tests.csv"), index=False)

        plot_heatmap(df, out_dir)
        plot_lines(df, "forgetting", "Forgetting (accuracy lost on A)",
                   "Forgetting vs replay memory", "forgetting_vs_memory.png", out_dir)
        plot_lines(df, "acc_b_after", "Final accuracy on task B",
                   "Does replay slow down learning task B?", "taskB_vs_memory.png", out_dir)
        for angle in sorted(df.angle.unique()):
            plot_curves(curves, angle, out_dir)
    else:
        for variable in varied:
            plot_single_variable(df, variable, out_dir)

    print(f"Figures saved in {out_dir}/")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python analyze.py <pilot|main|part1|width> [--fake]")
        sys.exit(1)
    analyze(args[0] + ("_fake" if "--fake" in args else ""))
