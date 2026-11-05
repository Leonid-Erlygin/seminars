"""Build the BNN seminar's explanatory figures as PNG and vector PDF.

Run with Python, NumPy, and Matplotlib installed. Analytical examples and
fixed grids keep the figures reproducible without remote assets or training.
The source uncertainty notebook and original active-learning notebook are
read-only references; this script only writes its own assets directory.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skoltech-bnn-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np


ASSETS = Path(__file__).resolve().parent / "assets"
BLUE = "#2369a1"
ORANGE = "#c77516"
GREEN = "#2b8868"
GRAY = "#89929a"
INK = "#253340"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.titleweight": "normal",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#aab2b9",
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(ASSETS / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def interval_reference(x, intervals):
    """Conditional eta for a population that gives each interval equal mass.

    The observed training design uses deterministic grids. This is a separate,
    explicitly specified reference population. Interval endpoints have zero
    probability, and eta is left undefined where the population has no support.
    """
    rates = np.zeros((2, len(x)))
    for label, regions in intervals.items():
        for lower, upper in regions:
            rates[label] += ((x > lower) & (x < upper)) / (upper-lower)
    total = rates.sum(axis=0)
    eta = np.full(len(x), np.nan)
    np.divide(rates[1], total, out=eta, where=total > 0)
    return eta, total > 0


def shade_unsupported(ax, x, supported):
    """Shade whole unsupported grid runs, rather than a probability estimate."""
    edges = np.diff(np.r_[False, ~supported, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops):
        ax.axvspan(x[start], x[min(stop, len(x)-1)], color=GRAY, alpha=0.12, lw=0)


def interval_geometry():
    scenarios = [
        ("Original four intervals", {0: [(-5, -3), (1, 2.5)],
                                     1: [(-3, -2), (2.5, 5)]}),
        ("Overlapping classes", {0: [(-2, 2)], 1: [(0, 4)]}),
        ("Separated classes", {0: [(-5, -3)], 1: [(3, 5)]}),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 4.8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1.5]})
    grid = np.linspace(-5.4, 5.4, 5401)
    for column, (title, intervals) in enumerate(scenarios):
        upper, lower = axes[:, column]
        eta, supported = interval_reference(grid, intervals)
        for ax in (upper, lower):
            shade_unsupported(ax, grid, supported)
            ax.set_xlim(-5.4, 5.4)
            ax.set_xticks([-5, -3, 0, 3, 5])
        for label, regions in intervals.items():
            color = BLUE if label == 0 else ORANGE
            for left, right in regions:
                points = np.linspace(left, right, 30)
                upper.plot([left, right], [label, label], color=color, lw=5, alpha=0.22)
                upper.scatter(points, np.full(30, label), color=color, s=8, zorder=3)
        upper.set(title=title, ylim=(-0.35, 1.35), yticks=[0, 1])
        upper.tick_params(labelbottom=False)
        if column == 0:
            for boundary in [-3, 2.5]:
                upper.axvline(boundary, color=INK, ls=":", lw=0.9, alpha=0.6)
        lower.plot(grid, eta, color=GREEN, lw=2)
        lower.set(ylim=(-0.08, 1.08), yticks=[0, 0.5, 1], xlabel=r"Input $x$")
        if column == 1:
            lower.text(1, 0.62, r"$\eta(x)=1/2$", ha="center", color=GREEN, fontsize=10)
        if column == 2:
            lower.text(0, 0.52, "No population support:\nreference is undefined", ha="center", fontsize=9)
    axes[0, 0].set_ylabel("Observed label")
    axes[1, 0].set_ylabel(r"Reference $\eta(x)=P(Y=1\mid x)$")
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    fig.text(0.5, -0.035,
             "Reference population: equal probability per interval, uniform within each interval. "
             "Gray marks unsupported inputs.", ha="center", fontsize=10)
    fig.text(0.5, -0.085,
             "The original 120-point grid includes opposite labels at the shared endpoints "
             r"$x=-3$ and $x=2.5$; these have zero population mass.", ha="center", fontsize=10)
    save(fig, "interval_geometry")


def binary_entropy(probability):
    probability = np.asarray(probability)
    return -(probability*np.log(np.clip(probability, 1e-300, 1))
             +(1-probability)*np.log(np.clip(1-probability, 1e-300, 1)))


def uncertainty_decomposition():
    examples = [
        ("Confident agreement", np.array([0.04, 0.04, 0.04, 0.04])),
        ("Uncertain agreement", np.array([0.5, 0.5, 0.5, 0.5])),
        ("Confident disagreement", np.array([0.02, 0.98, 0.02, 0.98])),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.2), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for column, (title, p1) in enumerate(examples):
        top, bottom = axes[:, column]
        mean = p1.mean()
        total = binary_entropy(mean)
        expected = binary_entropy(p1).mean()
        mutual_information = total-expected
        top.scatter(np.arange(1, 5), p1, s=45, color=INK, zorder=4)
        top.axhline(mean, color=GREEN, ls="--", lw=1.4)
        top.set(title=title, xlim=(0.5, 4.5), ylim=(-0.06, 1.06),
                yticks=[0, 0.5, 1], xticks=[1, 2, 3, 4], xlabel="Model draw")
        top.text(0.98, 0.50 if column == 0 else 0.18, f"Mean = {mean:.2f}",
                  color=GREEN, ha="right", transform=top.transAxes, fontsize=10)
        bottom.bar([0], [expected], width=0.58, color=BLUE, alpha=0.85)
        bottom.bar([0], [mutual_information], bottom=[expected], width=0.58,
                    color=ORANGE, alpha=0.85)
        bottom.axhline(np.log(2), color=GRAY, ls=":", lw=0.9)
        bottom.text(0, total+0.035, f"Total = {total:.3f}", ha="center", fontsize=10)
        bottom.set(xlim=(-0.8, 0.8), ylim=(0, 0.86), xticks=[], yticks=[0, 0.35, np.log(2)])
        bottom.set_yticklabels(["0", "0.35", r"$\log 2$"])
        bottom.text(0, 0.79, f"MI = {mutual_information:.3f}", ha="center", fontsize=10)
    axes[0, 0].set_ylabel(r"$P(Y=1\mid x,w)$")
    axes[1, 0].set_ylabel("Entropy (nats)")
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    fig.legend(handles=[Patch(facecolor=BLUE, alpha=0.85, label="Expected conditional entropy"),
                        Patch(facecolor=ORANGE, alpha=0.85, label="Mutual information / BALD")],
               loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=2)
    fig.text(0.5, -0.10,
             r"$H(\overline{p})=\mathbb{E}_q H(p_w)+I_q(Y;W\mid x)$: "
             "the last two cases have identical average predictions.", ha="center", fontsize=10)
    fig.text(0.5, -0.15,
             "This is a decomposition under the chosen model distribution q, "
             "not an automatic measurement of true data noise.", ha="center", fontsize=10)
    save(fig, "uncertainty_decomposition")


def cycle_box(ax, x, y, label, color):
    ax.add_patch(FancyBboxPatch((x-0.13, y-0.09), 0.26, 0.18,
                  boxstyle="round,pad=0.012,rounding_size=0.025",
                  facecolor="white", edgecolor=color, linewidth=1.4))
    ax.text(x, y, label, ha="center", va="center", fontsize=11, color=color)


def active_learning_cycle():
    fig, ax = plt.subplots(figsize=(10.4, 5.1), constrained_layout=True)
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    boxes = [(0.17, 0.74, "Current labeled\nset of IDs", BLUE),
             (0.50, 0.74, "Fit an MC-dropout\nclassification model", BLUE),
             (0.83, 0.74, "Score pool features;\nselect unique IDs", BLUE),
             (0.83, 0.31, "Oracle reveals labels\nfor selected IDs", ORANGE),
             (0.50, 0.31, "Move queried IDs:\npool → labeled", ORANGE),
             (0.17, 0.31, "Check remaining\ntraining-label budget", ORANGE)]
    for x, y, label, color in boxes:
        cycle_box(ax, x, y, label, color)
    segments = [((0.315, 0.74), (0.355, 0.74), BLUE),
                ((0.645, 0.74), (0.685, 0.74), BLUE),
                ((0.83, 0.63), (0.83, 0.42), ORANGE),
                ((0.685, 0.31), (0.645, 0.31), ORANGE),
                ((0.355, 0.31), (0.315, 0.31), ORANGE),
                ((0.17, 0.42), (0.17, 0.63), BLUE)]
    for start, end, color in segments:
        ax.annotate("", xy=end, xytext=start,
                     arrowprops={"arrowstyle": "->", "lw": 1.8, "color": color})
    ax.text(0.5, 0.97,
             "Acquisition receives inputs and pool IDs; labels stay hidden until the oracle step.",
             ha="center", va="top", fontsize=11)
    ax.text(0.54, 0.51,
             r"BALD score: $I_q(Y;W\mid x,\mathcal{D})$" + "\n"
             "Random acquisition is the comparison under the same label budget.",
             ha="center", va="center", fontsize=10)
    ax.text(0.5, 0.07,
             "Validation monitors learning. Evaluate the test set after the budget is spent.\n"
             "Validation labels are a separate resource from the reported training-label budget.",
             ha="center", va="center", fontsize=10)
    save(fig, "active_learning_cycle")


if __name__ == "__main__":
    for build in (interval_geometry, uncertainty_decomposition, active_learning_cycle):
        build()
        print(f"Created {build.__name__}.png and .pdf")
