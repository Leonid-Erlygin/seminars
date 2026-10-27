"""Build the sampling seminar's explanatory figures as PNG and vector PDF.

Run with Python, NumPy, SciPy, and Matplotlib installed. Outputs are written to
the ``assets`` directory beside this script. Every random example has a fixed
seed so that the figures can be regenerated exactly.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skoltech-sampling-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, norm


ASSETS = Path(__file__).resolve().parent / "assets"
BLUE = "#2369a1"
ORANGE = "#c77516"
GREEN = "#2b8868"
PURPLE = "#8b5da8"
GRAY = "#89929a"
INK = "#253340"
POSTERIOR = beta(3, 7)

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


def kernel(theta):
    theta = np.asarray(theta)
    return np.where((theta >= 0) & (theta <= 1), theta**2 * (1 - theta) ** 6, 0)


def inverse_cdf():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), layout="constrained")
    x = np.linspace(0, 2.5, 500)
    u = 0.7
    quantile = -np.log1p(-u) / 2
    ax = axes[0]
    ax.plot(x, -np.expm1(-2 * x), color=BLUE, label=r"$F(x)=1-e^{-2x}$")
    ax.plot([0, quantile, quantile], [u, u, 0], "--", color=ORANGE, lw=1.7)
    ax.scatter([quantile], [u], color=ORANGE, zorder=5)
    ax.annotate(r"$u=0.7$", (0.015, u), xytext=(0.09, 0.8))
    ax.text(quantile + 0.09, 0.12, r"$x=F^{-1}(u)$", color=ORANGE)
    ax.set(xlim=(0, 2.5), ylim=(0, 1.03), xlabel=r"$x$", ylabel=r"$F(x)$",
           title="Invert a cumulative probability")
    ax.legend(loc="lower right")

    rng = np.random.default_rng(2718)
    draws = -np.log1p(-rng.random(12000)) / 2
    ax = axes[1]
    ax.hist(draws, bins=np.linspace(0, 2.5, 36), density=True,
            color=BLUE, alpha=0.3, edgecolor="white", label="Transformed uniform draws")
    ax.plot(x, 2 * np.exp(-2 * x), color=BLUE, lw=2,
            label=r"$p(x)=2e^{-2x}$")
    ax.set(xlim=(0, 2.5), xlabel=r"$x$", ylabel="Density",
           title="The transform gives the target density")
    ax.legend(loc="upper right")
    save(fig, "inverse_cdf")


def monte_carlo_uncertainty():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), layout="constrained")
    rng = np.random.default_rng(314)
    means = rng.beta(3, 7, size=(5000, 100)).mean(axis=1)
    x = np.linspace(0, 1, 600)
    lo, hi = POSTERIOR.ppf([0.025, 0.975])
    axes[0].plot(x, POSTERIOR.pdf(x), color=BLUE, lw=2)
    axes[0].fill_between(x, POSTERIOR.pdf(x), where=(x >= lo) & (x <= hi),
                         color=BLUE, alpha=0.2)
    axes[0].axvline(0.3, color=INK, ls="--", lw=1)
    axes[0].text(0.53, 2.2, "Unknown coin bias\ngiven the observed data", fontsize=10)
    axes[0].set(xlim=(0, 1), ylim=(0, 3), xlabel=r"Coin bias $\theta$",
                ylabel="Posterior density", title="Posterior uncertainty")
    axes[0].text(0.98, 0.94, r"$\mathrm{SD}(\theta)\approx 0.138$",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=10)

    standard_error = np.sqrt(POSTERIOR.var() / 100)
    x_mean = np.linspace(0.24, 0.36, 500)
    axes[1].hist(means, bins=35, density=True, color=GREEN, alpha=0.3,
                 edgecolor="white", label="5,000 independent runs")
    axes[1].plot(x_mean, norm.pdf(x_mean, 0.3, standard_error), color=GREEN,
                 lw=2, label="Normal approximation")
    axes[1].axvline(0.3, color=INK, ls="--", lw=1, label="Exact mean")
    axes[1].set(xlim=(0.24, 0.36), ylim=(0, 40), xlabel=r"Estimate $\overline{\theta}_{100}$",
                ylabel="Density across runs", title="Monte Carlo uncertainty")
    axes[1].text(0.98, 0.95, "100 independent draws per run\n"
                 + r"$\mathrm{SE}(\overline{\theta}_{100})\approx 0.0138$",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=10)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.02, 0.72), fontsize=8)
    save(fig, "monte_carlo_uncertainty")


def rejection_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    rng = np.random.default_rng(1618)
    maximum = kernel(0.25)
    theta = rng.random(1100)
    ordinate = maximum * rng.random(theta.size)
    accept = ordinate <= kernel(theta)
    grid = np.linspace(0, 1, 600)
    ax = axes[0]
    ax.fill_between(grid, kernel(grid), color=BLUE, alpha=0.08)
    ax.scatter(theta[~accept], ordinate[~accept], s=7, color=GRAY, alpha=0.38,
               label="Rejected")
    ax.scatter(theta[accept], ordinate[accept], s=8, color=BLUE, alpha=0.65,
               label="Accepted")
    ax.plot(grid, kernel(grid), color=BLUE, lw=2,
            label=r"$\widetilde{\pi}(\theta)$")
    ax.axhline(maximum, color=ORANGE, lw=1.5, ls="--")
    ax.text(0.74, maximum * 0.91, r"$Mq(\theta)=M$", color=ORANGE, fontsize=10)
    ax.set(xlim=(0, 1), ylim=(0, maximum * 1.04), xlabel=r"Proposal $\theta$",
           ylabel=r"Vertical coordinate $y$", title="Uniform points under an envelope")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.54))

    axes[1].hist(theta[accept], bins=np.linspace(0, 1, 26), density=True,
                 color=BLUE, alpha=0.3, edgecolor="white", label="Accepted coordinates")
    axes[1].plot(grid, POSTERIOR.pdf(grid), color=BLUE, lw=2,
                 label=r"Target $\pi(\theta)$")
    axes[1].set(xlim=(0, 1), xlabel=r"Accepted $\theta$", ylabel="Density",
                title="Project accepted points onto the horizontal axis")
    axes[1].legend(loc="upper right")
    save(fig, "rejection_geometry")


def importance_weights():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.4), layout="constrained",
                              gridspec_kw={"height_ratios": [2, 1]})
    rng = np.random.default_rng(1732)
    grid = np.linspace(0.001, 0.999, 600)
    for column, (proposal, name, color) in enumerate(
            [(beta(1, 1), "Uniform proposal", ORANGE),
             (beta(2, 4), "Beta(2, 4) proposal", PURPLE)]):
        ax = axes[0, column]
        ax.plot(grid, POSTERIOR.pdf(grid), color=BLUE, lw=2, label=r"Target $\pi$")
        ax.plot(grid, proposal.pdf(grid), color=color, lw=2, ls="--",
                label=r"Proposal $q$")
        ax.set(xlim=(0, 1), ylim=(0, 3.2), ylabel="Density", title=name)
        ax.legend(loc="upper right")
        ax.tick_params(labelbottom=False)

        draws = proposal.rvs(90, random_state=rng)
        weights = kernel(draws) / proposal.pdf(draws)
        normalized = weights / weights.sum()
        ax = axes[1, column]
        # The same area scale in both panels makes weight concentration visible.
        ax.scatter(draws, rng.uniform(-0.45, 0.45, draws.size),
                    s=normalized * 6500, alpha=0.52, color=color,
                    edgecolor="white", linewidth=0.4)
        ax.set(xlim=(0, 1), ylim=(-0.7, 0.7), xlabel=r"Proposal draw $\theta_i$",
                yticks=[], ylabel="Weighted draws")
        ax.grid(False, axis="y")
    fig.supxlabel(r"Marker area is proportional to $\overline{w}_i$; vertical jitter only separates points.",
                   fontsize=10)
    save(fig, "importance_weights")


def metropolis_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), layout="constrained")
    grid = np.linspace(0, 1, 600)
    current, proposed, scale = 0.25, 0.55, 0.15
    ax = axes[0]
    ax.plot(grid, POSTERIOR.pdf(grid), color=BLUE, lw=2, label=r"Target $\pi$")
    ax.plot(grid, norm.pdf(grid, current, scale), color=ORANGE, ls="--", lw=1.5,
            label=r"Forward $q(z\mid x)$")
    ax.plot(grid, norm.pdf(grid, proposed, scale), color=PURPLE, ls=":", lw=2,
            label=r"Reverse $q(z\mid x')$")
    transition_density = norm.pdf(proposed, current, scale)
    ax.plot([current, proposed], [transition_density] * 2, color=INK, lw=1)
    ax.scatter([current, proposed], [transition_density] * 2, color=INK, s=25, zorder=6)
    for point in (current, proposed):
        ax.plot([point, point], [0, POSTERIOR.pdf(point)], color=BLUE, lw=1, alpha=0.6)
        ax.scatter([point], [POSTERIOR.pdf(point)], color=BLUE, s=30, zorder=5)
    ax.set_xticks([0, current, proposed, 1], ["0", r"$x=0.25$", r"$x'=0.55$", "1"])
    ax.annotate("Equal proposal densities", xy=(0.4, transition_density),
                 xytext=(0.45, 1.25), fontsize=9,
                 arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(xlim=(0, 1), ylim=(0, 3.55), xlabel="State / proposal coordinate",
            ylabel="Density", title="A symmetric random-walk proposal")
    ax.legend(loc="upper right", fontsize=8)

    rng = np.random.default_rng(199)
    states = np.empty(56)
    states[0] = 0.3
    rejected = np.zeros(states.size, dtype=bool)
    for t in range(1, len(states)):
        candidate = states[t - 1] + rng.normal(scale=0.15)
        ratio = kernel(candidate) / kernel(states[t - 1])
        accepted = rng.random() < min(1, ratio)
        states[t] = candidate if accepted else states[t - 1]
        rejected[t] = not accepted
    iteration = np.arange(states.size)
    axes[1].step(iteration, states, where="post", color=BLUE, lw=1.5)
    axes[1].scatter(iteration, states, color=BLUE, s=15, zorder=5)
    axes[1].scatter(iteration[rejected], states[rejected], color=ORANGE, s=32,
                     marker="s", zorder=6, label="Retained state after rejection")
    axes[1].set(xlim=(-1, 56), ylim=(0, 0.85), xlabel=r"Iteration $t$",
                ylabel=r"State $\theta_t$", title="The chain includes every repeated state")
    axes[1].legend(loc="upper left", fontsize=8)
    save(fig, "metropolis_geometry")


def mixture_logpdf(x):
    return logsumexp([norm.logpdf(x, -4, 0.6), norm.logpdf(x, 4, 0.6)], axis=0) - np.log(2)


def mixture_chain(rng, initial, n, independence):
    chain = np.empty(n)
    chain[0] = initial
    for i in range(1, n):
        current = chain[i - 1]
        candidate = rng.normal(0, 5) if independence else current + rng.normal(0, 0.4)
        log_ratio = mixture_logpdf(candidate) - mixture_logpdf(current)
        if independence:
            # Independence proposals are asymmetric: q(current)/q(candidate).
            log_ratio += norm.logpdf(current, 0, 5) - norm.logpdf(candidate, 0, 5)
        chain[i] = candidate if np.log(rng.random()) < min(0, log_ratio) else current
    return chain


def multimodal_mixing():
    fig = plt.figure(figsize=(9.6, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.6])
    density_ax = fig.add_subplot(gs[0, :])
    left_ax = fig.add_subplot(gs[1, 0])
    right_ax = fig.add_subplot(gs[1, 1], sharey=left_ax)
    grid = np.linspace(-7, 7, 900)
    density_ax.plot(grid, np.exp(mixture_logpdf(grid)), color=BLUE, lw=2,
                     label=r"Target: $\frac{1}{2}\mathcal{N}(-4,0.6^2)+\frac{1}{2}\mathcal{N}(4,0.6^2)$")
    density_ax.plot(grid, norm.pdf(grid, 0, 5), color=PURPLE, ls="--", lw=1.6,
                     label=r"Independence proposal: $q=\mathcal{N}(0,5^2)$")
    density_ax.set(xlim=(-7, 7), ylim=(0, 0.4), xlabel="State", ylabel="Density",
                   title="Separated modes can conceal poor exploration")
    density_ax.legend(loc="upper center", fontsize=9)
    rng = np.random.default_rng(14142)
    for ax, independence, title in [
            (left_ax, False, r"Local random walk: proposal SD $0.4$"),
            (right_ax, True, "Broad independence Metropolis–Hastings")]:
        for initial, color in [(-4, BLUE), (4, ORANGE)]:
            chain = mixture_chain(rng, initial, 450, independence)
            ax.plot(np.arange(chain.size), chain, color=color, lw=0.8, alpha=0.8,
                    label=f"Start at {initial:+d}")
        ax.axhline(0, color=GRAY, ls=":", lw=0.8)
        ax.set(xlim=(0, 449), ylim=(-6.5, 6.5), xlabel=r"Iteration $t$",
                title=title)
        ax.set_xticks([0, 100, 200, 300, 400])
    left_ax.set_ylabel(r"State $\theta_t$")
    left_ax.legend(loc="center right", fontsize=9)
    right_ax.tick_params(labelleft=False)
    save(fig, "multimodal_mixing")


if __name__ == "__main__":
    for build in (inverse_cdf, monte_carlo_uncertainty, rejection_geometry,
                  importance_weights, metropolis_geometry, multimodal_mixing):
        build()
        print(f"Created {build.__name__}.png and .pdf")
