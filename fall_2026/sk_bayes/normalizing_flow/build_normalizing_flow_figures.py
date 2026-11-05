"""Build the normalizing-flow seminar's explanatory PNG and vector PDF figures.

Run with Python, NumPy, and Matplotlib installed. Outputs are written to the
``assets`` directory beside this script. The figures use analytical transforms
and fixed grids, so no data files, downloads, or trained models are needed.

Convention: g maps a base variable z into a data variable x; f = g^{-1}.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skoltech-normalizing-flow-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
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


def normal_pdf(z):
    return np.exp(-0.5 * np.asarray(z) ** 2) / np.sqrt(2 * np.pi)


def change_of_variables():
    """The same shaded probability has twice the width and half the height."""
    mu, sigma = 1.0, 2.0
    z = np.linspace(-4, 4, 1001)
    x = mu + sigma * z
    probability = 0.38292492254802624  # P(-0.5 <= Z <= 0.5).
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)

    for ax, grid, density, interval, center, width, color, title, label in (
        (axes[0], z, normal_pdf(z), (-0.5, 0.5), 0.0, 1.0, BLUE,
         r"Base: $z\sim\mathcal{N}(0,1)$", r"$z$"),
        (axes[1], x, normal_pdf(z) / sigma, (0.0, 2.0), mu, sigma, ORANGE,
         r"Transform: $x=g(z)=1+2z$", r"$x$"),
    ):
        selected = (grid >= interval[0]) & (grid <= interval[1])
        ax.plot(grid, density, color=color, lw=2.2)
        ax.fill_between(grid, density, where=selected, color=color, alpha=0.26)
        for endpoint in interval:
            height = np.interp(endpoint, grid, density)
            ax.plot([endpoint, endpoint], [0, height], color=color, lw=1.0)
        ax.scatter([center], [density.max()], color=color, s=28, zorder=5)
        ax.annotate(
            "", xy=(interval[0], 0.043), xytext=(interval[1], 0.043),
            arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.2},
        )
        ax.text(center, 0.055, f"width {width:g}", ha="center", fontsize=9)
        ax.text(0.97, 0.93, f"Shaded probability\n{probability:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10)
        # A common coordinate scale makes the doubled width visually apparent.
        ax.set(xlim=(-5, 7), ylim=(0, 0.46),
               xlabel=label, ylabel="Density", title=title)

    axes[0].text(-4.7, 0.30, r"$p_0(0)\approx0.399$", color=BLUE, fontsize=10)
    axes[1].text(-4.7, 0.30, r"$q(1)\approx0.199$", color=ORANGE, fontsize=10)
    for ax in axes:
        ax.set_xticks([-4, -2, 0, 2, 4, 6])
    fig.text(
        0.5, -0.035,
        r"$q(x)=p_0((x-1)/2)/2$: stretching an interval by 2 divides its density by 2.",
        fontsize=11, ha="center",
    )
    save(fig, "change_of_variables")


def coupling(z):
    """A 2D affine coupling whose first coordinate is unchanged."""
    z = np.asarray(z)
    first = z[..., 0]
    scale = 0.45 * np.tanh(first)
    shift = 0.65 * np.sin(2 * first)
    return np.stack((first, z[..., 1] * np.exp(scale) + shift), axis=-1)


def coupling_geometry():
    """A grid makes conditional vertical scaling and the triangular map visible."""
    fig = plt.figure(figsize=(9.6, 6.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[4.5, 1.15])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    coordinates = np.linspace(-2, 2, 9)
    dense = np.linspace(-2, 2, 301)
    for ax, transform in zip(axes, (lambda z: z, coupling)):
        for value in coordinates:
            for line in (
                np.column_stack((np.full_like(dense, value), dense)),
                np.column_stack((dense, np.full_like(dense, value))),
            ):
                mapped = transform(line)
                ax.plot(mapped[:, 0], mapped[:, 1], color=BLUE, lw=0.9, alpha=0.65)
        ax.set(xlim=(-2.35, 2.35), ylim=(-3.75, 3.75), aspect="equal",
               xticks=[-2, -1, 0, 1, 2], yticks=[-3, -2, -1, 0, 1, 2, 3])
        ax.grid(False)

    # Sample the full cell boundary so its curved image is drawn accurately.
    horizontal = np.linspace(0.5, 1.0, 100)
    vertical = np.linspace(-0.5, 0.0, 100)
    boundary = np.concatenate(
        [np.column_stack((horizontal, np.full_like(horizontal, -0.5))),
         np.column_stack((np.full_like(vertical, 1.0), vertical)),
         np.column_stack((horizontal[::-1], np.zeros_like(horizontal))),
         np.column_stack((np.full_like(vertical, 0.5), vertical[::-1]))]
    )
    point = np.array([[0.75, -0.25]])
    for ax, transform in zip(axes, (lambda z: z, coupling)):
        cell = transform(boundary)
        mapped_point = transform(point)[0]
        ax.add_patch(Polygon(cell, closed=True, facecolor=ORANGE, edgecolor=ORANGE,
                             alpha=0.40, lw=1.8, zorder=3))
        ax.axvline(point[0, 0], color=GRAY, ls="--", lw=1.0, zorder=1)
        ax.scatter(*mapped_point, color=ORANGE, edgecolor="white", linewidth=0.6,
                   s=42, zorder=5)

    # Finite cell area is an integral of the local area scale, not one determinant.
    integration_grid = np.linspace(0.5, 1, 10001)
    scale = np.exp(0.45 * np.tanh(integration_grid))
    output_area = 0.25 * np.sum((scale[1:] + scale[:-1]) * np.diff(integration_grid))
    axes[0].set(title="Before: a regular grid", xlabel=r"$z_1$", ylabel=r"$z_2$")
    axes[1].set(title="After: shift and stretch the second coordinate",
                xlabel=r"$x_1=z_1$", ylabel=r"$x_2$")
    axes[0].text(0.04, 0.96, "Highlighted area = 0.250", transform=axes[0].transAxes,
                 va="top", fontsize=9)
    axes[1].text(0.04, 0.96, f"Highlighted area ≈ {output_area:.3f}",
                 transform=axes[1].transAxes, va="top", fontsize=9)

    caption = fig.add_subplot(gs[1, :])
    caption.axis("off")
    caption.text(
        0.5, 0.87,
        r"$x_1=z_1,\qquad x_2=z_2\exp(s(z_1))+t(z_1)$",
        ha="center", va="center", fontsize=13,
    )
    caption.text(
        0.5, 0.48,
        r"$s(z_1)=0.45\tanh(z_1),\quad t(z_1)=0.65\sin(2z_1),"
        r"\qquad\log|\det Dg(z)|=s(z_1)$",
        ha="center", va="center", fontsize=11,
    )
    caption.text(
        0.5, 0.08,
        "The first coordinate stays fixed. The Jacobian is triangular: its diagonal is "
        + r"$1,\exp(s(z_1))$.",
        ha="center", va="center", fontsize=10,
    )
    save(fig, "coupling_geometry")


def diagram_box(ax, x, y, label, color, width=0.12):
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, y - 0.066), width, 0.132,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, edgecolor=color, facecolor="white", transform=ax.transAxes,
    ))
    ax.text(x, y, label, ha="center", va="center", fontsize=12,
             transform=ax.transAxes, color=color)


def flow_directions():
    """Sampling and density evaluation traverse the same layers in opposite orders."""
    fig, ax = plt.subplots(figsize=(10.3, 4.7), constrained_layout=True)
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    xs = [0.09, 0.31, 0.50, 0.69, 0.91]
    ax.text(0.02, 0.97, "Generate samples: apply the generative layers in order",
             fontsize=12, color=BLUE, va="top")
    ax.text(0.02, 0.61, "Evaluate density: undo the layers in reverse order",
             fontsize=12, color=ORANGE, va="top")

    for y, color, labels, forward in (
        (0.80, BLUE, [r"$z\sim p_0$", r"$g_1$", r"$\cdots$", r"$g_K$", r"$x\sim q_\theta$"], True),
        (0.44, ORANGE, [r"$z=f_\theta(x)$", r"$g_1^{-1}$", r"$\cdots$", r"$g_K^{-1}$", r"$x$"], False),
    ):
        for index, (x, label) in enumerate(zip(xs, labels)):
            if index == 2:
                ax.text(x, y, label, color=color, ha="center", va="center", fontsize=17)
            else:
                diagram_box(ax, x, y, label, color, width=0.14 if index in (0, 4) else 0.11)
        # Keep arrow endpoints clear of boxes and the ellipsis.
        for left, right in ((0.18, 0.242), (0.377, 0.466), (0.534, 0.623), (0.758, 0.82)):
            start, end = (left, right) if forward else (right, left)
            ax.annotate("", xy=(end, y), xytext=(start, y),
                         arrowprops={"arrowstyle": "->", "lw": 1.7, "color": color})

    ax.text(0.5, 0.23,
             r"$g_\theta=g_K\circ\cdots\circ g_1,\qquad f_\theta=g_\theta^{-1}$",
             ha="center", fontsize=12)
    ax.text(0.5, 0.12,
             r"$\log q_\theta(x)=\log p_0(z)-\log|\det Dg_\theta(z)|$",
             ha="center", color=BLUE, fontsize=13)
    ax.text(0.5, 0.02,
             r"$\log q_\theta(x)=\log p_0(f_\theta(x))+\log|\det Df_\theta(x)|$",
             ha="center", color=ORANGE, fontsize=13)
    save(fig, "flow_directions")


if __name__ == "__main__":
    for build in (change_of_variables, coupling_geometry, flow_directions):
        build()
        print(f"Created {build.__name__}.png and .pdf")
