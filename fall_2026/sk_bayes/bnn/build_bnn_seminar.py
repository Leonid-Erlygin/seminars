"""Build synchronized BNN student/solution notebooks using the standard library.

Generate explanatory assets first with build_bnn_figures.py. This builder clears
saved outputs; execute_notebooks.py regenerates the worked solution outputs.
"""

from pathlib import Path
import base64
import json
import textwrap

from bnn_mnist_cells import add_mnist_section

HERE = Path(__file__).resolve().parent
CELLS = []


def clean(source):
    return textwrap.dedent(source).strip() + "\n"


def markdown(source):
    CELLS.append(("markdown", clean(source), None, None))


def code(source):
    CELLS.append(("code", clean(source), None, None))


def exercise(label, starter, solution):
    CELLS.append(("code", clean(starter), clean(solution), str(label)))


def answer(solution):
    CELLS.append(
        (
            "markdown",
            "Your interpretation:\n\n_Write your answer here._\n",
            "Interpretation. " + clean(solution),
            "written-answer",
        )
    )


def illustration(name, caption):
    CELLS.append(("figure", name, caption, None))


markdown(
    r"""
# Bayesian neural networks: predictions, uncertainty, and active learning

Skoltech · Bayesian machine learning

The Laplace seminar averaged predictions over a Gaussian approximation to a
parameter posterior. A Bayesian neural network asks the same question with
neural-network weights: how should predictions change when more than one
plausible function remains after observing data?

We reuse the first example from `spring_2025/seminar_ue_en.ipynb`: a binary
classifier on four one-dimensional intervals. The geometry makes individual
dropout functions and their disagreement visible. Three tasks implement
Monte Carlo prediction, an uncertainty decomposition, and label acquisition.
Task 4 extends the comparison to MNIST with a dropout CNN on a GPU.

Tasks 1–3 form a 90-minute core. Task 4 is a separate practical, and the two
appendices explore data geometry, Monte Carlo budget, and batch redundancy.
Prerequisites are Bayes' rule, categorical probabilities, expectation,
entropy, and basic PyTorch. No previous active-learning code is required.
"""
)
markdown(
    r"""
## Setup

Use Python 3.9+ with NumPy, Matplotlib, PyTorch, IPython, and a notebook
interface. Task 4 also needs a matching torchvision installation and a CUDA
GPU; it downloads public MNIST once and then reuses a local data cache.
Install `requirements.txt` in your chosen environment. Run cells in order.

Unfinished tasks raise `NotImplementedError` and their dependent experiments
are skipped. Incorrect completed implementations fail their checks visibly.
`RUN_MNIST` enables the GPU practical; without CUDA its training is explicitly
skipped. `RUN_APPENDICES` enables additional controlled experiments.

All small examples run on a CPU. Fixed seeds make comparisons repeatable on
a given software stack; last digits can differ between versions or devices.
`SAVE_FIGURES` exports the solution plots for the printable teacher's guide.
"""
)
code(
    r"""
from contextlib import contextmanager
from pathlib import Path
import copy
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from IPython.display import display, Markdown

RUN_MNIST = True
RUN_APPENDICES = False
SAVE_FIGURES = False
TOY_TRAIN_STEPS = 4000
TOY_AL_STEPS = 1000
MC_DRAWS = 200
DEVICE = torch.device("cpu")
torch.set_num_threads(1)
torch.manual_seed(2026)
np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "axes.titleweight": "normal",
        "figure.figsize": (9, 3.5),
    }
)
completed = set()
BLUE, ORANGE, GREEN = "#2369a1", "#c77516", "#2b8868"


def table(headers, rows):
    def fmt(value):
        return f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines += ["| " + " | ".join(map(fmt, row)) + " |" for row in rows]
    display(Markdown("\n".join(lines)))


def check_task(label, test):
    completed.discard(label)
    try:
        test()
    except NotImplementedError:
        print(f"Task {label}: unfinished; dependent experiments will be skipped.")
        return False
    completed.add(label)
    print(f"Task {label}: checks passed.")
    return True


def ready(*labels):
    missing = [str(label) for label in labels if label not in completed]
    if missing:
        print("Complete Task " + ", ".join(missing) + " to run this experiment.")
    return not missing


def assert_close(actual, expected, atol=1e-8, rtol=1e-8):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def show_figure(fig, name):
    if SAVE_FIGURES:
        assets = Path("assets")
        assets.mkdir(exist_ok=True)
        fig.savefig(assets / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(assets / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


print(
    f"PyTorch {torch.__version__}; small examples on {DEVICE}; CUDA available: {torch.cuda.is_available()}"
)
"""
)
markdown(
    r"""
## 1. From uncertain weights to uncertain predictions

For inputs $x$ and class $y\in\{0,\ldots,C-1\}$, a neural network with weights
$w$ defines logits $a_w(x)$ and categorical probabilities
$$p(y=c\mid x,w)=\operatorname{softmax}(a_w(x))_c.$$
Conditioning on the observed inputs requires no assumption that their
distribution is uniform. Given labelled observations $\mathcal D$, a weight
prior and likelihood define
$$p(w\mid\mathcal D)\propto p(w)\prod_{i=1}^N p(y_i\mid x_i,w).$$
Prediction integrates weights rather than inserting just one fitted vector:
$$p(y=c\mid x,\mathcal D)=\mathbb E_{p(w\mid\mathcal D)}p(y=c\mid x,w).$$
A nonlinear network usually makes this posterior intractable. If an
approximation $q(w)$ is available, independent draws give
$$\bar p_c(x)=\frac1T\sum_{t=1}^T p(y=c\mid x,w_t),\qquad w_t\sim q.$$
Average the probabilities, not the logits. In general
$\operatorname{softmax}(\mathbb E a_w)\ne\mathbb E\operatorname{softmax}(a_w)$.
The predicted class is $\arg\max_c\bar p_c$; it is not the MAP estimate of
the weights. More draws reduce numerical integration error under $q$ but
do not make $q$ equal to the desired posterior.

### Monte Carlo dropout

Dropout multiplies hidden features by random Bernoulli masks. During training
this regularizes the network. Retaining dropout during prediction defines a
distribution over stochastic networks. Under particular modelling and
regularization assumptions this has a variational interpretation [1].
For a general prior, variational inference would maximize
$$\mathcal L(q)=\mathbb E_q\left[\sum_i\log p(y_i\mid x_i,w)\right]
                  -D_{\rm KL}(q\Vert p(w)).$$
Our practical uses dropout and L2 regularization as an approximate method;
its masks are not exact posterior weight samples, and changing an arbitrary
dropout rate or weight decay does not establish posterior calibration.
"""
)
markdown(
    r"""
## 2. Reuse the one-dimensional example

Keep the original interval dictionary and 30 equally spaced observations
per interval:
$$\begin{aligned}
y=0:&\quad [-5,-3]\ \text{and}\ [1,2.5],\\
y=1:&\quad [-3,-2]\ \text{and}\ [2.5,5].
\end{aligned}$$
There are 120 observations. `linspace` includes both ends, so the shared
points $-3$ and $2.5$ each appear with opposing labels. The central gap
$(-2,1)$ contains no observations. We preserve these details rather than
silently changing the motivating example. The training grid is deterministic;
it is not a claim that these 120 inputs were iid uniform draws.

The network follows the same architecture:
`1 → 10 → 50 → 20 → 1`, with sigmoid hidden activations and dropout probability
0.2 before the second hidden sigmoid. The input scaling is the original
$u=(x+5)/10$. For numerical stability, the network returns two centred logits
$(-a/2,a/2)$ instead of applying a final sigmoid; class-1 probability is still
$\operatorname{sigmoid}(a)$. Cross entropy is the corresponding binary log loss.
"""
)
illustration(
    "interval_geometry",
    "The original observation geometry has a central gap and two shared endpoints. "
    "The reference distributions for overlap/gap experiments are defined separately; "
    "a conditional label probability is undefined where the input has zero density.",
)
code(
    r'''
END_POINTS = {0: [[-5, -3], [1, 2.5]], 1: [[-3, -2], [2.5, 5]]}


def interval_data(end_points=END_POINTS, points_per_interval=30):
    x, y = [], []
    for label, intervals in end_points.items():
        for left, right in intervals:
            x.append(torch.linspace(left, right, points_per_interval))
            y.append(torch.full((points_per_interval,), label, dtype=torch.long))
    return torch.cat(x)[:, None], torch.cat(y)


class SharedMaskDropout(nn.Module):
    """Ordinary training dropout; one feature mask per MC function draw."""

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        self.mc_active = False
        self.mc_mask = None

    def forward(self, x):
        if self.mc_active:
            if self.mc_mask is None:
                self.mc_mask = torch.empty_like(x[:1]).bernoulli_(1 - self.p) / (1 - self.p)
            return x * self.mc_mask
        return torch.nn.functional.dropout(x, p=self.p, training=self.training)


@contextmanager
def sampled_model(model):
    """One stochastic function; freeze normalization, then restore every mode."""
    modes = [(module, module.training) for module in model.modules()]
    shared = [
        (module, module.mc_active, module.mc_mask)
        for module in model.modules()
        if isinstance(module, SharedMaskDropout)
    ]
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, SharedMaskDropout):
                module.mc_active, module.mc_mask = True, None
            elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        with torch.no_grad():
            yield
    finally:
        for module, active, mask in shared:
            module.mc_active, module.mc_mask = active, mask
        for module, training in modes:
            module.training = training


class ToyDropoutNet(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 50),
            SharedMaskDropout(dropout),
            nn.Sigmoid(),
            nn.Linear(50, 20),
            nn.Sigmoid(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        a = self.net((x + 5) / 10)
        return torch.cat([-0.5 * a, 0.5 * a], dim=-1)


class ActiveToyNet(nn.Module):
    """Smaller network for fitting repeatedly to very small labelled subsets."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            SharedMaskDropout(dropout),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x / 5)


def fit_toy(x, y, seed, steps=TOY_TRAIN_STEPS, dropout=0.2, active=False):
    torch.manual_seed(seed)
    model = ActiveToyNet(dropout) if active else ToyDropoutNet(dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.03 if active else 0.1,
        total_steps=steps,
        div_factor=10,
        final_div_factor=100,
    )
    history = []
    model.train()
    for step in range(steps):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 400 == 0 or step == steps - 1:
            history.append((step, loss.item()))
    model.eval()
    return model, np.array(history)


x_train, y_train = interval_data()
assert x_train.shape == (120, 1)
assert torch.bincount(y_train).tolist() == [60, 60]
toy_model, toy_history = fit_toy(x_train, y_train, seed=11)
x_grid = torch.linspace(-7, 7, 501)[:, None]
with torch.no_grad():
    deterministic_probability = toy_model(x_grid).softmax(-1)[:, 1]
fig, ax = plt.subplots(figsize=(9, 3))
for c, color in [(0, BLUE), (1, ORANGE)]:
    ax.scatter(
        x_train[y_train == c, 0], y_train[y_train == c], color=color, s=16, label=f"Class {c}"
    )
ax.plot(x_grid[:, 0], deterministic_probability, color=GREEN, label="Dropout off")
ax.axvspan(-2, 1, color="0.7", alpha=0.15, label="No observations")
ax.set(
    xlabel="x",
    ylabel="Class-1 probability / label",
    ylim=(-0.08, 1.08),
    title="The reused interval classifier",
)
ax.legend(ncol=4, fontsize=9)
plt.tight_layout()
show_figure(fig, "toy_observations")
'''
)
markdown(
    r"""
## Task 1 — Average stochastic predictions

Complete `mc_probabilities(model, inputs, n_draws=200)` and return `(T,N,C)`.
The model returns raw class logits. The supplied `sampled_model` context
activates dropout, freezes normalization layers, disables gradients, and
restores previous modes even after an exception. Each context represents
one new draw; put it inside the loop and apply softmax over classes.

For the toy network, the supplied shared mask is held fixed for every input
in a draw. Connecting those predictions across the grid therefore draws one
coherent function. Ordinary PyTorch dropout uses independent masks for rows
in a batch: it estimates pointwise marginals but does not produce coherent
function curves. Task 4 will explicitly use that pointwise interpretation.

Explain why the deterministic evaluation curve need not equal the MC mean,
and why averaging logits would compute a different quantity.
"""
)
exercise(
    "1",
    r'''
def mc_probabilities(model, inputs, n_draws=200):
    """Raw inputs (N,d) -> probability draws (T,N,C), with no gradient history."""
    if n_draws < 1 or len(inputs) == 0:
        raise ValueError("Need at least one draw and one input")
    # TODO: draw one stochastic model at a time and stack class probabilities.
    raise NotImplementedError("Task 1")
''',
    r'''
def mc_probabilities(model, inputs, n_draws=200):
    """Raw inputs (N,d) -> probability draws (T,N,C), with no gradient history."""
    if n_draws < 1 or len(inputs) == 0:
        raise ValueError("Need at least one draw and one input")
    draws = []
    for _ in range(n_draws):
        with sampled_model(model):
            draws.append(model(inputs).softmax(dim=-1))
    return torch.stack(draws, dim=0)
''',
)
code(
    r"""
def test_mc_probabilities():
    class Constant(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor([-2.0, 1.0], dtype=torch.float64))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.value.expand(len(x), -1)

    model = Constant()
    p = mc_probabilities(model, torch.zeros(4, 1, dtype=torch.float64), 5)
    assert p.shape == (5, 4, 2) and p.dtype == torch.float64 and not p.requires_grad
    assert model.calls == 5
    assert_close(p, model.value.softmax(-1).expand(5, 4, 2))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        net = ToyDropoutNet().double()
        net.train()
        net.net[0].eval()  # Deliberately mixed modes.
        before = [m.training for m in net.modules()]
        shared = next(m for m in net.modules() if isinstance(m, SharedMaskDropout))
        p = mc_probabilities(net, torch.ones(4, 1, dtype=torch.float64), 64)
        assert [m.training for m in net.modules()] == before
        assert not shared.mc_active and shared.mc_mask is None
        assert_close(p, p[:, :1].expand_as(p))  # Same function at repeated x.
        assert p[:, 0, 1].std() > 1e-6
        assert_close(p.sum(-1), torch.ones(64, 4, dtype=torch.float64))
        bn_model = nn.Sequential(nn.BatchNorm1d(1), nn.Linear(1, 2)).double()
        bn_model.train()
        before_mean = bn_model[0].running_mean.clone()
        mc_probabilities(bn_model, torch.ones(4, 1, dtype=torch.float64), 3)
        assert bn_model.training and bn_model[0].training
        assert_close(bn_model[0].running_mean, before_mean)
        try:
            with sampled_model(net):
                raise RuntimeError("Restoration check")
        except RuntimeError:
            pass
        assert [m.training for m in net.modules()] == before
        assert not shared.mc_active and shared.mc_mask is None


check_task(1, test_mc_probabilities)
"""
)
code(
    r"""
if ready(1):
    torch.manual_seed(13)
    toy_draws = mc_probabilities(toy_model, x_grid, MC_DRAWS)
    positive = toy_draws[:, :, 1]
    mean_probability = positive.mean(0)
    low, high = torch.quantile(positive, positive.new_tensor([0.05, 0.95]), dim=0)
    mean_se = positive.std(0) / math.sqrt(MC_DRAWS)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axvspan(-2, 1, color="0.7", alpha=0.15)
    for draw in positive[:15]:
        ax.plot(x_grid[:, 0], draw, color=BLUE, alpha=0.16, lw=0.8)
    ax.fill_between(
        x_grid[:, 0],
        low,
        high,
        color=BLUE,
        alpha=0.14,
        label="Pointwise 5–95% range under dropout",
    )
    ax.plot(x_grid[:, 0], mean_probability, color=BLUE, lw=2, label="MC probability mean")
    ax.plot(
        x_grid[:, 0], deterministic_probability, color=ORANGE, ls="--", label="Dropout off"
    )
    for label, color in [(0, BLUE), (1, ORANGE)]:
        ax.scatter(x_train[y_train == label, 0], y_train[y_train == label], color=color, s=10)
    ax.set(
        xlabel="x",
        ylabel="P(class 1)",
        ylim=(-0.05, 1.05),
        title="One mask gives one function across the grid",
    )
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    show_figure(fig, "toy_mc_predictions")
    selected = [
        int(torch.argmin((x_grid[:, 0] - value).abs())) for value in [-4.0, -1.0, 1.7, 6.0]
    ]
    table(
        ["x", "MC mean", "Component-probability SD", "MC SE of mean"],
        [
            [
                x_grid[i, 0].item(),
                mean_probability[i].item(),
                positive[:, i].std().item(),
                mean_se[i].item(),
            ]
            for i in selected
        ],
    )
"""
)
answer(
    r"""
The MC mean averages probabilities under the dropout distribution. Turning
dropout off uses a deterministic network, and nonlinear activations prevent
the two operations from commuting. The shaded component-probability range
describes variation under the approximation; it is not an MC confidence
interval for its mean and is not an exact posterior credible band. The MC SE
describes numerical integration error at one fixed input. More draws reduce
that error while leaving the fitted dropout distribution unchanged.
"""
)
markdown(
    r"""
## 3. Two routes to the same uncertain prediction

For one input, write $p_t\in[0,1]^C$ for the class probability vector of draw
$t$, and $\bar p=T^{-1}\sum_t p_t$. Categorical entropy is
$$H(p)=-\sum_{c=1}^C p_c\log p_c,\qquad 0\log0=0.$$
Use natural logs, so binary entropy is at most $\log2$ nats. Under a chosen
weight distribution $q$, the predictive entropy decomposes as
$$\underbrace{H(\mathbb E_q p_w)}_{\text{predictive entropy}}
=\underbrace{\mathbb E_q H(p_w)}_{\text{expected conditional entropy}}
 +\underbrace{I_q(Y;W\mid x,\mathcal D)}_{\text{mutual information}}.$$
Equivalently,
$$I_q=\mathbb E_q\,D_{\rm KL}\big(p(y\mid x,W)\Vert\mathbb E_q p(y\mid x,W)\big).$$
The first term is large when individual models each predict ambiguously.
The MI term is large when their class probabilities disagree. These are
often called aleatoric and epistemic components, respectively. The labels
are relative to the model and $q$: the first is not automatically the true
irreducible data uncertainty, and a small MI need not show that a region is
well understood.

Replace expectations with averages over draws:
$$\widehat I=H(\bar p)-\frac1T\sum_tH(p_t).$$
For the empirical mixture this is nonnegative up to roundoff. With iid draws,
the average conditional entropy is unbiased for its $q$ expectation, whereas
$H(\bar p)$ and consequently $\widehat I$ are generally downward biased by
concavity. At $T=1$, $\widehat I=0$ for every input.
"""
)
illustration(
    "uncertainty_decomposition",
    "Ambiguous agreement and confident disagreement can have the same predictive "
    "mean and entropy. Expected conditional entropy and mutual information distinguish "
    "them under the chosen distribution of models.",
)
markdown(
    r"""
## Task 2 — Compute entropy and disagreement

1. Implement `categorical_entropy(probabilities)` with a sum over the last
   axis. Handle zero probabilities without `NaN` values.
2. Implement `uncertainty_terms(draws)` for `(T,N,C)` with `T >= 2`. Return
   the mean probability, predictive entropy, expected conditional entropy,
   mutual information, and pointwise MC SE of the mean probabilities.
3. Compare an ensemble where every model predicts `(0.5,0.5)` with an
   ensemble split equally between `(1,0)` and `(0,1)`.

Use `std(..., unbiased=True)/sqrt(T)` for the probability MC SE. Only numerical
roundoff should require clamping a tiny negative MI; do not replace MI by
entropy or predictive variance.
"""
)
code(
    r"""
def validate_probabilities(probabilities):
    if (
        not probabilities.is_floating_point()
        or probabilities.ndim < 1
        or probabilities.shape[-1] < 2
    ):
        raise ValueError("Expected floating class probabilities with at least two classes")
    if (
        not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not torch.allclose(
        probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Class probabilities must sum to one")
"""
)
exercise(
    "2",
    r'''
def categorical_entropy(probabilities):
    """(...,C) -> entropy (...), in nats, with 0 log 0 = 0."""
    validate_probabilities(probabilities)
    # TODO: stable elementwise p log p, summed over classes.
    raise NotImplementedError("Task 2")


def uncertainty_terms(draws):
    """(T,N,C), T>=2 -> dictionary of predictive and MC-integration summaries."""
    if draws.ndim != 3 or draws.shape[0] < 2:
        raise ValueError("Expected (T,N,C) with at least two draws")
    validate_probabilities(draws)
    # TODO: mean, both entropies, their difference, and probability MC SE.
    raise NotImplementedError("Task 2")
''',
    r'''
def categorical_entropy(probabilities):
    """(...,C) -> entropy (...), in nats, with 0 log 0 = 0."""
    validate_probabilities(probabilities)
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return -(probabilities * safe.log()).sum(dim=-1)


def uncertainty_terms(draws):
    """(T,N,C), T>=2 -> dictionary of predictive and MC-integration summaries."""
    if draws.ndim != 3 or draws.shape[0] < 2:
        raise ValueError("Expected (T,N,C) with at least two draws")
    validate_probabilities(draws)
    mean = draws.mean(dim=0)
    predictive = categorical_entropy(mean)
    expected = categorical_entropy(draws).mean(dim=0)
    return {
        "mean_probability": mean,
        "predictive_entropy": predictive,
        "expected_entropy": expected,
        "mutual_information": (predictive - expected).clamp_min(0),
        "probability_mc_se": draws.std(dim=0, unbiased=True) / math.sqrt(draws.shape[0]),
    }
''',
)
code(
    r"""
def test_uncertainty():
    point_mass = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    assert_close(categorical_entropy(point_mass), torch.zeros(2, dtype=torch.float64))
    uncertain = torch.full((4, 1, 2), 0.5, dtype=torch.float64)
    disagreement = point_mass.repeat(2, 1)[:, None]
    a, b = uncertainty_terms(uncertain), uncertainty_terms(disagreement)
    log2 = uncertain.new_tensor(math.log(2))
    assert_close(a["mean_probability"], b["mean_probability"])
    assert_close(a["predictive_entropy"][0], log2)
    assert_close(a["expected_entropy"][0], log2)
    assert_close(a["mutual_information"], torch.zeros(1, dtype=torch.float64))
    assert_close(b["expected_entropy"], torch.zeros(1, dtype=torch.float64))
    assert_close(b["mutual_information"][0], log2)
    assert_close(
        b["probability_mc_se"], torch.full((1, 2), 1 / math.sqrt(12), dtype=torch.float64)
    )
    draws = torch.tensor(
        [
            [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]],
            [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]],
            [[0.4, 0.4, 0.2], [0.3, 0.3, 0.4]],
        ],
        dtype=torch.float64,
    )
    terms = uncertainty_terms(draws)
    mean = draws.mean(0)
    # Independent KL identity checks both axes and the entropy difference.
    kl = (draws * (draws / mean).log()).sum(-1).mean(0)
    assert_close(terms["mutual_information"], kl)
    assert terms["mean_probability"].shape == terms["probability_mc_se"].shape == (2, 3)
    assert terms["predictive_entropy"].shape == terms["expected_entropy"].shape == (2,)
    try:
        uncertainty_terms(draws[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("One draw has no sample-variance MC SE")


check_task(2, test_uncertainty)
"""
)
code(
    r"""
if ready(1, 2):
    toy_uncertainty = uncertainty_terms(toy_draws)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(x_grid[:, 0], toy_uncertainty["mean_probability"][:, 1], color=BLUE)
    axes[0].scatter(x_train[:, 0], y_train, c=y_train, cmap="coolwarm", s=12)
    axes[0].set(
        ylabel="P(class 1)", title="The model's mean prediction and uncertainty components"
    )
    for key, label, color in [
        ("predictive_entropy", "Predictive entropy", BLUE),
        ("expected_entropy", "Expected conditional entropy", GREEN),
        ("mutual_information", "Mutual information", ORANGE),
    ]:
        axes[1].plot(x_grid[:, 0], toy_uncertainty[key], color=color, label=label)
    for ax in axes:
        ax.axvspan(-2, 1, color="0.7", alpha=0.15)
    axes[1].set(xlabel="x", ylabel="Nats", ylim=(-0.02, math.log(2) + 0.04))
    axes[1].legend(fontsize=9, ncol=3)
    plt.tight_layout()
    show_figure(fig, "toy_uncertainty")
    table(
        ["x", "Predictive entropy", "Conditional entropy", "MI"],
        [
            [
                x_grid[i, 0].item(),
                toy_uncertainty["predictive_entropy"][i].item(),
                toy_uncertainty["expected_entropy"][i].item(),
                toy_uncertainty["mutual_information"][i].item(),
            ]
            for i in selected
        ],
    )
"""
)
markdown(
    r"""
Interpretation questions. Where does this fitted dropout model disagree?
Does the gap always have larger MI than observed regions? Can two inputs
with the same predictive entropy suggest different reasons for uncertainty?
How does increasing the number of MC draws differ from collecting labels?
"""
)
answer(
    r"""
Inspect all three curves together. High expected conditional entropy means
that individual stochastic models tend to be ambiguous; high MI means their
probability vectors differ. The decomposition holds under this fitted
approximation, so it does not prove what the true noise level is. Missing
data do not force this architecture to disagree, and extrapolation can remain
confident. More draws improve integration under the same approximation;
additional informative labels can change the fitted model itself.
"""
)

markdown(
    r"""
## 4. When should we ask for another label?

In pool-based active learning the inputs of an unlabelled pool are available,
but their labels can be requested only through an oracle. A labelling budget
counts the initial labelled examples and every newly queried example.
Acquisition receives pool inputs and model predictions, never hidden labels.

Predictive-entropy acquisition selects large $H(\bar p(x))$. BALD instead
selects large disagreement [2,3]:
$$a_{\rm BALD}(x)=I_q(Y;W\mid x,\mathcal D).$$
For a Bayesian posterior, this is the expected information gained about
weights by observing the label. Under an approximate $q$, define
$$q_y(w\mid x)\propto q(w)p(y\mid x,w).$$
Then $I_q=\mathbb E_{y\sim\bar p}\,D_{\rm KL}(q_y\Vert q)$. Refitting our
dropout network after querying a label need not equal that formal update.
Thus BALD is an approximate acquisition rule, not a guarantee of optimal
learning or of winning every comparison with random selection.

High predictive entropy may come from ambiguity that all models share.
BALD targets disagreement instead. Both can be useful; neither criterion
knows the true data-generating uncertainty. Querying the largest individual
scores for a batch can also acquire redundant neighbours. A sum of pointwise
BALD scores is not the joint information gain of the whole label batch [4].
"""
)
illustration(
    "active_learning_cycle",
    "Score pool inputs before asking for labels. The oracle reveals labels only "
    "after unique IDs have been selected. Initial labels count towards the same "
    "budget; evaluation labels are a separate resource.",
)
markdown(
    r"""
## Task 3 — Acquire labels without looking at them

Implement `acquisition_indices(pool_ids, draws, n_query, strategy, generator)`.
`pool_ids` is a CPU integer vector of unique global IDs, and `draws` contains
`(T,N,C)` probabilities in exactly that pool order. Return unique global IDs,
not positions inside the remaining pool.

1. For `random`, sample without replacement using the supplied generator.
2. For `entropy`, select the largest predictive entropies.
3. For `bald`, select the largest mutual informations. Break score ties by
   choosing the smaller global ID first.

The supplied oracle checks for repeated queries and budget overruns. Its
private labels model information access; this Python class is a teaching
interface, not a security boundary. Explain why repeating an index wastes
budget and why querying a top-scoring batch need not maximize joint information.
"""
)
code(
    r"""
class LabelOracle:
    def __init__(self, inputs, hidden_labels, budget):
        self.inputs = inputs.clone()
        self._hidden_labels = hidden_labels.clone()
        self.budget = budget
        self._labelled = {}

    @property
    def remaining_ids(self):
        return torch.tensor(
            [i for i in range(len(self.inputs)) if i not in self._labelled], dtype=torch.long
        )

    @property
    def spent(self):
        return len(self._labelled)

    def query(self, ids):
        ids = torch.as_tensor(ids)
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Oracle IDs must be integers, not floats or booleans")
        ids = ids.to(dtype=torch.long, device="cpu")
        if ids.ndim != 1 or len(torch.unique(ids)) != len(ids):
            raise ValueError("A query must be a vector of unique IDs")
        values = ids.tolist()
        if any(i < 0 or i >= len(self.inputs) or i in self._labelled for i in values):
            raise ValueError("Query contains an invalid or already labelled ID")
        if self.spent + len(values) > self.budget:
            raise ValueError("Query would exceed the total label budget")
        labels = self._hidden_labels[ids].clone()
        for i, label in zip(values, labels.tolist()):
            self._labelled[i] = label
        return ids, labels

    def labelled_data(self):
        ids = torch.tensor(list(self._labelled), dtype=torch.long)
        labels = torch.tensor(list(self._labelled.values()), dtype=torch.long)
        return self.inputs[ids], labels
"""
)
ACQUISITION_START = r'''
def acquisition_indices(pool_ids, draws, n_query, strategy="bald", generator=None):
    """Select unique global IDs; no oracle or labels are accepted as arguments."""
    if pool_ids.ndim != 1 or pool_ids.dtype != torch.long or pool_ids.device.type != "cpu":
        raise ValueError("pool_ids must be a CPU int64 vector")
    if draws.ndim != 3 or draws.shape[1] != len(pool_ids) or draws.device.type != "cpu":
        raise ValueError("Expected CPU draws (T,N,C) matching the pool")
    if len(pool_ids.unique()) != len(pool_ids) or not 0 <= n_query <= len(pool_ids):
        raise ValueError("IDs must be unique and the query must fit inside the pool")
    if strategy not in ("random", "entropy", "bald"):
        raise ValueError("Unknown strategy")
'''
exercise(
    "3",
    ACQUISITION_START
    + r"""
    # TODO: sample without replacement or rank pointwise scores with stable ties.
    raise NotImplementedError("Task 3")
""",
    ACQUISITION_START
    + r"""
    if strategy == "random":
        return pool_ids[torch.randperm(len(pool_ids), generator=generator)[:n_query]]
    terms = uncertainty_terms(draws)
    key = "predictive_entropy" if strategy == "entropy" else "mutual_information"
    by_id = torch.argsort(pool_ids)
    by_score = torch.argsort(terms[key][by_id], descending=True, stable=True)
    return pool_ids[by_id[by_score[:n_query]]]
""",
)
code(
    r"""
def test_acquisition():
    if not ready(2):
        raise NotImplementedError("Task 2 prerequisite")
    ids = torch.tensor([41, 7, 99])
    draws = torch.tensor(
        [
            [[0.5, 0.5], [1.0, 0.0], [0.9, 0.1]],
            [[0.5, 0.5], [1.0, 0.0], [0.9, 0.1]],
            [[0.5, 0.5], [1.0, 0.0], [0.9, 0.1]],
            [[0.5, 0.5], [0.0, 1.0], [0.9, 0.1]],
        ],
        dtype=torch.float64,
    )
    assert acquisition_indices(ids, draws, 1, "entropy").tolist() == [41]
    assert acquisition_indices(ids, draws, 1, "bald").tolist() == [7]
    assert acquisition_indices(ids, draws, 3, "bald").tolist() == [7, 41, 99]
    assert len(acquisition_indices(ids, draws, 0)) == 0
    tied = torch.full((4, 3, 2), 0.5, dtype=torch.float64)
    assert acquisition_indices(ids, tied, 3, "entropy").tolist() == [7, 41, 99]
    rng1, rng2 = torch.Generator().manual_seed(10), torch.Generator().manual_seed(10)
    random_ids = acquisition_indices(ids, draws, 3, "random", rng1)
    assert random_ids.unique().numel() == 3
    assert torch.equal(random_ids, ids[torch.randperm(3, generator=rng2)])
    oracle = LabelOracle(torch.arange(5.0)[:, None], torch.tensor([0, 1, 0, 1, 0]), budget=3)
    queried, labels = oracle.query(torch.tensor([3, 1]))
    assert queried.tolist() == [3, 1] and labels.tolist() == [1, 1] and oracle.spent == 2
    assert oracle.remaining_ids.tolist() == [0, 2, 4]
    for invalid in [
        torch.tensor([0, 0]),
        torch.tensor([1]),
        torch.tensor([0, 2]),
        torch.tensor([0.9]),
        torch.tensor([False]),
    ]:
        before = oracle.spent
        try:
            oracle.query(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid query accepted")
        assert oracle.spent == before  # Rejections must not mutate the state.


check_task(3, test_acquisition)
"""
)
markdown(
    r"""
### A controlled one-dimensional comparison

Reuse the same 120 inputs as a finite acquisition pool, now pretending that
their labels have not yet been purchased. Each repeat starts with 12 random
IDs, chosen without labels. Acquire six new labels on each of four rounds,
ending at 36 labels including the initial 12. Random, entropy, and BALD use
the same initial IDs within each repeat.

For this repeated small-data experiment, use a supplied `1 → 32 → 32 → 2`
tanh network, with dropout 0.2 after its first hidden layer and input scaling
$x/5$. It is shared by all acquisition strategies. This simpler network fits
the initially labelled subset more reliably than the deep sigmoid network;
the original architecture in Tasks 1–2 remains the reused example.

At each budget, retrain a fresh network for 1,000 full-batch optimizer steps,
using Adam, weight decay $10^{-4}$, and a OneCycle maximum learning rate 0.03.
Initialization and stochastic-training seeds agree across strategies at the
same round and repeat. Acquisition and evaluation use 64 MC draws. The
training budget and final label count are fixed before results are inspected.
The fixed weight decay on a mean training loss is an optimization convention;
its equivalent penalty on a summed loss grows with the label count. This
comparison does not trace exact Bayesian updates under one fixed prior.

For evaluation only, define a separate population: choose one of the four
intervals uniformly, then sample an input uniformly within that interval and
use its class. This is distinct from the deterministic training grid and is
not globally uniform in $x$. Generate 512 validation and 1,024 final-test
observations. Their evaluation labels are a separate benchmark resource,
outside the 36 acquired training labels. Validation curves do not select
hyperparameters or stopping times; the final test is evaluated after all
acquisition runs finish.
"""
)
code(
    r"""
def sample_interval_population(n, seed, end_points=END_POINTS):
    components = [
        (label, left, right) for label, ranges in end_points.items() for left, right in ranges
    ]
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(components), (n,), generator=rng)
    u = torch.rand(n, generator=rng)
    x, y = torch.empty(n, 1), torch.empty(n, dtype=torch.long)
    for k, (label, left, right) in enumerate(components):
        chosen = indices == k
        x[chosen, 0] = left + (right - left) * u[chosen]
        y[chosen] = label
    return x, y


def classification_metrics(probabilities, labels):
    correct = (probabilities.argmax(-1) == labels).float().mean().item()
    true_p = probabilities[torch.arange(len(labels)), labels]
    nll = -true_p.clamp_min(torch.finfo(probabilities.dtype).tiny).log().mean().item()
    one_hot = torch.nn.functional.one_hot(labels, probabilities.shape[1]).to(
        probabilities.dtype
    )
    brier = (probabilities - one_hot).square().sum(-1).mean().item()
    return {"accuracy": correct, "nll": nll, "brier": brier}


def run_toy_active_learning():
    validation_x, validation_y = sample_interval_population(512, 31)
    test_x, test_y = sample_interval_population(1024, 32)
    records, final_models, initial_ids_by_repeat = [], [], []
    for repeat in range(3):
        initial = torch.randperm(
            len(x_train), generator=torch.Generator().manual_seed(100 + repeat)
        )[:12]
        initial_ids_by_repeat.append(initial)
        for strategy in ["random", "entropy", "bald"]:
            oracle = LabelOracle(x_train, y_train, budget=36)
            oracle.query(initial)
            query_rng = torch.Generator().manual_seed(200 + repeat)
            for round_index in range(5):
                labelled_x, labelled_y = oracle.labelled_data()
                model, _ = fit_toy(
                    labelled_x,
                    labelled_y,
                    seed=1000 + 100 * repeat + round_index,
                    steps=TOY_AL_STEPS,
                    active=True,
                )
                torch.manual_seed(2000 + 100 * repeat + round_index)
                val_probability = mc_probabilities(model, validation_x, 64).mean(0)
                row = {
                    "repeat": repeat,
                    "strategy": strategy,
                    "budget": oracle.spent,
                    **classification_metrics(val_probability, validation_y),
                }
                records.append(row)
                if round_index == 4:
                    final_models.append((repeat, strategy, model, oracle))
                    break
                remaining = oracle.remaining_ids
                torch.manual_seed(3000 + 100 * repeat + round_index)
                draws = mc_probabilities(model, x_train[remaining], 64)
                selected_ids = acquisition_indices(remaining, draws, 6, strategy, query_rng)
                oracle.query(selected_ids)
            assert oracle.spent == 36
    # Keep test evaluation outside every acquisition loop.
    final_results = []
    for repeat, strategy, model, oracle in final_models:
        torch.manual_seed(4000 + repeat)
        probability = mc_probabilities(model, test_x, 256).mean(0)
        _, labels = oracle.labelled_data()
        final_results.append(
            {
                "repeat": repeat,
                "strategy": strategy,
                "class_counts": torch.bincount(labels, minlength=2),
                **classification_metrics(probability, test_y),
            }
        )
    return records, final_results, initial_ids_by_repeat


if ready(1, 2, 3):
    started = time.perf_counter()
    toy_al_records, toy_al_final, toy_initial_ids = run_toy_active_learning()
    print(f"Paired toy experiments completed in {time.perf_counter()-started:.1f} seconds.")
    rows = []
    for strategy in ["random", "entropy", "bald"]:
        results = [r for r in toy_al_final if r["strategy"] == strategy]
        accuracy = np.array([r["accuracy"] for r in results])
        rows.append(
            [
                strategy,
                accuracy.mean(),
                accuracy.std(ddof=1),
                np.mean([r["nll"] for r in results]),
                np.mean([r["brier"] for r in results]),
            ]
        )
    table(["Strategy", "Final test accuracy", "Repeat SD", "Test NLL", "Test Brier"], rows)
"""
)
code(
    r"""
if ready(1, 2, 3):
    budgets = [12, 18, 24, 30, 36]
    colors = {"random": "0.45", "entropy": BLUE, "bald": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for strategy in ["random", "entropy", "bald"]:
        for ax, metric in zip(axes, ["accuracy", "nll"]):
            values = np.array(
                [
                    [
                        next(
                            r[metric]
                            for r in toy_al_records
                            if r["strategy"] == strategy
                            and r["repeat"] == rep
                            and r["budget"] == budget
                        )
                        for budget in budgets
                    ]
                    for rep in range(3)
                ]
            )
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(budgets, mean, marker="o", color=colors[strategy], label=strategy)
            ax.fill_between(budgets, mean - sd, mean + sd, color=colors[strategy], alpha=0.12)
    axes[0].set(
        xlabel="Acquired training labels (initial labels included)",
        ylabel="Validation accuracy",
        ylim=(0, 1.02),
    )
    axes[1].set(
        xlabel="Acquired training labels (initial labels included)", ylabel="Validation NLL"
    )
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle("Three paired repeats; bands show repeat SD, not confidence intervals")
    plt.tight_layout()
    show_figure(fig, "toy_learning_curves")

    # Show a single, explicitly labelled acquisition decision before querying.
    oracle = LabelOracle(x_train, y_train, budget=18)
    oracle.query(toy_initial_ids[0])
    x_labelled, y_labelled = oracle.labelled_data()
    initial_model, _ = fit_toy(
        x_labelled, y_labelled, seed=1000, steps=TOY_AL_STEPS, active=True
    )
    pool_ids = oracle.remaining_ids
    torch.manual_seed(3000)
    pool_draws = mc_probabilities(initial_model, x_train[pool_ids], 64)
    pool_scores = uncertainty_terms(pool_draws)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].scatter(
        x_train[pool_ids, 0],
        pool_scores["predictive_entropy"],
        s=13,
        color=BLUE,
        label="Predictive entropy",
    )
    axes[0].scatter(
        x_train[pool_ids, 0], pool_scores["mutual_information"], s=13, color=ORANGE, label="MI"
    )
    axes[0].set(
        ylabel="Score (nats)",
        title="First query from one repeat: pool scoring uses no hidden labels",
    )
    for j, strategy in enumerate(["random", "entropy", "bald"]):
        chosen = acquisition_indices(
            pool_ids, pool_draws, 6, strategy, torch.Generator().manual_seed(200)
        )
        axes[1].scatter(
            x_train[chosen, 0],
            np.full(len(chosen), j),
            marker="|",
            s=250,
            color=colors[strategy],
            label=strategy,
        )
    axes[1].scatter(x_labelled[:, 0], np.full(len(x_labelled), -0.6), s=10, color="0.4")
    axes[1].set(
        xlabel="x",
        yticks=[-0.6, 0, 1, 2],
        yticklabels=["Initial inputs", "random", "entropy", "bald"],
    )
    for ax in axes:
        ax.axvspan(-2, 1, color="0.7", alpha=0.13)
    axes[0].legend(ncol=2, fontsize=9)
    plt.tight_layout()
    show_figure(fig, "toy_acquisition")
"""
)
markdown(
    r"""
Interpretation questions. Compare final test scores and the spread across
repeats. Did one strategy improve every criterion on every seed? Where do
the first queried inputs lie? What information can an acquisition rule use,
and which labels count towards the reported budget?
"""
)
answer(
    r"""
Compare like-for-like budgets, including the initial 12 labels, and describe
the actual repeated results. The acquisition rule can inspect pool features
and predictions, but hidden labels become available only after querying.
Validation and test labels are a separate evaluation resource in this
simulation. Closely spaced high-score inputs can be redundant, and unstable
dropout approximations or refitting can change a ranking. An average advantage
on three repeats is evidence for this setting, not a universal guarantee that
BALD dominates random or entropy acquisition.
"""
)

add_mnist_section(markdown, code, exercise, answer)

markdown(
    r"""
## 7. What the four tasks establish

A BNN prediction is a mixture of class probabilities. Dropout provides one
computational approximation to that mixture. Entropy and disagreement measure
different properties of its predictions; active learning uses these properties
to choose which labels to request. The MNIST practical keeps the same ideas
while adding minibatched CNN prediction and an explicit acquisition budget.

More Monte Carlo draws, a more expressive approximation, additional labelled
data, and a better acquisition rule address different limitations. Keep those
effects separate when reading uncertainty plots and learning curves.
"""
)
markdown(
    r"""
## Appendix A. Controlled geometry: overlap and gaps

The original example combines several class transitions with a missing-data
region. To separate interpretations, fit two new scenarios:

- Overlap: class 0 on `[-2,2]` and class 1 on `[0,4]`.
- Gap: class 0 on `[-5,-3]` and class 1 on `[3,5]`.

For a reference population, choose an interval uniformly, then an input
uniformly within it. Define
$$r_c(x)=\sum_{(a,b)\in\mathcal I_c}\frac{\mathbf1\{a<x<b\}}{b-a},
\qquad \eta(x)=\frac{r_1(x)}{r_0(x)+r_1(x)}.$$
The common interval-mixture weight cancels. $\eta$ is defined only where the
denominator is positive. Equal-width overlapping intervals give
$\eta(x)=1/2$ in $(0,2)$, so the true conditional label entropy there is
$\log2$. Unequal widths would generally give a different conditional
probability. The gap scenario has no conditional reference inside its gap.

These are controlled populations for interpretation, not a claim that an
approximate dropout decomposition will recover their true uncertainty exactly.
"""
)
code(
    r"""
def reference_eta(grid, end_points):
    density = torch.zeros(len(grid), 2, dtype=grid.dtype)
    for label, ranges in end_points.items():
        for left, right in ranges:
            density[:, label] += ((grid[:, 0] > left) & (grid[:, 0] < right)).to(
                grid.dtype
            ) / (right - left)
    support = density.sum(-1) > 0
    eta = torch.full((len(grid),), float("nan"), dtype=grid.dtype)
    eta[support] = density[support, 1] / density[support].sum(-1)
    return eta, support


if RUN_APPENDICES and ready(1, 2):
    scenarios = [
        ("Overlapping classes", {0: [[-2, 2]], 1: [[0, 4]]}),
        ("Missing input region", {0: [[-5, -3]], 1: [[3, 5]]}),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    grid = torch.linspace(-7, 7, 501)[:, None]
    for column, (name, end_points) in enumerate(scenarios):
        x, y = interval_data(end_points)
        model, _ = fit_toy(x, y, seed=50 + column)
        torch.manual_seed(60 + column)
        terms = uncertainty_terms(mc_probabilities(model, grid, MC_DRAWS))
        eta, support = reference_eta(grid, end_points)
        axes[0, column].plot(
            grid[:, 0], terms["mean_probability"][:, 1], color=BLUE, label="Dropout mean"
        )
        axes[0, column].plot(
            grid[:, 0], eta, color="0.3", ls="--", label="Reference eta on support"
        )
        axes[0, column].scatter(x[:, 0], y, c=y, cmap="coolwarm", s=9)
        axes[0, column].set(title=name, ylabel="P(class1)", ylim=(-0.05, 1.05))
        for key, label, color in [
            ("predictive_entropy", "Predictive", BLUE),
            ("expected_entropy", "Conditional", GREEN),
            ("mutual_information", "MI", ORANGE),
        ]:
            axes[1, column].plot(grid[:, 0], terms[key], label=label, color=color)
        safe_eta = eta[support]
        true_entropy = -(
            torch.special.xlogy(safe_eta, safe_eta)
            + torch.special.xlogy(1 - safe_eta, 1 - safe_eta)
        )
        truth = torch.full_like(eta, float("nan"))
        truth[support] = true_entropy
        axes[1, column].plot(
            grid[:, 0], truth, color="0.3", ls="--", label="Reference label entropy"
        )
        axes[1, column].set(xlabel="x", ylabel="Nats", ylim=(-0.02, 0.74))
        axes[0, column].legend(fontsize=8)
        axes[1, column].legend(fontsize=8, ncol=2)
    plt.tight_layout()
    show_figure(fig, "geometry_scenarios")
"""
)
markdown(
    r"""
## Appendix B. Monte Carlo budget and batch redundancy

For a fixed fitted model and input, the probability mean has an iid MC SE
$s_p/\sqrt T$. Entropy and MI are nonlinear plug-in estimates and need not
obey an identical finite-sample SE formula. A single draw always gives zero
estimated disagreement. Increasing $T$ can stabilize acquisition rankings,
but cannot fix an inadequate approximate posterior.

The repeated experiment below changes only the dropout draws and their count.
It does not refit the model or collect labels. Its reference is a long MC
estimate under this model, not an exact Bayesian posterior quantity.
"""
)
code(
    r"""
if RUN_APPENDICES and ready(1, 2):
    points = torch.tensor([[-4.0], [-1.0], [1.7], [6.0]])
    torch.manual_seed(70)
    long_draws = mc_probabilities(toy_model, points, 5000)
    reference = uncertainty_terms(long_draws)["mutual_information"]
    rows = []
    for count in [1, 5, 20, 100, 500]:
        estimates = []
        for repeat in range(30):
            torch.manual_seed(7100 + 100 * count + repeat)
            draws = mc_probabilities(toy_model, points, count)
            estimate = categorical_entropy(draws.mean(0)) - categorical_entropy(draws).mean(0)
            estimates.append(estimate)
        estimates = torch.stack(estimates)
        rows.append(
            [
                count,
                estimates[:, 1].mean().item(),
                estimates[:, 1].std().item(),
                reference[1].item(),
            ]
        )
    table(
        [
            "MC draws",
            "Mean MI estimate at x=-1",
            "SD across MC repetitions",
            "5000-draw reference",
        ],
        rows,
    )
"""
)
markdown(
    r"""
For a batch $B$, joint information involves the joint predictive distribution
of all its labels. Two almost identical inputs can each have a large individual
BALD score while providing overlapping information. Our top-$k$ task uses
pointwise scores; it is not BatchBALD [4]. A diversity heuristic can reduce
neighbour redundancy but is not automatically a joint-information estimator.

Uncertainty is also not a universal out-of-distribution detector. A network
can extrapolate confidently, and a wrong approximation can make all its
stochastic predictions agree. Reliability must be evaluated against a stated
task, reference distribution, and scoring rule. Neither a compelling plot
nor high test accuracy alone establishes calibrated probabilities.
"""
)
markdown(
    r"""
## References and attribution

[1] Yarin Gal and Zoubin Ghahramani.
[Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.html).
ICML, 2016.

[2] Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel.
[Bayesian Active Learning for Classification and Preference Learning](https://arxiv.org/abs/1112.5745).
2011.

[3] Yarin Gal, Riashat Islam, and Zoubin Ghahramani.
[Deep Bayesian Active Learning with Image Data](https://proceedings.mlr.press/v70/gal17a.html).
ICML, 2017.

[4] Andreas Kirsch, Joost van Amersfoort, and Yarin Gal.
[BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning](https://proceedings.neurips.cc/paper/2019/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html).
NeurIPS, 2019.

The original interval dictionary, observation grid, sigmoid architecture,
dropout placement, and training schedule are adapted from the first example
in `spring_2025/seminar_ue_en.ipynb`. The supplied
`Sk_BML_BNN_ActiveLearning_Intro_201125 (2).ipynb`, credited there to Ivan
Nazarov, motivates the MNIST active-learning comparison. The new notes add
aligned exercises and solutions, coherent toy function draws, stable entropy,
explicit information access, paired repeats, and independent evaluation.
Both source notebooks are retained as supplied.
"""
)


def build(edition):
    solutions = edition == "solutions"
    cells = []
    for index, (kind, starter, solution, label) in enumerate(CELLS):
        metadata = {}
        if label:
            metadata["tags"] = [label if label == "written-answer" else f"task-{label}"]
        if kind == "figure":
            filename = f"{starter}.png"
            payload = base64.b64encode((HERE / "assets" / filename).read_bytes()).decode(
                "ascii"
            )
            cell = {
                "cell_type": "markdown",
                "metadata": metadata,
                "source": [
                    f'![{starter.replace("_"," ")}](attachment:{filename})\n\n{solution}\n'
                ],
                "attachments": {filename: {"image/png": payload}},
            }
        else:
            source = solution if solutions and solution is not None else starter
            if index == 0:
                source = source.replace(
                    "active learning\n",
                    "active learning — " + ("solutions" if solutions else "exercises") + "\n",
                    1,
                )
            if solutions and kind == "code":
                source = source.replace("RUN_APPENDICES = False", "RUN_APPENDICES = True")
                source = source.replace("SAVE_FIGURES = False", "SAVE_FIGURES = True")
            cell = {
                "cell_type": kind,
                "metadata": metadata,
                "source": source.splitlines(keepends=True),
            }
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
        cell["id"] = f"bnn-{index:03d}"
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9.19",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "bnn_seminar": {"edition": edition, "main_tasks": 4, "appendices": 2},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = HERE / f"bnn_{edition}.ipynb"
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Built {path.name}: {len(cells)} cells")


if __name__ == "__main__":
    build("student")
    build("solutions")
