"""GPU MNIST active-learning cells for the BNN seminar builder.

Importing this module only defines the cell factory. Data access, training and
figures happen in the generated notebooks, after the exercise checks pass.
"""


def add_mnist_section(markdown, code, exercise, answer):
    markdown(
        r"""
## Task 4 — Buy labels for handwritten digits

The one-dimensional experiment made the acquisition rule visible. Now use a
small dropout CNN on MNIST and compare three ways to spend the same label
budget: uniform random sampling, predictive entropy, and BALD.

Your task is to implement the Monte Carlo prediction routine used both for
acquisition and evaluation. The model, label oracle, training loop and plots
are supplied. Complete the CPU checks first. With `RUN_MNIST=True`, an
available CUDA device then enables the experiment; earlier tasks remain on
the CPU. The default experiment uses three paired repeats and usually takes
a few minutes on a recent GPU. Changing the budget is an experimental
decision: choose it before looking at the curves or test results.

### 6.1 What is compared?

Each repeat starts from the same 6,000 unlabelled training images and the
same 100 uniformly sampled labelled images for all three strategies. We
acquire 100 more labels at a time, ending with 500. Pool construction and
initial selection never inspect class labels. An oracle exposes labels only
for requested pool indices. All 100 initial labels count towards the budget.

At each budget we reset the CNN to the same seeded initialization and use
150 Adam updates with the same batch size and learning rate. The methods
therefore have equal training budgets, including the random control.
Acquisition computation is additional work for entropy and BALD. We select
the 100 largest pointwise scores without any batch-diversity correction;
several selected digits can be redundant. This is ordinary batch BALD
selection, not the joint-information method BatchBALD.

A separate, fixed set of 1,000 training images supplies evaluation labels
for the learning curves. These labels are a separate evaluation resource,
not part of the acquisition budget and never passed to fitting or selection.
The curves do not change the fixed stopping rule or hyperparameters. The
official test set is evaluated only after all acquisition runs finish.
BALD is not guaranteed to beat random sampling on a particular seed or at
this small compute budget.

### 6.2 Dropout at prediction time

`model.train()` would also update batch-normalization statistics if the
architecture contained batch normalization. Instead, the supplied context
manager evaluates the model, activates only dropout modules, and restores
every module's previous mode afterwards.

Return probabilities with shape `(T,N,10)`: Monte Carlo draw, image, class.
For classification, average probabilities, not logits. Standard PyTorch
dropout gives independent masks to different images. That suffices for
each image's marginal entropy and mutual information; a row of this tensor
is not a coherent function draw across images or input minibatches. The
one-dimensional plots used a shared mask for that different purpose.
"""
    )
    code(
        r'''
from contextlib import contextmanager
from pathlib import Path
import copy
import os
import time

MNIST_POOL_SIZE = 6_000
MNIST_VALIDATION_SIZE = 1_000
MNIST_INITIAL_LABELS = 100
MNIST_QUERY_SIZE = 100
MNIST_ACQUISITIONS = 4
MNIST_REPEATS = 3
MNIST_TRAIN_UPDATES = 150
MNIST_BATCH_SIZE = 64
MNIST_MC_DRAWS = 15
MNIST_RETRAIN = True
MNIST_DATA = Path(os.environ.get("BNN_MNIST_DATA", "/tmp/sk_bayes_mnist/data"))
MNIST_CACHE = Path(os.environ.get("BNN_MNIST_CACHE", "/tmp/sk_bayes_bnn_mnist"))
MNIST_STRATEGIES = ("random", "entropy", "bald")
MNIST_COLORS = {"random": "#787878", "entropy": "#d58b27", "bald": "#267b9e"}
mnist_enabled = False


@contextmanager
def mnist_dropout_only(model):
    """Enable only dropout and restore all module modes, also after errors."""
    previous = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        yield
    finally:
        for module, training in previous:
            module.training = training


def mnist_inputs(images, device):
    """Integer pixels -> standard MNIST normalization on the requested device."""
    return (images.to(device=device, dtype=torch.float32) / 255.0 - 0.1307) / 0.3081


def mnist_show_figure(figure, name):
    """Optional print-quality copies for the teacher's guide."""
    if globals().get("SAVE_FIGURES", False):
        destination = Path("assets")
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / name, bbox_inches="tight")
    plt.show()
'''
    )
    markdown(
        r"""
Complete `mc_cnn_probabilities`.

1. Inside `torch.inference_mode()` and `mnist_dropout_only(model)`, repeat
   prediction `n_draws` times.
2. For each draw, process all images in minibatches using `mnist_inputs`.
   Apply `softmax` over the class axis of the model's logits.
3. Concatenate image minibatches in their original order and stack draws.
   Return a CPU float32 tensor of shape `(n_draws,N,10)`, with no gradient
   history. Neither the model's modes nor batch-normalization statistics
   should change. Use the device of the model's first parameter.

The supplied argument checks also prevent an empty list of minibatches.
"""
    )
    exercise(
        "4",
        r'''
def mc_cnn_probabilities(model, images, n_draws=15, batch_size=512):
    """Predict probabilities: uint8 (N,1,28,28) -> CPU float32 (T,N,10)."""
    if n_draws < 1 or batch_size < 1 or len(images) == 0:
        raise ValueError("Use positive draws, positive batch size, and nonempty images.")
    # TODO: preserve modes, activate dropout, batch images, average no logits.
    raise NotImplementedError("Task 4")
''',
        r'''
def mc_cnn_probabilities(model, images, n_draws=15, batch_size=512):
    """Predict probabilities: uint8 (N,1,28,28) -> CPU float32 (T,N,10)."""
    if n_draws < 1 or batch_size < 1 or len(images) == 0:
        raise ValueError("Use positive draws, positive batch size, and nonempty images.")
    device = next(model.parameters()).device
    draws = []
    with torch.inference_mode(), mnist_dropout_only(model):
        for _ in range(n_draws):
            batches = []
            for start in range(0, len(images), batch_size):
                x = mnist_inputs(images[start : start + batch_size], device)
                batches.append(model(x).softmax(dim=-1).float().cpu())
            draws.append(torch.cat(batches, dim=0))
    return torch.stack(draws, dim=0)
''',
    )
    code(
        r"""
def test_mnist_mc_prediction():
    class Probe(nn.Module):
        def __init__(self, dropout):
            super().__init__()
            self.bn = nn.BatchNorm2d(1)
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(28 * 28, 10)

        def forward(self, x):
            return self.linear(self.dropout(self.bn(x).flatten(1)))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        images = torch.randint(0, 256, (9, 1, 28, 28), dtype=torch.uint8)
        deterministic = Probe(0.0).float()
        deterministic.eval()
        with torch.no_grad():
            deterministic.linear.weight.zero_()
            deterministic.linear.bias.copy_(torch.linspace(-1, 1, 10, dtype=torch.float32))
        expected = torch.softmax(torch.linspace(-1, 1, 10, dtype=torch.float32), dim=-1)
        probabilities = mc_cnn_probabilities(deterministic, images, 3, 4)
        torch.testing.assert_close(probabilities, expected.expand(3, 9, 10))
        assert probabilities.dtype == torch.float32 and probabilities.device.type == "cpu"
        assert not probabilities.requires_grad

        class PixelProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

            def forward(self, x):
                return x.flatten(1)[:, :10] + self.anchor

        # Input-sensitive oracle checks scaling and minibatch order independently.
        expected_pixels = ((images.float() / 255.0 - 0.1307) / 0.3081).flatten(1)
        expected_pixels = expected_pixels[:, :10].softmax(-1)
        ordered = mc_cnn_probabilities(PixelProbe(), images, 3, 4)
        torch.testing.assert_close(ordered, expected_pixels.expand(3, 9, 10))

        stochastic = Probe(0.5).float()
        stochastic.train()
        stochastic.bn.eval()  # Deliberately mixed modes must also be restored.
        before_modes = [module.training for module in stochastic.modules()]
        before_mean = stochastic.bn.running_mean.clone()
        before_var = stochastic.bn.running_var.clone()
        draws = mc_cnn_probabilities(stochastic, images, 8, 4)
        assert draws.shape == (8, 9, 10)
        assert torch.isfinite(draws).all() and (draws >= 0).all()
        torch.testing.assert_close(draws.sum(-1), torch.ones(8, 9, dtype=torch.float32))
        assert draws.var(dim=0).max() > 1e-5, "Dropout must remain stochastic."
        assert before_modes == [module.training for module in stochastic.modules()]
        torch.testing.assert_close(stochastic.bn.running_mean, before_mean)
        torch.testing.assert_close(stochastic.bn.running_var, before_var)

        try:
            with mnist_dropout_only(stochastic):
                raise RuntimeError("Check restoration on an interrupted prediction.")
        except RuntimeError:
            pass
        assert before_modes == [module.training for module in stochastic.modules()]


mnist_task_ready = False
try:
    test_mnist_mc_prediction()
    mnist_task_ready = True
    print("Task 4 CPU checks passed: probabilities, stochastic dropout, and mode restoration.")
except NotImplementedError:
    print("Complete Task 4 to enable the GPU experiment.")
"""
    )
    answer(
        r"""
The class probabilities must sum to one for each draw and image. If dropout
is disabled, all draws coincide and the mutual-information estimate is zero.
If logits are averaged before softmax, the resulting distribution is not the
Monte Carlo approximation to the posterior predictive. Restoring module
modes matters when this helper is called from a training notebook.
"""
    )
    markdown(
        r"""
### 6.3 A fixed-budget experiment

The first enabled run downloads MNIST through
[`torchvision.datasets.MNIST`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
The default data cache is shared with the normalizing-flow seminar; results
and fitted checkpoints are saved separately under `/tmp/sk_bayes_bnn_mnist`.
Set `BNN_MNIST_DATA` and `BNN_MNIST_CACHE` before running the notebook to
choose persistent locations. `MNIST_RETRAIN=True` fits on every enabled run;
set it to `False` to replay a matching completed experiment.

The following implementation is supplied. Notice that the acquisition
function receives images and pool indices, never a vector of hidden labels.
The oracle is a teaching interface, not a security boundary: it represents
the information a real annotator would reveal after a query.
"""
    )
    code(
        r'''
class MnistDropoutCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MnistLabelOracle:
    def __init__(self, hidden_labels):
        self._hidden_labels = hidden_labels.clone().cpu()
        self.queried = set()

    def query(self, pool_ids):
        requested = torch.as_tensor(pool_ids)
        if requested.ndim != 1 or requested.dtype not in (torch.int32, torch.int64):
            raise ValueError("Oracle IDs must be an integer vector, not floats or booleans.")
        ids = requested.to(device="cpu", dtype=torch.long).tolist()
        if len(ids) != len(set(ids)) or any(index in self.queried for index in ids):
            raise ValueError("Every new label must be queried exactly once.")
        if any(index < 0 or index >= len(self._hidden_labels) for index in ids):
            raise IndexError("Pool index outside the oracle's collection.")
        self.queried.update(ids)
        return self._hidden_labels[torch.tensor(ids, dtype=torch.long)]


def test_mnist_oracle_ids():
    oracle = MnistLabelOracle(torch.tensor([2, 4, 6]))
    for invalid in [[0.9], [False], [1, 1], [3]]:
        try:
            oracle.query(invalid)
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError("Oracle accepted an invalid ID request")
        assert not oracle.queried
    assert oracle.query([2, 0]).tolist() == [6, 2]


test_mnist_oracle_ids()


def mnist_uncertainty(draws):
    """Pointwise (predictive entropy, expected entropy, BALD), in nats."""
    probabilities = draws.clamp_min(torch.finfo(draws.dtype).tiny)
    mean = probabilities.mean(dim=0)
    predictive = -(mean * mean.log()).sum(dim=-1)
    expected = -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=0)
    return predictive, expected, (predictive - expected).clamp_min(0)


def mnist_metrics(mean_probabilities, labels):
    labels = labels.cpu()
    probabilities = mean_probabilities.cpu().clamp_min(1e-12)
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(labels).float()
    nll = -probabilities[torch.arange(len(labels)), labels].log().mean()
    ece = torch.tensor(0.0)
    edges = torch.linspace(0, 1, 16)
    for lower, upper in zip(edges[:-1], edges[1:]):
        inside = (confidence > lower) & (confidence <= upper)
        if inside.any():
            ece += (
                inside.float().mean()
                * (correct[inside].mean() - confidence[inside].mean()).abs()
            )
    return {"accuracy": float(correct.mean()), "nll": float(nll), "ece": float(ece)}


def mnist_fit(pool_inputs, labelled_ids, labelled_values, seed):
    """Fresh seeded initialization; the same number of optimizer updates."""
    torch.manual_seed(seed)
    model = MnistDropoutCNN().float().to(pool_inputs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = pool_inputs[torch.as_tensor(labelled_ids, device=pool_inputs.device)]
    y = labelled_values.to(pool_inputs.device)
    model.train()
    for _ in range(MNIST_TRAIN_UPDATES):
        batch = torch.randint(len(x), (MNIST_BATCH_SIZE,), device=x.device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x[batch]), y[batch])
        loss.backward()
        optimizer.step()
    return model


def mnist_run_active_learning(
    pool_images, hidden_labels, validation_images, validation_labels, device
):
    pool_inputs = mnist_inputs(pool_images, device)
    runs = []
    for repeat in range(MNIST_REPEATS):
        initial_generator = torch.Generator().manual_seed(18_000 + repeat)
        initial = torch.randperm(len(pool_images), generator=initial_generator)[
            :MNIST_INITIAL_LABELS
        ]
        for strategy in MNIST_STRATEGIES:
            oracle = MnistLabelOracle(hidden_labels)
            labelled_ids = initial.tolist()
            labelled_values = oracle.query(labelled_ids)
            random_generator = torch.Generator().manual_seed(28_000 + repeat)
            history, acquisitions = [], []
            first_diagnostic = None
            model = None
            for round_index in range(MNIST_ACQUISITIONS + 1):
                seed = 38_000 + 100 * repeat + round_index
                model = mnist_fit(pool_inputs, labelled_ids, labelled_values, seed)
                torch.manual_seed(seed + 100_000)
                validation_draws = mc_cnn_probabilities(
                    model, validation_images, MNIST_MC_DRAWS
                )
                metrics = mnist_metrics(validation_draws.mean(0), validation_labels)
                history.append({"labels": len(labelled_ids), **metrics})
                print(
                    f"repeat {repeat + 1}/{MNIST_REPEATS} | {strategy:7s} | "
                    f"labels {len(labelled_ids):3d} | validation accuracy "
                    f"{metrics['accuracy']:.3f}"
                )
                if round_index == MNIST_ACQUISITIONS:
                    break

                labelled_mask = torch.zeros(len(pool_images), dtype=torch.bool)
                labelled_mask[labelled_ids] = True
                candidates = torch.arange(len(pool_images))[~labelled_mask]
                n_query = min(MNIST_QUERY_SIZE, len(candidates))
                if strategy == "random":
                    local_selection = torch.randperm(
                        len(candidates), generator=random_generator
                    )[:n_query]
                    scores = None
                else:
                    torch.manual_seed(seed + 200_000)
                    candidate_draws = mc_cnn_probabilities(
                        model, pool_images[candidates], MNIST_MC_DRAWS
                    )
                    predictive, expected, information = mnist_uncertainty(candidate_draws)
                    scores = predictive if strategy == "entropy" else information
                    # Stable sorting makes equal-score ties reproducible.
                    local_selection = torch.argsort(scores, descending=True, stable=True)[
                        :n_query
                    ]
                    if repeat == 0 and round_index == 0 and strategy == "bald":
                        first_diagnostic = {
                            "pool_ids": candidates.clone(),
                            "predictive": predictive.clone(),
                            "expected": expected.clone(),
                            "information": information.clone(),
                            "probabilities": candidate_draws.mean(0),
                            "selected_local": local_selection.clone(),
                        }
                selected = candidates[local_selection]
                # The first and only label lookup for these selected examples.
                revealed = oracle.query(selected.tolist())
                acquisitions.append(
                    {
                        "pool_ids": selected.clone(),
                        "labels": revealed.clone(),
                        "scores": None if scores is None else scores[local_selection].clone(),
                    }
                )
                labelled_ids.extend(selected.tolist())
                labelled_values = torch.cat([labelled_values, revealed])
                assert len(oracle.queried) == len(labelled_ids)
            assert (
                len(labelled_ids)
                == MNIST_INITIAL_LABELS + MNIST_ACQUISITIONS * MNIST_QUERY_SIZE
            )
            runs.append(
                {
                    "repeat": repeat,
                    "strategy": strategy,
                    "history": history,
                    "initial_ids": initial.clone(),
                    "acquisitions": acquisitions,
                    "diagnostic": first_diagnostic,
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                }
            )
    # Paired methods must start with exactly the same labels and fit.
    for repeat in range(MNIST_REPEATS):
        paired = [run for run in runs if run["repeat"] == repeat]
        assert all(torch.equal(paired[0]["initial_ids"], run["initial_ids"]) for run in paired)
        assert max(run["history"][0]["accuracy"] for run in paired) == min(
            run["history"][0]["accuracy"] for run in paired
        )
    return runs
'''
    )
    code(
        r"""
mnist_results = None
if not mnist_task_ready:
    print("MNIST experiment skipped: complete Task 4 first.")
elif not globals().get("RUN_MNIST", True):
    print("MNIST experiment disabled by RUN_MNIST=False.")
elif not torch.cuda.is_available():
    print("No CUDA device: CPU checks passed; GPU training is explicitly skipped.")
else:
    from torchvision.datasets import MNIST

    mnist_device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(mnist_device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    MNIST_CACHE.mkdir(parents=True, exist_ok=True)
    mnist_configuration = {
        "version": 1,
        "pool": MNIST_POOL_SIZE,
        "validation": MNIST_VALIDATION_SIZE,
        "initial": MNIST_INITIAL_LABELS,
        "query": MNIST_QUERY_SIZE,
        "acquisitions": MNIST_ACQUISITIONS,
        "repeats": MNIST_REPEATS,
        "updates": MNIST_TRAIN_UPDATES,
        "batch": MNIST_BATCH_SIZE,
        "draws": MNIST_MC_DRAWS,
    }
    mnist_checkpoint = MNIST_CACHE / "bnn_mnist_active_learning.pt"
    training_data = MNIST(str(MNIST_DATA), train=True, download=True)
    partition_generator = torch.Generator().manual_seed(7_026)
    partition = torch.randperm(len(training_data), generator=partition_generator)
    pool_ids = partition[:MNIST_POOL_SIZE]
    validation_ids = partition[MNIST_POOL_SIZE : MNIST_POOL_SIZE + MNIST_VALIDATION_SIZE]
    assert len(set(pool_ids.tolist()) & set(validation_ids.tolist())) == 0
    mnist_pool_images = training_data.data[pool_ids, None].clone()
    mnist_validation_images = training_data.data[validation_ids, None].clone()
    mnist_validation_labels = training_data.targets[validation_ids].clone()
    if not MNIST_RETRAIN and mnist_checkpoint.exists():
        cached = torch.load(mnist_checkpoint, map_location="cpu", weights_only=False)
        if cached["configuration"] != mnist_configuration:
            raise ValueError(
                "Cached experiment has a different configuration; set MNIST_RETRAIN=True."
            )
        mnist_results = cached
        print("Loaded completed experiment:", mnist_checkpoint)
    else:
        start = time.perf_counter()
        runs = mnist_run_active_learning(
            mnist_pool_images,
            training_data.targets[pool_ids],
            mnist_validation_images,
            mnist_validation_labels,
            mnist_device,
        )

        # All fits, acquisition choices, and stopping decisions are complete.
        test_data = MNIST(str(MNIST_DATA), train=False, download=True)
        test_images = test_data.data[:, None]
        for run in runs:
            model = MnistDropoutCNN().float().to(mnist_device)
            model.load_state_dict(run["state_dict"])
            torch.manual_seed(500_000 + run["repeat"])
            test_draws = mc_cnn_probabilities(model, test_images, MNIST_MC_DRAWS)
            run["test"] = mnist_metrics(test_draws.mean(0), test_data.targets)
        torch.cuda.synchronize()
        mnist_results = {
            "configuration": mnist_configuration,
            "runs": runs,
            "runtime_seconds": time.perf_counter() - start,
            "gpu": torch.cuda.get_device_name(mnist_device),
        }
        torch.save(mnist_results, mnist_checkpoint)
        print(
            f"Completed in {mnist_results['runtime_seconds']:.1f} seconds; saved {mnist_checkpoint}"
        )
    mnist_enabled = True
    print("Final test metrics, after the fixed acquisition budgets:")
    print("strategy     accuracy (mean ± repeat SD)       NLL       ECE")
    for strategy in MNIST_STRATEGIES:
        selected = [run for run in mnist_results["runs"] if run["strategy"] == strategy]
        accuracies = np.array([run["test"]["accuracy"] for run in selected])
        nll = np.mean([run["test"]["nll"] for run in selected])
        ece = np.mean([run["test"]["ece"] for run in selected])
        print(
            f"{strategy:8s}       {accuracies.mean():.3f} ± {accuracies.std(ddof=1):.3f}"
            f"                 {nll:.3f}     {ece:.3f}"
        )
"""
    )
    markdown(
        r"""
### 6.4 Learning curves and paired outcomes

The bands below show one standard deviation across repeats, not a
confidence interval. With three repeats, treat small differences cautiously.
The connected endpoint dots show the pairing: each line joins strategies
that started with the same labelled images and model seed. Accuracy and log
loss answer different questions; confidence can worsen while accuracy rises.
"""
    )
    code(
        r"""
if mnist_enabled:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), constrained_layout=True)
    for strategy in MNIST_STRATEGIES:
        selected = [run for run in mnist_results["runs"] if run["strategy"] == strategy]
        budgets = [point["labels"] for point in selected[0]["history"]]
        for ax, metric, label in zip(
            axes[:2],
            ("accuracy", "nll"),
            ("Validation accuracy", "Validation negative log likelihood"),
        ):
            values = np.array(
                [[point[metric] for point in run["history"]] for run in selected]
            )
            mean, spread = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(budgets, mean, marker="o", color=MNIST_COLORS[strategy], label=strategy)
            ax.fill_between(
                budgets, mean - spread, mean + spread, color=MNIST_COLORS[strategy], alpha=0.13
            )
            ax.set(xlabel="Queried labels, including initial 100", ylabel=label)
            ax.grid(alpha=0.18)
    for repeat in range(MNIST_REPEATS):
        paired = [
            next(
                run
                for run in mnist_results["runs"]
                if run["repeat"] == repeat and run["strategy"] == strategy
            )
            for strategy in MNIST_STRATEGIES
        ]
        endpoints = [run["test"]["accuracy"] for run in paired]
        axes[2].plot(range(3), endpoints, color="#aaaaaa", alpha=0.6, linewidth=1)
        axes[2].scatter(
            range(3),
            endpoints,
            color=[MNIST_COLORS[s] for s in MNIST_STRATEGIES],
            s=45,
            zorder=3,
        )
    axes[0].legend(frameon=False)
    axes[2].set(
        xticks=range(3),
        xticklabels=MNIST_STRATEGIES,
        ylabel="Final test accuracy",
        title="Paired repeats at 500 labels",
    )
    axes[2].grid(axis="y", alpha=0.18)
    mnist_show_figure(fig, "mnist_learning_curves.pdf")
"""
    )
    markdown(
        r"""
### 6.5 What does BALD buy?

At the first query, the entropy and BALD models are exactly the same fitted
CNN. The scatter separates total predictive uncertainty from disagreement
between dropout predictions. A digit can have high predictive entropy but
relatively small mutual information when the sampled models agree that it
is ambiguous. Both quantities depend on the approximation and fit; they
are not ground-truth labels for two kinds of uncertainty.

The right-hand panel shows the high-BALD batch on top of the full candidate
pool. Selection uses only the score; class labels are revealed afterwards.
"""
    )
    code(
        r"""
if mnist_enabled:
    diagnostic_run = next(
        run
        for run in mnist_results["runs"]
        if run["repeat"] == 0 and run["strategy"] == "bald"
    )
    diagnostic = diagnostic_run["diagnostic"]
    entropy = diagnostic["predictive"].numpy()
    information = diagnostic["information"].numpy()
    selected = diagnostic["selected_local"].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    expected = diagnostic["expected"].numpy()
    scatter = axes[0].scatter(entropy, information, c=expected, cmap="viridis", s=9, alpha=0.6)
    limit = max(entropy.max(), information.max())
    axes[0].plot([0, limit], [0, limit], color="#888888", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Predictive entropy (nats)",
        ylabel="BALD / mutual information (nats)",
        title="Same model, two acquisition criteria",
    )
    fig.colorbar(scatter, ax=axes[0], label="Expected conditional entropy (nats)")
    axes[1].scatter(
        entropy, information, color="#bbbbbb", s=9, alpha=0.3, label="Candidate pool"
    )
    axes[1].scatter(
        entropy[selected],
        information[selected],
        color=MNIST_COLORS["bald"],
        s=23,
        alpha=0.9,
        label="100 purchased labels",
    )
    axes[1].set(
        xlabel="Predictive entropy (nats)",
        ylabel="BALD / mutual information (nats)",
        title="First BALD query",
    )
    axes[1].legend(frameon=False)
    mnist_show_figure(fig, "mnist_uncertainty.pdf")
"""
    )
    code(
        r"""
if mnist_enabled:
    fig, axes = plt.subplots(3, 8, figsize=(13, 6), constrained_layout=True)
    for row, strategy in enumerate(MNIST_STRATEGIES):
        run = next(
            run
            for run in mnist_results["runs"]
            if run["repeat"] == 0 and run["strategy"] == strategy
        )
        purchase = run["acquisitions"][0]
        for column, ax in enumerate(axes[row]):
            pool_id = int(purchase["pool_ids"][column])
            ax.imshow(mnist_pool_images[pool_id, 0], cmap="gray", vmin=0, vmax=255)
            label = int(purchase["labels"][column])
            score = purchase["scores"]
            title = f"revealed {label}"
            if score is not None:
                title += f"\nscore {float(score[column]):.2f}"
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(strategy, color=MNIST_COLORS[strategy], fontsize=11)
    fig.suptitle("First acquired examples: labels are shown only after each query")
    mnist_show_figure(fig, "mnist_acquired_digits.pdf")
"""
    )
    markdown(
        r"""
### 6.6 Annotation mix

This diagnostic inspects the labels that have already been purchased. It
does not inspect the remaining pool labels. A changing class mix can expose
redundant queries or a class-specific weakness, but class balance alone is
not an acquisition objective and does not establish which strategy is best.
"""
    )
    code(
        r"""
if mnist_enabled:
    counts = []
    for strategy in MNIST_STRATEGIES:
        rounds = []
        for round_index in range(MNIST_ACQUISITIONS):
            labels = torch.cat(
                [
                    run["acquisitions"][round_index]["labels"]
                    for run in mnist_results["runs"]
                    if run["strategy"] == strategy
                ]
            )
            rounds.append(torch.bincount(labels, minlength=10).numpy() / MNIST_REPEATS)
        counts.append(np.array(rounds))
    vmax = max(values.max() for values in counts)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, strategy, values in zip(axes, MNIST_STRATEGIES, counts):
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for row in range(values.shape[0]):
            for column in range(10):
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.0f}",
                    ha="center",
                    va="center",
                    color="white" if values[row, column] > 0.6 * vmax else "#333333",
                    fontsize=8,
                )
        ax.set(
            xticks=range(10),
            yticks=range(MNIST_ACQUISITIONS),
            yticklabels=[
                f"{100 * (i + 1)}→{100 * (i + 2)}" for i in range(MNIST_ACQUISITIONS)
            ],
            xlabel="Revealed class",
            ylabel="Budget before → after query",
            title=strategy,
        )
    fig.colorbar(image, ax=axes, label="Mean purchased examples per repeat")
    mnist_show_figure(fig, "mnist_acquired_classes.pdf")
"""
    )
    markdown(
        r"""
Discuss the results before changing the experiment.

1. Does the ranking persist across the paired repeats? Which differences
   are large relative to the spread?
2. Why can entropy and BALD select different digits from the same CNN?
   Why can pointwise selection buy many similar digits at once?
3. What resources are omitted from a simple label-budget comparison?
   Account for the separate evaluation labels and acquisition computation.
4. What would you change to study calibration, distribution shift, or joint
   batch selection? State a new protocol before using the test set again.
"""
    )
    answer(
        r"""
There is no required winner. BALD targets disagreement under this dropout
approximation, while predictive entropy also responds to uncertainty shared
by the sampled models. Either score can be poor when the approximation or
fit is poor. Small-data CNN training and three repeats introduce substantial
variation. Equal labelled budgets and training updates make the comparison
interpretable, but do not make wall-clock costs equal.

A useful follow-up is a separately planned experiment with diversity-aware
batch selection or a distribution shift. Reusing the observed test results
to choose a new method would turn the test set into a development set.
"""
    )
