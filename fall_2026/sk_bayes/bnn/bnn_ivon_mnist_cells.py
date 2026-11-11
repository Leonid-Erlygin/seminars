"""Supplied GPU IVON/dropout comparison for the BNN seminar builder."""


def add_ivon_mnist_section(markdown, code):
    markdown(
        r"""
### Supplied demonstration: IVON and dropout on MNIST

This is a demonstration, not an additional task. It runs even when the
exercise functions above are unfinished. We fit the same CNN to the same
1,000 uniformly sampled training images, once with dropout and Adam and
once with IVON and dropout disabled. Each of three paired repeats uses
identical initial parameter means and an identical stream of minibatch
indices. Both methods receive 600 gradient evaluations, with batch size
64. Dropout masks and IVON weight samples are different sources of noise;
equal random seeds do not make those samples equivalent.

We use fixed learning rates of 0.001 for Adam and 0.1 for IVON, and a
regularization coefficient of $10^{-4}$ for both. IVON uses
`ess=1000`, `hess_init=1`, `beta2=0.99`, and `clip_radius=0.1`.
It sees the mean cross-entropy of each minibatch; its optimizer handles
the Gaussian-prior contribution, so we do not add another KL penalty.
The two approximations and optimizers differ; this is a small teaching
comparison with a matched gradient budget, not a benchmark of their best
possible performance.

A disjoint set of 1,000 training images is used for validation. During
development, constant IVON learning rates 0.1 and 0.03 and one cosine
schedule were checked on this validation split; the fixed 0.1 setting is
shown here. There is no per-repeat tuning, checkpoint selection, or early
stopping. The official test set is loaded only after all six fits finish.
The table reports mean and sample standard deviation over the three
paired repeats. ECE uses 15 equal-width confidence bins and is sensitive
to the binning choice. The multiclass Brier score is
$N^{-1}\sum_i\sum_c(\bar p_{ic}-\mathbf1[y_i=c])^2$.

`RUN_IVON=False` disables this entire supplied comparison;
`RUN_MNIST=False` disables its GPU part. The data cache is shared with
Task 4, but the fitted models and training labels are chosen independently
of its active-learning experiment.
"""
    )
    code(
        r'''
IVON_MNIST_LABELS = 1_000
IVON_MNIST_VALIDATION = 1_000
IVON_MNIST_REPEATS = 3
IVON_MNIST_UPDATES = 600
IVON_MNIST_BATCH_SIZE = 64
IVON_MNIST_DRAWS = 30
IVON_MNIST_COLORS = {"dropout": "#d58b27", "ivon": "#267b9e"}


def ivon_mnist_dropout_predictions(model, images, n_draws=30, batch_size=512):
    """Supplied MC prediction, independent of the Task 4 implementation."""
    device = next(model.parameters()).device
    samples = []
    with torch.inference_mode(), mnist_dropout_only(model):
        for _ in range(n_draws):
            batches = []
            for start in range(0, len(images), batch_size):
                x = mnist_inputs(images[start:start + batch_size], device)
                batches.append(model(x).softmax(-1).float().cpu())
            samples.append(torch.cat(batches))
    return torch.stack(samples)


def ivon_mnist_metrics(draws, labels):
    mean = draws.mean(0)
    metrics = mnist_metrics(mean, labels)
    targets = torch.nn.functional.one_hot(labels.cpu(), num_classes=10).float()
    metrics["brier"] = float((mean - targets).square().sum(-1).mean())
    return metrics


def ivon_mnist_fit(images, labels, method, seed):
    """Identical initialization and batch stream; one gradient per update."""
    device = torch.device("cuda")
    torch.manual_seed(seed)
    model = MnistDropoutCNN().float().to(device)
    initial_parameters = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu()
    if method == "ivon":
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        optimizer = ivon.IVON(
            model.parameters(), lr=0.1, ess=len(labels), hess_init=1.0,
            beta2=0.99, clip_radius=0.1, weight_decay=1e-4,
        )
    elif method == "dropout":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        raise ValueError("Unknown method")
    x, y = mnist_inputs(images, device), labels.to(device)
    batch_generator = torch.Generator(device=device).manual_seed(seed + 1_000)
    model.train()
    for _ in range(IVON_MNIST_UPDATES):
        indices = torch.randint(
            len(y), (IVON_MNIST_BATCH_SIZE,), device=device, generator=batch_generator
        )
        optimizer.zero_grad(set_to_none=True)
        if method == "ivon":
            with ivon_weight_sample(model, optimizer, train=True):
                loss = torch.nn.functional.cross_entropy(model(x[indices]), y[indices])
                loss.backward()
        else:
            loss = torch.nn.functional.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
        optimizer.step()
    return model, optimizer, initial_parameters


def ivon_mnist_predict(model, optimizer, images):
    if optimizer is None:
        return ivon_mnist_dropout_predictions(model, images, IVON_MNIST_DRAWS)
    device = next(model.parameters()).device
    return ivon_predict_probabilities(
        model, optimizer, mnist_inputs(images, device), IVON_MNIST_DRAWS, 512
    )


ivon_mnist_results = None
if not globals().get("RUN_IVON", True):
    print("Supplied IVON MNIST comparison disabled by RUN_IVON=False.")
elif not globals().get("ivon_available", False):
    print("Supplied IVON MNIST comparison skipped: install ivon-opt and rerun the IVON import cell.")
elif not globals().get("RUN_MNIST", True):
    print("Supplied IVON MNIST comparison disabled by RUN_MNIST=False.")
elif not torch.cuda.is_available():
    print("Supplied IVON MNIST comparison requires CUDA; the one-dimensional example remains available.")
else:
    from torchvision.datasets import MNIST

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    training_data = MNIST(str(MNIST_DATA), train=True, download=True)
    partition = torch.randperm(
        len(training_data), generator=torch.Generator().manual_seed(92_026)
    )
    train_ids = partition[:IVON_MNIST_LABELS]
    valid_ids = partition[IVON_MNIST_LABELS:IVON_MNIST_LABELS + IVON_MNIST_VALIDATION]
    assert not set(train_ids.tolist()) & set(valid_ids.tolist())
    train_images = training_data.data[train_ids, None]
    train_labels = training_data.targets[train_ids]
    valid_images = training_data.data[valid_ids, None]
    valid_labels = training_data.targets[valid_ids]
    fits, validation_rows = [], []
    for repeat in range(IVON_MNIST_REPEATS):
        paired_initials = []
        for method in ("dropout", "ivon"):
            model, optimizer, initial = ivon_mnist_fit(
                train_images, train_labels, method, seed=8_026 + repeat
            )
            paired_initials.append(initial)
            torch.manual_seed(102_026 + repeat)
            draws = ivon_mnist_predict(
                model, optimizer if method == "ivon" else None, valid_images
            )
            metrics = ivon_mnist_metrics(draws, valid_labels)
            validation_rows.append({"repeat": repeat, "method": method, **metrics})
            fits.append((repeat, method, model, optimizer))
            print(
                f"repeat {repeat + 1}/{IVON_MNIST_REPEATS} | {method:7s} | "
                f"validation accuracy {metrics['accuracy']:.3f}, NLL {metrics['nll']:.3f}"
            )
        torch.testing.assert_close(paired_initials[0], paired_initials[1], rtol=0, atol=0)

    # Training, hyperparameters and stopping are fixed before test labels are loaded.
    test_data = MNIST(str(MNIST_DATA), train=False, download=True)
    test_images, test_labels = test_data.data[:, None], test_data.targets
    test_rows, first_draws = [], {}
    for repeat, method, model, optimizer in fits:
        torch.manual_seed(202_026 + repeat)
        draws = ivon_mnist_predict(
            model, optimizer if method == "ivon" else None, test_images
        )
        test_rows.append({"repeat": repeat, "method": method, **ivon_mnist_metrics(draws, test_labels)})
        if repeat == 0:
            first_draws[method] = draws
    torch.cuda.synchronize()
    ivon_mnist_results = {
        "validation": validation_rows,
        "test": test_rows,
        "draws": first_draws,
        "test_images": test_images,
        "test_labels": test_labels,
        "seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(f"\nOfficial MNIST test set; {IVON_MNIST_REPEATS} paired repeats; mean ± sample SD")
    print(f"{'method':10s} {'accuracy':18s} {'NLL':18s} {'Brier':18s} {'ECE (15 bins)':18s}")
    for method in ("dropout", "ivon"):
        cells = []
        for metric in ("accuracy", "nll", "brier", "ece"):
            values = torch.tensor([row[metric] for row in test_rows if row["method"] == method])
            cells.append(f"{values.mean():.4f} ± {values.std(unbiased=True):.4f}")
        print(f"{method:10s} " + "  ".join(cells))
    print(f"GPU: {ivon_mnist_results['gpu']}; elapsed: {ivon_mnist_results['seconds']:.1f} s")
    del fits, model, optimizer, draws
    torch.cuda.empty_cache()
'''
    )
    code(
        r'''
if ivon_mnist_results is not None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for axis, metric, title in zip(
        axes[:3], ("accuracy", "nll", "brier"),
        ("Test accuracy ↑", "Test predictive NLL ↓", "Test Brier score ↓"),
    ):
        for position, method in enumerate(("dropout", "ivon")):
            values = torch.tensor([
                row[metric] for row in ivon_mnist_results["test"] if row["method"] == method
            ])
            axis.bar(position, float(values.mean()), color=IVON_MNIST_COLORS[method], alpha=0.8)
            axis.errorbar(
                position, float(values.mean()), yerr=float(values.std(unbiased=True)),
                color="black", capsize=5, fmt="none",
            )
            axis.scatter(torch.full_like(values, position).numpy(), values.numpy(), color="black", s=12)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["MC dropout", "IVON"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, 1)
    axis = axes[3]
    edges = torch.linspace(0, 1, 16)
    for method in ("dropout", "ivon"):
        mean = ivon_mnist_results["draws"][method].mean(0)
        confidence, prediction = mean.max(-1)
        correct = prediction.eq(ivon_mnist_results["test_labels"]).float()
        xs, ys = [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            inside = (confidence > lower) & (confidence <= upper)
            if inside.any():
                xs.append(float(confidence[inside].mean()))
                ys.append(float(correct[inside].mean()))
        ece = next(row["ece"] for row in ivon_mnist_results["test"] if row["method"] == method)
        axis.plot(xs, ys, "o-", color=IVON_MNIST_COLORS[method], label=f"{method}: ECE {ece:.3f}", ms=4)
    axis.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    axis.set(xlabel="Mean confidence", ylabel="Fraction correct", xlim=(0, 1), ylim=(0, 1), title="Reliability: first paired repeat")
    axis.legend(fontsize=8)
    fig.suptitle("Same 1,000 labels and CNN; 600 updates per fit; 30 predictive draws")
    fig.tight_layout()
    mnist_show_figure(fig, "ivon_mnist_comparison.pdf")

    # Selection uses probabilities only; true labels are displayed afterwards.
    average_entropy = torch.stack([
        mnist_uncertainty(ivon_mnist_results["draws"][method])[0]
        for method in ("dropout", "ivon")
    ]).mean(0)
    example_ids = torch.argsort(average_entropy, descending=True, stable=True)[:4]
    fig, axes = plt.subplots(3, len(example_ids), figsize=(13, 7), gridspec_kw={"height_ratios": [1.1, 1, 1]})
    for column, index in enumerate(example_ids.tolist()):
        image_axis = axes[0, column]
        image_axis.imshow(ivon_mnist_results["test_images"][index, 0], cmap="gray_r")
        image_axis.set_title(f"Test image {index}; label {int(ivon_mnist_results['test_labels'][index])}")
        image_axis.axis("off")
        for row, method in enumerate(("dropout", "ivon"), start=1):
            draws = ivon_mnist_results["draws"][method][:, index]
            mean, deviation = draws.mean(0), draws.std(0, unbiased=True)
            axis = axes[row, column]
            axis.bar(range(10), mean.numpy(), color=IVON_MNIST_COLORS[method], alpha=0.8)
            axis.errorbar(range(10), mean.numpy(), yerr=deviation.numpy(), fmt="none", ecolor="0.2", capsize=2, lw=0.8)
            axis.set(xticks=range(10), ylim=(0, 1.05), xlabel="Digit class")
            entropy = float(-(mean * mean.clamp_min(1e-12).log()).sum())
            axis.set_title(f"{method}; predictive entropy {entropy:.2f} nats", fontsize=10)
            if column == 0:
                axis.set_ylabel("Probability")
    fig.suptitle("Four largest mean predictive entropies across the two methods\nBars: posterior predictive mean; error bars: SD across weight or dropout draws", y=1.01)
    fig.tight_layout()
    mnist_show_figure(fig, "ivon_mnist_examples.pdf")
'''
    )
    markdown(
        r"""
Read the comparison as several questions. Which approximation predicts
the correct digit more often? Which assigns better probabilities to the
observed labels? Does either become overconfident? Accuracy, NLL, Brier
score and ECE measure different properties, so a method need not win all
four. The uncertainty bars show variation of sampled predictions, not
confidence intervals for the unknown true class probabilities.

The digit panel deliberately selects the four largest average predictive
entropies across both methods, using no test labels. These images explain
where the fitted models are uncertain; they are not a representative
sample for estimating overall performance. A more sophisticated posterior
approximation does not by itself guarantee better predictions at this
small label and optimization budget.
"""
    )
