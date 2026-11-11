"""Supplied IVON demonstrations; no additional student exercises.

All runtime code is embedded in the generated notebooks. The official IVON
optimizer is an explicit dependency rather than a local reimplementation.
"""

from bnn_ivon_mnist_cells import add_ivon_mnist_section


def add_ivon_section(markdown, code):
    markdown(
        r"""
## Supplied example — Explicit variational weights with IVON

This example is fully implemented in both notebooks. There is no task to
complete: use the saved solution figures for a short demonstration, or run
the supplied cells to reproduce them. The four seminar tasks are unchanged.

MC dropout samples hidden-feature masks. Another approximation learns an
explicit distribution over network weights,
$$q(w)=\mathcal N\!\left(w;\mu,\operatorname{diag}(\sigma^2)\right),\qquad
w=\mu+\sigma\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I).$$
With $N$ observations, variational inference minimizes
$$J(q)=\mathbb E_q\!\left[-\frac1N\sum_{i=1}^N\log p(y_i\mid x_i,w)\right]
       +\frac1N D_{\rm KL}(q\Vert p(w)).$$
Both the weight means and their uncertainty must be estimated. IVON
(Improved Variational Online Newton, ICML 2024) optimizes this objective
using stochastic weight draws and an efficient curvature estimate [5]. Its
posterior here is still a diagonal Gaussian: newer optimization does not
automatically make that approximation more expressive or better calibrated.

### Read the training loop

The official `ivon-opt` package provides the optimizer. With mean cross
entropy, `ess=N`, and `weight_decay=delta`, its convention corresponds to
an isotropic Gaussian prior of precision $N\delta$. Its standard deviations
are $\sigma_j=[N(h_j+\delta)]^{-1/2}$, where $h$ is the optimizer's curvature
state. `hess_init` initializes that state; it is not the prior precision.
The optimizer handles the prior term: do not add another KL or L2 penalty
to the sampled cross entropy. `loss.backward()` runs inside
`optimizer.sampled_params(train=True)`; `optimizer.step()` runs after the
context restores the means. At prediction time, sample with `train=False`,
apply softmax, and average probabilities. These cells supply a wrapper that
also restores parameters if prediction is interrupted.

The demonstration runs without completing any earlier task. Install
`ivon-opt==0.1.3` through `requirements.txt`; if it is missing, these cells
print an installation instruction and leave the earlier tasks runnable.
"""
    )
    code(
        r'''
ivon_available = False
if RUN_IVON:
    try:
        import ivon
        ivon_available = True
    except ImportError:
        print("Supplied IVON example skipped. Install ivon-opt==0.1.3 and rerun this cell.")
else:
    print("Supplied IVON example disabled by RUN_IVON=False.")


@contextmanager
def ivon_weight_sample(model, optimizer, train=False):
    """Official weight sampling, with restoration even after an exception."""
    parameters = list(model.parameters())
    means = [parameter.detach().clone() for parameter in parameters]
    modes = [(module, module.training) for module in model.modules()]
    try:
        with optimizer.sampled_params(train=train):
            yield
    finally:
        with torch.no_grad():
            for parameter, mean in zip(parameters, means):
                parameter.copy_(mean)
        for module, training in modes:
            module.training = training


def ivon_predict_probabilities(model, optimizer, inputs, n_draws=200, batch_size=512):
    """One weight vector per draw, held fixed across all input minibatches."""
    if n_draws < 1 or batch_size < 1 or len(inputs) == 0:
        raise ValueError("Use positive draws/batch size and nonempty inputs.")
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    draws = []
    try:
        model.eval()  # IVON samples weights; no additional dropout masks.
        with torch.no_grad():
            for _ in range(n_draws):
                with ivon_weight_sample(model, optimizer):
                    batches = [
                        model(inputs[start:start + batch_size].to(device)).softmax(-1).float().cpu()
                        for start in range(0, len(inputs), batch_size)
                    ]
                    draws.append(torch.cat(batches))
    finally:
        for module, training in modes:
            module.training = training
    return torch.stack(draws)


def demo_dropout_probabilities(model, inputs, n_draws=200):
    """Supplied version: the demonstration does not depend on Task 1."""
    draws = []
    for _ in range(n_draws):
        with sampled_model(model):
            draws.append(model(inputs).softmax(-1).cpu())
    return torch.stack(draws)


def demo_uncertainty(draws):
    """Supplied decomposition, independent of the Task 2 implementation."""
    def entropy(p):
        return -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
    mean = draws.mean(0)
    predictive, expected = entropy(mean), entropy(draws).mean(0)
    return {
        "mean_probability": mean,
        "predictive_entropy": predictive,
        "expected_entropy": expected,
        "mutual_information": (predictive - expected).clamp_min(0),
        "probability_mc_se": draws.std(0) / math.sqrt(len(draws)),
    }


def ivon_posterior_sd(optimizer):
    return torch.cat([
        (group["ess"] * (group["hess"] + group["weight_decay"])).rsqrt()
        for group in optimizer.param_groups
    ]).detach().cpu()
'''
    )
    markdown(
        r"""
### Compare the same interval classifier

Both methods use the original 120 observations and the same supplied tanh
network from the active-learning experiment: `1 → 32 → 32 → 2`, with input
scaling $x/5$. The original deep sigmoid network stalled near chance during
IVON development runs; using this shared architecture gives both methods a
usable fit without changing the motivating data. Each paired repeat
starts from exactly the same weights and performs 4,000 full-batch updates
with one stochastic gradient evaluation per update. The dropout model keeps
probability 0.2 and the supplied tanh network's Adam/OneCycle recipe
(maximum learning rate 0.03). The IVON
model disables dropout and samples all weights instead. Prediction uses
200 draws for each method. Three initialization seeds show how much the
comparison changes when the models are refitted.

Optimizers need different learning-rate settings. IVON uses a cosine decay,
`hess_init=1`, `clip_radius=0.1`, and the settings printed below. These were
chosen using training and a separate validation population before the final
test comparison. Equal gradient budgets do not imply equal wall time, and
the dropout regularizer does not establish the same prior as IVON's explicit
Gaussian prior.

The final test population chooses one of the four intervals uniformly, then
samples uniformly inside it. It contains 2,048 fresh observations shared by
both methods; its labels are used only after all fits finish. It does not
put observations in the central gap or outside the intervals. The plots
show extrapolation there, without a ground-truth uncertainty claim.
"""
    )
    code(
        r'''
IVON_TOY_SEEDS = (11, 12, 13)
IVON_TOY_DRAWS = 200
IVON_TOY_LR = 0.1
IVON_TOY_BETA2 = 0.999
IVON_TOY_DECAY = 1e-4
ivon_toy_results = []
ivon_toy_example = None


def fit_ivon_toy(seed, steps=TOY_TRAIN_STEPS):
    torch.manual_seed(seed)
    model = ActiveToyNet(dropout=0.0)
    optimizer = ivon.IVON(
        model.parameters(), lr=IVON_TOY_LR, ess=len(x_train),
        hess_init=1.0, beta2=IVON_TOY_BETA2,
        weight_decay=IVON_TOY_DECAY, clip_radius=0.1,
    )
    history = []
    model.train()
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = IVON_TOY_LR * (
            0.05 + 0.95 * (1 + math.cos(math.pi * step / steps)) / 2
        )
        with ivon_weight_sample(model, optimizer, train=True):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x_train), y_train)
            loss.backward()
        optimizer.step()
        if step % 400 == 0 or step == steps - 1:
            history.append((step, loss.item()))
    model.eval()
    return model, optimizer, np.asarray(history)


if ivon_available:
    # Verify paired initialization before spending any training budget.
    torch.manual_seed(11)
    initial_dropout = ActiveToyNet(0.2)
    torch.manual_seed(11)
    initial_ivon = ActiveToyNet(0.0)
    for left, right in zip(initial_dropout.parameters(), initial_ivon.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    ivon_fits = []
    for seed in IVON_TOY_SEEDS:
        started = time.perf_counter()
        dropout_model, dropout_history = fit_toy(
            x_train, y_train, seed=seed, steps=TOY_TRAIN_STEPS, active=True
        )
        dropout_seconds = time.perf_counter() - started
        started = time.perf_counter()
        weight_model, weight_optimizer, weight_history = fit_ivon_toy(seed)
        ivon_seconds = time.perf_counter() - started
        ivon_fits.append((seed, dropout_model, weight_model, weight_optimizer,
                          dropout_history, weight_history, dropout_seconds, ivon_seconds))
        print(f"Supplied toy fit: seed {seed}; dropout {dropout_seconds:.1f}s; IVON {ivon_seconds:.1f}s")

    # A genuinely stochastic predictor must preserve both means and optimizer state.
    weight_model, weight_optimizer = ivon_fits[0][2:4]
    before = [p.detach().clone() for p in weight_model.parameters()]
    before_hess = [g["hess"].clone() for g in weight_optimizer.param_groups]
    before_momentum = [g["momentum"].clone() for g in weight_optimizer.param_groups]
    before_step = weight_optimizer.current_step
    repeated = torch.tensor([[-1.0], [-1.0], [1.7], [1.7]])
    checked = ivon_predict_probabilities(weight_model, weight_optimizer, repeated, 32, 1)
    assert checked.shape == (32, 4, 2) and not checked.requires_grad
    torch.testing.assert_close(checked[:, 0], checked[:, 1])
    torch.testing.assert_close(checked[:, 2], checked[:, 3])
    torch.testing.assert_close(checked.sum(-1), torch.ones(32, 4))
    assert torch.isfinite(checked).all() and checked.var(0).max() > 1e-8
    sd = ivon_posterior_sd(weight_optimizer)
    assert torch.isfinite(sd).all() and (sd > 0).all()
    try:
        with ivon_weight_sample(weight_model, weight_optimizer):
            raise RuntimeError("Check interrupted sampling")
    except RuntimeError:
        pass
    for old, current in zip(before, weight_model.parameters()):
        torch.testing.assert_close(old, current, rtol=0, atol=0)
    for i, group in enumerate(weight_optimizer.param_groups):
        torch.testing.assert_close(before_hess[i], group["hess"], rtol=0, atol=0)
        torch.testing.assert_close(before_momentum[i], group["momentum"], rtol=0, atol=0)
    assert before_step == weight_optimizer.current_step
    print("Supplied IVON checks passed: coherent draws, positive variance and state restoration.")

    # Test labels are inspected only after the entire fitting protocol above.
    ivon_test_x, ivon_test_y = sample_interval_population(2048, 91027)
    for seed, dropout_model, weight_model, weight_optimizer, dh, wh, ds, ws in ivon_fits:
        for name, model, optimizer, seconds in [
            ("MC dropout", dropout_model, None, ds),
            ("IVON", weight_model, weight_optimizer, ws),
        ]:
            torch.manual_seed(91000 + seed)
            draws = (
                demo_dropout_probabilities(model, ivon_test_x, IVON_TOY_DRAWS)
                if optimizer is None else
                ivon_predict_probabilities(model, optimizer, ivon_test_x, IVON_TOY_DRAWS)
            )
            ivon_toy_results.append({"method": name, "seed": seed, "seconds": seconds,
                                     **classification_metrics(draws.mean(0), ivon_test_y)})
        if seed == IVON_TOY_SEEDS[0]:
            torch.manual_seed(92011)
            dropout_grid = demo_dropout_probabilities(dropout_model, x_grid, IVON_TOY_DRAWS)
            torch.manual_seed(92011)
            ivon_grid = ivon_predict_probabilities(weight_model, weight_optimizer, x_grid, IVON_TOY_DRAWS)
            ivon_toy_example = {"dropout": dropout_grid, "ivon": ivon_grid,
                                "dropout_history": dh, "ivon_history": wh,
                                "sd": ivon_posterior_sd(weight_optimizer)}

    print(f"IVON: N={len(x_train)}, lr={IVON_TOY_LR}, beta2={IVON_TOY_BETA2}, "
          f"delta={IVON_TOY_DECAY}; implied prior SD={(len(x_train)*IVON_TOY_DECAY)**-0.5:.3f}")
    rows = []
    for name in ("MC dropout", "IVON"):
        records = [r for r in ivon_toy_results if r["method"] == name]
        row = [name]
        for key in ("accuracy", "nll", "brier", "seconds"):
            values = np.array([r[key] for r in records])
            row.append(f"{values.mean():.4f} ± {values.std(ddof=1):.4f}")
        rows.append(row)
    table(["Method", "Test accuracy", "Test NLL", "Test Brier (sum over classes)", "Fit seconds"], rows)
    print("Mean ± training-repeat SD over three seeds; same 2,048 test observations. Not a confidence interval.")
'''
    )
    code(
        r'''
if ivon_toy_example is not None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey="row")
    for column, (key, title, color) in enumerate([
        ("dropout", "MC dropout: random masks", BLUE),
        ("ivon", "IVON: Gaussian weight draws", ORANGE),
    ]):
        draws = ivon_toy_example[key]
        terms = demo_uncertainty(draws)
        low, high = torch.quantile(draws[:, :, 1], torch.tensor([0.05, 0.95]), dim=0)
        for draw in draws[:12, :, 1]:
            axes[0, column].plot(x_grid[:, 0], draw, color=color, alpha=0.15, lw=0.8)
        axes[0, column].fill_between(x_grid[:, 0], low, high, color=color, alpha=0.15,
                                     label="Pointwise 5–95% under q")
        axes[0, column].plot(x_grid[:, 0], terms["mean_probability"][:, 1], color=color,
                             lw=2, label="Predictive mean")
        axes[0, column].scatter(x_train[:, 0], y_train, c=y_train, cmap="coolwarm", s=10)
        axes[0, column].set(title=title, ylabel="P(class 1)", ylim=(-0.05, 1.05))
        for field, label, line_color in [
            ("predictive_entropy", "Predictive", BLUE),
            ("expected_entropy", "Expected conditional", GREEN),
            ("mutual_information", "MI", ORANGE),
        ]:
            axes[1, column].plot(x_grid[:, 0], terms[field], label=label, color=line_color)
        axes[1, column].set(xlabel="x", ylabel="Nats", ylim=(-0.02, 0.73))
        for ax in axes[:, column]:
            ax.axvspan(-2, 1, color="0.7", alpha=0.13)
            ax.legend(fontsize=8)
    fig.suptitle("Same observations and architecture; illustrative fit, seed 11")
    plt.tight_layout()
    show_figure(fig, "ivon_toy_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for key, label, color in [("dropout", "MC dropout", BLUE), ("ivon", "IVON", ORANGE)]:
        history = ivon_toy_example[key + "_history"]
        axes[0].plot(history[:, 0], history[:, 1], marker=".", label=label, color=color)
    axes[0].set(xlabel="Full-batch update", ylabel="One-draw mean training CE",
                title="Training likelihood term, not the full VI objective")
    axes[0].legend(fontsize=9)
    axes[1].hist(ivon_toy_example["sd"].numpy(), bins=35, color=ORANGE, alpha=0.8)
    axes[1].set(xlabel="Learned weight standard deviation", ylabel="Parameter count",
                title="IVON stores uncertainty for every weight and bias")
    plt.tight_layout()
    show_figure(fig, "ivon_toy_training")

    rows = []
    for value in [-4.0, -1.0, 1.7, 6.0]:
        index = int((x_grid[:, 0] - value).abs().argmin())
        for key, label in [("dropout", "MC dropout"), ("ivon", "IVON")]:
            terms = demo_uncertainty(ivon_toy_example[key])
            rows.append([float(x_grid[index, 0]), label,
                         float(terms["mean_probability"][index, 1]),
                         float(terms["mutual_information"][index]),
                         float(terms["probability_mc_se"][index, 1])])
    table(["x", "Method", "P(class 1)", "MI (nats)", "MC SE of probability mean"], rows)
'''
    )
    markdown(
        r"""
Reading the comparison. The ribbons describe the spread of component
probabilities under each fitted approximation, not a confidence interval for
the true class probability. Probability MC SE describes integration error;
it is a different quantity from either ribbon width or MI. Check the training
curves and test scores before attributing a difference solely to posterior
quality: these methods also optimize different objectives with different
optimizers. A larger MI is not automatically better. Neither approximation
is required to be uncertain inside the gap or outside the observed range.
The three-repeat table supports a comparison for this setting, not a universal
ranking of dropout and variational BNNs.
"""
    )
    add_ivon_mnist_section(markdown, code)
