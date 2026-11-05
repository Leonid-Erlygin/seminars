"""Notebook cells for the GPU MNIST extension; imported by the seminar builder.

The builder supplies its four cell helpers. This module never trains a model or
downloads data at import time. Its generated notebook cells own all MNIST work.
"""


def add_mnist_section(markdown, code, exercise, answer):
    markdown(r'''
## Task 4 — Fit MNIST on a GPU

The two-dimensional example exposes the geometry. Now fit the same kind of
distribution to all 784 pixels of a handwritten digit. The model and training
loop are supplied; your task is to connect an integer image to the continuous
density with the correct change of variables and units.

This extension uses CUDA and `torchvision`. With `RUN_MNIST=True`, completing
Task 4 enables an actual MNIST training run. The default is 12 epochs on 50,000
training images, with 10,000 separate validation images. The official 10,000
test images are evaluated after checkpoint selection. Expect several minutes
on a recent GPU; slower GPUs can take longer. Set the budget before examining
test results. With no CUDA device, the CPU checks still run and training is
explicitly skipped. The earlier tasks remain usable on a CPU.

The first enabled run downloads MNIST through
[torchvision.datasets.MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
Later runs reuse the local files. Data and model checkpoints live outside the
seminar folder, under `/tmp/sk_bayes_mnist` by default. Set `NF_MNIST_CACHE`
before running the notebook to choose a persistent cache. The default
`MNIST_RETRAIN=True` performs the fitting task on each enabled run. Set it to
`False` to reuse a matching saved checkpoint for a cached demonstration.

### 6.1 From integer pixels to a continuous density

For an integer image $I\in\{0,\ldots,255\}^{784}$, draw independent
$\varepsilon_j\sim\operatorname{Uniform}(0,1)$ and define
$$u_j=\frac{I_j+\varepsilon_j}{256},\qquad
y_j=\operatorname{logit}(u_j)=\log u_j-\log(1-u_j).$$
Dequantization places a continuous point inside the pixel's bin; dividing by
255 would give the wrong bin width. The uncontracted logit is a bijection
from $(0,1)^{784}$ to $\mathbb R^{784}$, with
$$\log\left|\det\frac{\partial y}{\partial u}\right|
=\sum_{j=1}^{784}[-\log u_j-\log(1-u_j)].$$
Our high-dimensional flow generates logits, $y=g_\phi(z)$. Its supplied
`log_prob(y)` returns $\log q_y(y)$, including the flow's own determinants.
Thus
$$\log q_u(u)=\log q_y(y)
 +\sum_j[-\log u_j-\log(1-u_j)].$$

The corresponding discrete model assigns each integer image the probability
mass of its bin. Jensen's inequality gives
$$\log P_\phi(I)=\log\!\int_{[0,1)^{784}}
\frac{q_u((I+\varepsilon)/256)}{256^{784}}\,d\varepsilon
\;\ge\;\mathbb E_\varepsilon[\log q_u(u)]-784\log256.$$
The expected dequantized negative log score is an upper bound on the exact
discrete NLL. A single noise draw is a Monte Carlo estimator of that bound,
and need not itself bound the exact NLL. Report bits per dimension as
$$\operatorname{bpd}(I,\varepsilon)
=\frac{-\log q_y(y)-\log|\det(\partial y/\partial u)|+784\log256}
       {784\log2}.$$
This explains the extra 8 bits per pixel. A continuous log density at a point
is not the exact probability of an integer image.

Noise and logit preprocessing use float64 to avoid rounding a bright pixel
to $u=1$ before taking its logarithm. Exact zero draws are redrawn. Logits
are then cast to the model's dtype. We do not clip image values or contract
their support; the generative image map is simply $u=\operatorname{sigmoid}(y)$.
''')
    code(r'''
from pathlib import Path
import os

MNIST_CACHE = Path(os.environ.get("NF_MNIST_CACHE", "/tmp/sk_bayes_mnist"))
MNIST_EPOCHS = 12
MNIST_BATCH_SIZE = 256
MNIST_LAYERS = 10
MNIST_HIDDEN = 512
MNIST_RETRAIN = True
MNIST_DIM = 28 * 28
mnist_enabled = False

def mnist_uniform(shape, device, generator):
    """Numerically open Uniform(0,1), kept in float64 for image preprocessing."""
    noise = torch.rand(shape, dtype=torch.float64, device=device, generator=generator)
    zero = noise == 0
    while zero.any():
        noise[zero] = torch.rand((int(zero.sum()),), dtype=torch.float64,
                                device=device, generator=generator)
        zero = noise == 0
    return noise

def mnist_preprocess(images, noise):
    """Supplied: integer pixels -> float64 (logits, log|dy/du|)."""
    u = (images.to(torch.float64) + noise.to(torch.float64)) / 256.0
    y = torch.log(u) - torch.log1p(-u)
    logdet = (-torch.log(u) - torch.log1p(-u)).flatten(1).sum(dim=1)
    return y.flatten(1), logdet
''')
    markdown(r'''
Complete `mnist_nll_terms(model, images, noise)`.

1. Use the supplied preprocessing, then cast the logits to the model's dtype
   and device. `images` is a uint8 tensor `(B,1,28,28)`; `noise` has the same
   shape and contains values strictly between zero and one.
2. Evaluate `model.log_prob(y)`. Add the preprocessing log determinant to
   obtain the density in $u$ coordinates. Include the $784\log256$ bin-width
   correction when forming `nll_nats`.
3. Return `(nll_nats, bits_per_dim)`, each `(B,)`. Here `nll_nats` is the
   dequantized discrete-NLL estimator in nats, not the continuous NLL alone.
   Preserve gradients through the model; the training objective will be the
   mean of `bits_per_dim`.

The CPU checks include a model whose implied pixel density is uniform. Its
answer is exactly 8 bpd for every image, including zero and 255-valued pixels.
This checks both the logit Jacobian and the bin-width correction.
''')
    exercise('4', r'''
def mnist_nll_terms(model, images, noise):
    """Return dequantized (nll_nats:(B,), bits_per_dim:(B,))."""
    # TODO: logit preprocessing, model density, Jacobian, and discrete units.
    raise NotImplementedError("Task 4")
''', r'''
def mnist_nll_terms(model, images, noise):
    """Return dequantized (nll_nats:(B,), bits_per_dim:(B,))."""
    parameter = next(model.parameters())
    y, preprocessing_logdet = mnist_preprocess(images, noise)
    y = y.to(device=parameter.device, dtype=parameter.dtype)
    preprocessing_logdet = preprocessing_logdet.to(device=parameter.device,
                                                   dtype=parameter.dtype)
    log_density_u = model.log_prob(y) + preprocessing_logdet
    nll_nats = -log_density_u + MNIST_DIM * math.log(256)
    return nll_nats, nll_nats / (MNIST_DIM * math.log(2))
''')
    code(r'''
def test_mnist_objective():
    class UniformPixelOracle(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        def log_prob(self, y):
            # Independent standard logistic logits imply uniform pixel values.
            return (-torch.nn.functional.softplus(y)
                    -torch.nn.functional.softplus(-y)).sum(dim=1) + self.offset

    oracle = UniformPixelOracle()
    images = torch.stack([torch.zeros(1, 28, 28),
                          torch.full((1, 28, 28), 255),
                          torch.arange(784).remainder(256).reshape(1, 28, 28)]).to(torch.uint8)
    noise = torch.linspace(0.001, 0.999, images.numel(), dtype=torch.float64).reshape(images.shape)
    nll, bpd = mnist_nll_terms(oracle, images, noise)
    assert nll.shape == bpd.shape == (3,)
    assert nll.dtype == bpd.dtype == torch.float64 and nll.device.type == "cpu"
    assert_close(nll, torch.full_like(nll, 784 * math.log(256)), atol=2e-9)
    assert_close(bpd, torch.full_like(bpd, 8.0), atol=1e-10)
    grad, = torch.autograd.grad(bpd.mean(), oracle.offset)
    assert_close(grad, grad.new_tensor(-1 / (784 * math.log(2))))

    class NormalLogitOracle(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        def log_prob(self, y):
            return -0.5 * (y.square() + math.log(2*math.pi)).sum(dim=1) + self.anchor

    normal = NormalLogitOracle()
    nll, bpd = mnist_nll_terms(normal, images, noise)
    u = (images.double() + noise) / 256
    y = torch.log(u / (1-u)).flatten(1)
    reference = (0.5 * (y.square() + math.log(2*math.pi)).sum(dim=1)
                 + (torch.log(u) + torch.log1p(-u)).flatten(1).sum(dim=1)
                 + 784 * math.log(256))
    assert_close(nll, reference, atol=1e-8)
    assert_close(bpd, reference / (784 * math.log(2)), atol=1e-10)
    assert torch.isfinite(nll).all()
    print("Uniform pixel density: 8 bpd; normal-logit oracle and model gradient: correct.")

check_task(4, test_mnist_objective)
''')
    markdown(r'''
### 6.2 A supplied 784-dimensional RealNVP

Each layer preserves one half of a flattened 28 by 28 checkerboard and
transforms the other half. The masks alternate. Its conditioner is an MLP
with two hidden layers of width 512 and SiLU activations; the final layer
starts at zero. The log scale is bounded by $1.5\tanh(s_{\rm raw})$.

A final generating affine map $y=\mu+\exp(a)\odot h$ learns a separate location
and scale for every pixel in logit space. Its initial moments use a subset
of training images only. Its log determinant $\sum_j a_j$ is part of the
model density. This gives the couplings a useful starting coordinate system
without fitting any preprocessing parameter on validation or test images.
This compact MLP model is an educational image flow; the original RealNVP
image architecture also uses convolutional networks and multiple scales.
''')
    code(r'''
class MNISTCoupling(nn.Module):
    def __init__(self, mask, hidden=512):
        super().__init__()
        self.register_buffer("mask", mask.float().reshape(1, -1))
        self.net = nn.Sequential(nn.Linear(MNIST_DIM, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2*MNIST_DIM))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def transform(self, value, inverse=False):
        preserved = self.mask * value
        raw_s, raw_t = self.net(preserved).chunk(2, dim=-1)
        s = 1.5 * torch.tanh(raw_s) * (1-self.mask)
        t = raw_t * (1-self.mask)
        if inverse:
            output = preserved + (1-self.mask) * (value-t) * torch.exp(-s)
            return output, -s.sum(dim=-1)
        output = preserved + (1-self.mask) * (value*torch.exp(s)+t)
        return output, s.sum(dim=-1)

class MNISTRealNVP(nn.Module):
    def __init__(self, n_layers=10, hidden=512):
        super().__init__()
        row, column = torch.meshgrid(torch.arange(28), torch.arange(28), indexing="ij")
        checkerboard = ((row + column) % 2).flatten()
        self.layers = nn.ModuleList([
            MNISTCoupling(checkerboard if k % 2 == 0 else 1-checkerboard, hidden)
            for k in range(n_layers)])
        self.location = nn.Parameter(torch.zeros(MNIST_DIM))
        self.log_scale = nn.Parameter(torch.zeros(MNIST_DIM))

    @torch.no_grad()
    def initialize(self, training_logits):
        self.location.copy_(training_logits.mean(dim=0))
        self.log_scale.copy_(training_logits.std(dim=0).clamp_min(0.1).log())

    def transform(self, value, inverse=False):
        if inverse:
            value = (value-self.location) * torch.exp(-self.log_scale)
            logdet = -self.log_scale.sum().expand(len(value))
            for layer in reversed(self.layers):
                value, delta = layer.transform(value, inverse=True)
                logdet = logdet + delta
            return value, logdet
        logdet = value.new_zeros(len(value))
        for layer in self.layers:
            value, delta = layer.transform(value)
            logdet = logdet + delta
        return self.location + torch.exp(self.log_scale)*value, logdet+self.log_scale.sum()

    def log_prob(self, logits):
        z, inverse_logdet = self.transform(logits, inverse=True)
        return standard_normal_log_prob(z) + inverse_logdet

    @torch.no_grad()
    def sample_images(self, n, generator=None, temperature=1.0):
        parameter = self.location
        z = torch.randn(n, MNIST_DIM, generator=generator,
                        device=parameter.device, dtype=parameter.dtype)
        logits, _ = self.transform(temperature*z)
        return torch.sigmoid(logits).reshape(n, 1, 28, 28)

def test_mnist_supplied_model():
    # Small CPU model: nonzero couplings plus a nontrivial final affine map.
    with torch.random.fork_rng():
        torch.manual_seed(400)
        model = MNISTRealNVP(n_layers=2, hidden=12).double()
        with torch.no_grad():
            model.location.normal_(0, 0.1)
            model.log_scale.uniform_(-0.2, 0.2)
            for layer in model.layers:
                layer.net[-1].weight.normal_(0, 0.01)
                layer.net[-1].bias.normal_(0, 0.02)
        z = torch.randn(2, MNIST_DIM, dtype=torch.float64)
        y, ld = model.transform(z)
        recovered, inverse_ld = model.transform(y, inverse=True)
        assert_close(recovered, z)
        assert_close(ld+inverse_ld, torch.zeros_like(ld))
        assert_close(model.log_prob(y), standard_normal_log_prob(z)-ld)
    print("Supplied image flow: round trip, affine determinant, and density identity passed.")

test_mnist_supplied_model()
''')
    markdown(r'''
### 6.3 Data, validation, and GPU training

Shuffle the official training set once with a fixed seed, then reserve 10,000
images for validation. Use fresh dequantization noise for each training batch
and a fixed independent noise draw for each validation image. Select the
lowest validation-bpd checkpoint. The test set and its fixed noise draw are
used after selection. Digit labels are used only in the visualizations.

The training curve averages fresh-noise minibatch scores collected while
parameters change within an epoch. Validation scores evaluate a fixed model
and fixed noise. Their small fluctuations need not match. A saved checkpoint
also records the configuration, history, and selected epoch. Changing the
configuration starts a new training run rather than silently reusing it.
''')
    code(r'''
@torch.no_grad()
def mnist_scores(model, images, noise, batch_size=512):
    model.eval()
    scores = []
    for start in range(0, len(images), batch_size):
        _, bpd = mnist_nll_terms(model, images[start:start+batch_size],
                               noise[start:start+batch_size])
        scores.append(bpd.detach().cpu().double())
    return torch.cat(scores)

def mnist_train_or_load(model, training, validation, validation_noise, config):
    checkpoint_path = MNIST_CACHE / "mnist_realnvp_v1.pt"
    if checkpoint_path.exists() and not MNIST_RETRAIN:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved.get("config") == config:
            model.load_state_dict(saved["state_dict"])
            model.eval()
            print(f"Loaded validation-selected checkpoint: {checkpoint_path}")
            return saved["history"], saved["best_epoch"], saved["elapsed_seconds"]
        print("Cached configuration differs; training the requested model.")

    parameter = model.location
    rng = torch.Generator(device=parameter.device).manual_seed(404)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                        T_max=MNIST_EPOCHS, eta_min=5e-5)
    initial = mnist_scores(model, validation, validation_noise).mean().item()
    best_value, best_epoch = initial, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [[0, float("nan"), initial]]
    torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, MNIST_EPOCHS+1):
        model.train()
        order = torch.randperm(len(training), generator=rng, device=parameter.device)
        total, count = 0.0, 0
        for start in range(0, len(training), MNIST_BATCH_SIZE):
            images = training[order[start:start+MNIST_BATCH_SIZE]]
            noise = mnist_uniform(images.shape, parameter.device, rng)
            _, bpd = mnist_nll_terms(model, images, noise)
            loss = bpd.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MNIST loss; inspect logit inputs and scales")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()*len(images)
            count += len(images)
        scheduler.step()
        value = mnist_scores(model, validation, validation_noise).mean().item()
        history.append([epoch, total/count, value])
        if value < best_value:
            best_value, best_epoch = value, epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        print(f"Epoch {epoch:2d}/{MNIST_EPOCHS}: train {total/count:.4f}, validation {value:.4f} bpd")
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    model.load_state_dict(best_state)
    model.eval()
    MNIST_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save({"config": config, "state_dict": best_state, "history": history,
                "best_epoch": best_epoch, "elapsed_seconds": elapsed}, temporary)
    temporary.replace(checkpoint_path)
    print(f"Selected epoch {best_epoch}; GPU training took {elapsed/60:.2f} minutes.")
    return history, best_epoch, elapsed

if not RUN_MNIST:
    print("MNIST training disabled: set RUN_MNIST=True to enable the GPU extension.")
elif not ready(4):
    print("MNIST training skipped until Task 4 passes its CPU checks.")
elif not torch.cuda.is_available():
    print("MNIST GPU extension skipped: no CUDA device. Task 4 CPU checks remain available.")
else:
    from torchvision.datasets import MNIST
    mnist_device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(401)
    train_source = MNIST(str(MNIST_CACHE / "data"), train=True, download=True)
    test_source = MNIST(str(MNIST_CACHE / "data"), train=False, download=True)
    split = torch.randperm(len(train_source), generator=torch.Generator().manual_seed(402))
    mnist_train = train_source.data[split[:50000], None].to(mnist_device)
    mnist_validation = train_source.data[split[50000:], None].to(mnist_device)
    mnist_test = test_source.data[:, None].to(mnist_device)
    mnist_test_labels = test_source.targets
    validation_rng = torch.Generator(device=mnist_device).manual_seed(403)
    mnist_validation_noise = mnist_uniform(mnist_validation.shape, mnist_device, validation_rng)
    mnist_model = MNISTRealNVP(MNIST_LAYERS, MNIST_HIDDEN).to(mnist_device)
    initialization_rng = torch.Generator(device=mnist_device).manual_seed(405)
    initial_images = mnist_train[:8192]
    initial_noise = mnist_uniform(initial_images.shape, mnist_device, initialization_rng)
    initial_logits, _ = mnist_preprocess(initial_images, initial_noise)
    mnist_model.initialize(initial_logits.float())
    del initial_logits, initial_noise
    mnist_config = {"version": 1, "epochs": MNIST_EPOCHS, "layers": MNIST_LAYERS,
                    "hidden": MNIST_HIDDEN, "batch_size": MNIST_BATCH_SIZE,
                    "split_seed": 402, "validation_noise_seed": 403,
                    "training_seed": 404, "initialization_seed": 405,
                    "learning_rate": 5e-4, "dequantization": "open_uniform_exact_logit"}
    print(f"GPU: {torch.cuda.get_device_name(0)}; parameters: "
          f"{sum(p.numel() for p in mnist_model.parameters()):,}")
    mnist_history, mnist_best_epoch, mnist_elapsed = mnist_train_or_load(
        mnist_model, mnist_train, mnist_validation, mnist_validation_noise, mnist_config)
    # Test evaluation begins only after model selection and restoration.
    test_rng = torch.Generator(device=mnist_device).manual_seed(406)
    mnist_test_noise = mnist_uniform(mnist_test.shape, mnist_device, test_rng)
    mnist_test_bpd = mnist_scores(mnist_model, mnist_test, mnist_test_noise)
    table(["MNIST quantity", "Value"],
          [["Training images", len(mnist_train)], ["Validation images", len(mnist_validation)],
           ["Test images", len(mnist_test)], ["Selected epoch", mnist_best_epoch],
           ["Test dequantized bpd", mnist_test_bpd.mean().item()],
           ["Test SE across images / fixed noise draws", mnist_test_bpd.std().item()/math.sqrt(len(mnist_test_bpd))]])
    mnist_enabled = True
''')
    markdown(r'''
### 6.4 Inspect what was learned

Read the held-out dequantized score together with the samples. The grids below
show consecutive held-out images and unfiltered independent model draws.
There is no class conditioning or rejection of unattractive samples.
The compact model may produce incomplete or ambiguous digits; likelihood and
visual quality diagnose different aspects of fit.
''')
    code(r'''
def mnist_mosaic(images, columns=8, gap=2):
    """Display-only tiling; every panel uses the same grayscale limits."""
    images = images.detach().cpu().reshape(-1, 28, 28).numpy()
    rows = math.ceil(len(images)/columns)
    canvas = np.zeros((rows*28+(rows-1)*gap, columns*28+(columns-1)*gap))
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[row*(28+gap):row*(28+gap)+28, column*(28+gap):column*(28+gap)+28] = image
    return canvas

def mnist_show(ax, images, title, columns=8):
    ax.imshow(mnist_mosaic(images, columns), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.axis("off")

if mnist_enabled:
    history = np.asarray(mnist_history)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history[1:, 0], history[1:, 1], color="#2369a1", label="Training: online / fresh noise")
    ax.plot(history[:, 0], history[:, 2], color="#c77516", label="Validation: fixed noise")
    ax.axvline(mnist_best_epoch, color="0.5", ls="--", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="Dequantized bits per dimension", title="Full 28 × 28 MNIST density fitting")
    ax.legend(fontsize=9); plt.tight_layout(); plt.show()

    draw_rng = torch.Generator(device=mnist_device).manual_seed(407)
    mnist_generated = mnist_model.sample_images(32, generator=draw_rng)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    mnist_show(axes[0], mnist_test[:32].float()/255, "Held-out MNIST observations")
    mnist_show(axes[1], mnist_generated, "Independent samples from the fitted flow")
    plt.tight_layout(); plt.show()
''')
    markdown(r'''
Temperature changes the distribution: replacing $z\sim\mathcal N(0,I)$ by
$Tz$ uses the base $\mathcal N(0,T^2I)$. Only $T=1$ samples the model whose
likelihood was evaluated above. Using the same base draws across rows makes
the visual effect easier to compare. A lower temperature can make some
images look cleaner while reducing diversity; it does not improve the
reported likelihood of the original model.
''')
    code(r'''
if mnist_enabled:
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.5))
    for ax, temperature in zip(axes, [0.65, 0.85, 1.0]):
        rng = torch.Generator(device=mnist_device).manual_seed(408)
        draws = mnist_model.sample_images(12, generator=rng, temperature=temperature)
        label = "fitted model" if temperature == 1 else "changed base distribution"
        mnist_show(ax, draws, f"T = {temperature:g}: {label}", columns=12)
    plt.tight_layout(); plt.show()
''')
    markdown(r'''
Latent interpolation is another view of the learned map. Encode two held-out
images, interpolate linearly between their latent vectors, and decode the path.
The endpoints are supplied observations, and the intermediate images are a
deterministic path. These are not independent generated samples. Linear
interpolation can also pass through atypical latent regions, so a smooth
visual transition is not a proof of a good density model.
''')
    code(r'''
if mnist_enabled:
    with torch.no_grad():
        interpolation_logits, _ = mnist_preprocess(mnist_test[:100], mnist_test_noise[:100])
        interpolation_z, _ = mnist_model.transform(interpolation_logits.float(), inverse=True)
        t = torch.linspace(0, 1, 9, device=mnist_device)[:, None]
        fig, axes = plt.subplots(3, 1, figsize=(10, 4.5))
        for ax, (first, second) in zip(axes, [(0, 1), (2, 3), (4, 5)]):
            path = (1-t)*interpolation_z[first] + t*interpolation_z[second]
            logits, _ = mnist_model.transform(path)
            digits = torch.sigmoid(logits).reshape(-1, 1, 28, 28)
            a, b = int(mnist_test_labels[first]), int(mnist_test_labels[second])
            mnist_show(ax, digits, f"Encoded held-out {a} → {b}: deterministic latent interpolation", columns=9)
        plt.tight_layout(); plt.show()
''')
    markdown(r'''
Finally, check numerical inversion separately from generation. Passing an
observed, dequantized image through $f_\phi$ and back through $g_\phi$ should
recover that same continuous image. This follows from the architecture, even
for a poorly trained model. Training uses float32 with accelerated matrix
multiplication; this small diagnostic uses a float64 copy of the fitted
weights to check inversion without reduced-precision matrix products.
The error map uses its own color scale to make tiny floating-point errors
visible; it is not a sample-quality metric.
''')
    code(r'''
if mnist_enabled:
    with torch.no_grad():
        observed = mnist_test[:8]
        noise = mnist_test_noise[:8]
        logits64, _ = mnist_preprocess(observed, noise)
        diagnostic_model = copy.deepcopy(mnist_model).double()
        z, inverse_ld = diagnostic_model.transform(logits64, inverse=True)
        recovered_logits, forward_ld = diagnostic_model.transform(z)
        original_u = (observed.double()+noise)/256
        recovered_u = torch.sigmoid(recovered_logits).reshape_as(original_u)
        pixel_error = (recovered_u-original_u).abs()
        max_error = pixel_error.max().item()
        assert max_error < 2e-10
        assert_close(recovered_logits, logits64, atol=1e-9, rtol=1e-9)
        assert_close(inverse_ld+forward_ld, torch.zeros_like(inverse_ld), atol=1e-8, rtol=0)
    table(["Numerical inversion diagnostic (float64)", "Maximum absolute error"],
          [["Pixel value in [0,1]", max_error],
           ["Logit coordinate", (recovered_logits-logits64).abs().max().item()],
           ["Forward + inverse log determinant", (inverse_ld+forward_ld).abs().max().item()]])
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.8))
    mnist_show(axes[0], original_u, "Input: held-out images with fixed dequantization noise", columns=8)
    mnist_show(axes[1], recovered_u, "Reconstructed with float64 inverse and generating maps", columns=8)
    errors = mnist_mosaic(pixel_error, columns=8, gap=0)
    im = axes[2].imshow(errors, cmap="magma", vmin=0,
                       vmax=max(max_error, torch.finfo(torch.float64).eps), interpolation="nearest")
    axes[2].set_title("Absolute pixel error: magnified color scale", fontsize=11)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.018, pad=0.02, label="Absolute error")
    plt.tight_layout(); plt.show()
    del diagnostic_model
''')
    markdown(r'''
Interpretation questions. Why does the score need both the logit determinant
and the 8-bit bin correction? What changes when a sample is drawn at $T<1$?
Why can reconstruction be almost perfect while independent samples are poor?
Which errors does the reported test SE measure, and which does it omit?
''')
    answer(r'''
The logit determinant converts a density per unit logit into a density per
unit normalized pixel. The bin correction then accounts for a width of
$1/256$ in each integer-pixel coordinate. Omitting either changes the objective
and its units. Temperature modifies the base and therefore the generated
distribution; only $T=1$ matches the evaluated model. Reconstruction tests
numerical invertibility using an observation as input, while independent
generation starts from an unrelated base draw and tests the learned density.
The test SE measures variation across the held-out image/noise pairs at fixed
fitted parameters. It omits training randomness, dataset-split variation,
approximation error, and uncertainty about exact discrete probabilities.
''')
