Normalizing flows: from change of variables to learned densities
Skoltech · Bayesian machine learning

Start here
  normalizing_flow_student.ipynb        Seminar notes and four coding tasks.
  normalizing_flow_solutions.ipynb      Worked answers, executed checks and plots.
  normalizing_flow_teachers_guide.pdf   Teaching plan, theory, solutions and hints.
  normalizing_flow_teachers_guide.tex   Editable LaTeX source of the guide.

The main lesson follows the MCMC and logistic/Laplace seminar structure:
motivation and derivation, a focused implementation, checks, an experiment,
and interpretation. Tasks 1-3 implement a scalar affine flow, an affine
coupling layer, and RealNVP composition/density/sampling. Supplied training
fits a six-layer model to locally generated noisy moons and compares it with
a Gaussian fitted to the same training observations. Task 4 trains a
784-dimensional RealNVP on full-resolution MNIST using a CUDA GPU. Students
implement the image likelihood and bits-per-dimension calculation; the model,
training loop, and visualizations are supplied. Separate training, validation,
and test sets distinguish fitting, checkpoint selection, and evaluation.

Tasks 1-3 form the 90-minute core; a 60-minute route is in the guide. Allow an
additional practical session for Task 4, including GPU training and discussion
of generated digits, reconstruction, temperatures, and latent interpolation.
Appendix A develops reverse-KL variational inference on a curved target and
compares the learned flow with Laplace and an analytic reference. Appendix B
examines repeated masks, support restrictions, and approximation versus
integration error. Neither appendix is required for the main seminar. The
teacher's guide also explains the purpose of flows and published applications
in image density estimation, scientific posterior inference, and molecular
sampling, with primary references and limitations.

Running
  Python 3.9+, numpy, matplotlib, torch (2.0+), torchvision (0.15+), ipykernel,
  and a notebook
  interface such as Jupyter or VS Code. Install requirements.txt in your
  preferred environment, using compatible PyTorch/torchvision CUDA builds.
  Tasks 1-3 and the appendices generate data locally and run on a CPU.
  Task 4 downloads public MNIST once, then trains on a CUDA GPU. Without CUDA,
  its numerical checks run on CPU and its training is explicitly skipped.
  Run notebook cells in order. RUN_MNIST=True enables the GPU experiment;
  set it to False to run just the small examples and Task 4 checks.

MNIST data and generated training artifacts are cached outside the repository
under /tmp/sk_bayes_mnist by default. Set NF_MNIST_CACHE to a persistent path
before running the notebook to keep them across temporary-directory cleanup.
MNIST_RETRAIN=True fits from scratch by default; set it to False explicitly
to reuse a matching validation-selected checkpoint for a demonstration.
The image model uses ten checkerboard couplings, width-512 conditioners,
12 epochs, and batches of 256. The official 60,000 training images are split
50,000/10,000 for fitting/validation; the official 10,000 test images are
reserved for final evaluation. Labels do not enter the training objective.
The reference GPU run uses PyTorch 2.4.1+cu121 and torchvision 0.19.1+cu121
on an NVIDIA L40. Runtime and last digits can differ on other hardware.
The notebooks embed all saved plots and contain their own executable code.

Unfinished tasks raise NotImplementedError and are skipped by their checks;
dependent experiments report the missing task. Incorrect completed solutions
fail visibly. RUN_APPENDICES defaults to False for students and True for
solutions. All three explanatory figures are embedded in both notebooks and
remain visible when a notebook is copied independently of the assets folder.

Numerical conventions
  g maps base z to data x; inverse=True applies g^{-1}.
  Each transform returns the signed log absolute determinant of the map
  actually applied, one scalar per observation. log_prob encodes x and adds
  the inverse log determinant. All log scores use natural logarithms and the
  same original observation coordinates. Learned parameters are point
  estimates; exact model density evaluation does not imply an exact target fit.

Rebuilding (run from this directory, or use full script paths)
  python build_normalizing_flow_figures.py
  python build_normalizing_flow_seminar.py
  python execute_notebooks.py

The builder synchronizes exercise and solution cells and clears outputs.
execute_notebooks.py requires the additional packages nbformat and nbclient.
It validates both notebooks and executes them in real Jupyter kernels; student
outputs are checked in memory and kept empty on disk, while successful
solution outputs are saved. The full solution execution includes GPU training.
It also exports the real/generated MNIST comparison to assets/mnist_samples.png
and assets/mnist_samples.pdf for the teacher's guide.
For a fast execution of the small examples and Task 4 checks:
  python execute_notebooks.py --skip-mnist
To check optional student guards as well:
  python execute_notebooks.py student --appendices

Compile the guide with a standard LaTeX installation or Tectonic:
  pdflatex -interaction=nonstopmode -halt-on-error normalizing_flow_teachers_guide.tex
  tectonic normalizing_flow_teachers_guide.tex

build_normalizing_flow_figures.py maintains the explanatory figures in PNG
and vector PDF. build_normalizing_flow_seminar.py maintains the core lesson;
normalizing_flow_mnist_cells.py maintains Task 4. The builder synchronizes
their prose, task code, checks, and written answers into both notebooks.
The guide is edited separately.

Seminar_RealNVP.ipynb is the supplied previous-year reference and is preserved.
Primary references and attribution appear at the end of the new materials.
