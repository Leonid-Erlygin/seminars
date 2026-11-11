Bayesian neural networks: predictions, uncertainty, and active learning
Skoltech · Bayesian machine learning

Start here
  bnn_student.ipynb        Seminar notes and four implementation tasks.
  bnn_solutions.ipynb      Worked answers, executed checks, tables and plots.
  bnn_teachers_guide.pdf   Theory, teaching plans, worked answers and hints.
  bnn_teachers_guide.tex   Editable LaTeX source.

The seminar follows the MCMC, Laplace and normalizing-flow lesson structure:
derive a quantity, implement a focused function, check it independently, then
run a supplied experiment and interpret the result. The 90-minute core has
three tasks: MC-dropout prediction, categorical entropy/MI decomposition,
and label-free random/entropy/BALD acquisition. Task 4 is a separate GPU
practical with MNIST, a dropout CNN and three paired acquisition comparisons.

A fully implemented IVON-versus-dropout example follows Task 4 in both
notebooks. It adds no exercise or required seminar time: use the executed
solution plots for a short demonstration. Students can run its cells even
when the four exercise functions are unfinished. The example compares an
explicit diagonal-Gaussian weight posterior with MC dropout on the original
interval data and in a separate supervised GPU MNIST experiment. The guide
includes its theory, measured results and suggested discussion.

The first example explicitly reuses spring_2025/seminar_ue_en.ipynb:
  Class 0: [-5,-3] and [1,2.5]
  Class 1: [-3,-2] and [2.5,5]
There are 30 equally spaced observations per interval, with the original
sigmoid hidden architecture, dropout placement and 4,000-step schedule.
The new notes explain the shared endpoints, distinguish deterministic grids
from a separate evaluation population, and use a shared dropout mask for
coherent function curves. The original notebook is not modified.

The repeated small-data acquisition experiment uses a separate supplied
two-hidden-layer tanh network, width 32, trained for 1,000 steps per budget.
This keeps optimization adequate with only 12 initial labels. All acquisition
strategies share its architecture, seeds and training budget; the original
sigmoid model remains the motivating example for Tasks 1 and 2.

The supplied IVON interval comparison uses this same tanh architecture for
both methods, identical initial weights, 4,000 full-batch updates and 200
predictive draws. Three paired fits are evaluated on the same 2,048 new
interval-population observations. Training/validation pilots found the
original deep sigmoid network stalled under IVON, so it is retained in the
original tasks and not mixed into this paired architecture comparison.
IVON settings are fixed before test scoring: lr 0.1 with cosine decay to
0.05 of its initial value, beta2 0.999, hess_init 1, clip_radius 0.1,
ess 120 and weight_decay 1e-4. Optimizers and approximate priors differ.

The supplied GPU comparison uses the same CNN, 1,000 uniformly selected
training labels, a disjoint validation set, 600 updates, identical initial
parameter means and minibatch streams, and 30 predictive draws. It reports
three paired repeats on the official test set, calibration curves and
uncertain-digit probability panels. It is separate from active learning.
The recorded run shows dropout ahead of IVON at this small compute budget;
the purpose is to compare approximations, not assume a newer method wins.

The supplied Sk_BML_BNN_ActiveLearning_Intro_201125 (2).ipynb motivates the
image active-learning comparison. Its random/BALD comparison is retained
with explicit budgets, unique oracle queries, independent evaluation and
paired repeats. The new implementation has no remote helper-code install.

Appendix A compares overlap with missing input support. Appendix B changes
only the Monte Carlo budget and discusses redundant batch queries. Neither
appendix is required for the core seminar. The guide also explains what
MC dropout does and does not establish about Bayesian uncertainty.

Running
  Python 3.9+, NumPy, Matplotlib, PyTorch, torchvision, IPython, ipykernel and
  ivon-opt 0.1.3 and a notebook interface such as Jupyter or VS Code. Install requirements.txt
  with matching PyTorch/torchvision builds. All small examples run on CPU.
  Task 4 uses CUDA; without it, CPU checks run and GPU training is skipped.

Run notebook cells in order. Unfinished exercises raise NotImplementedError;
their experiments are skipped. Incorrect completed solutions fail visibly.
RUN_MNIST=True enables the image practical. RUN_APPENDICES defaults to False
for students and True for solutions. SAVE_FIGURES defaults to True only for
solutions and exports figures into assets for the printed guide.
RUN_IVON=True enables the supplied comparison. Set it to False for a faster
core-only run. If ivon-opt is not installed, that example prints an install
instruction while the original tasks remain runnable. Without CUDA, its
one-dimensional comparison still runs and its MNIST part is skipped.

MNIST data reuse /tmp/sk_bayes_mnist/data by default. BNN experiment caches
live separately under /tmp/sk_bayes_bnn_mnist. Set BNN_MNIST_DATA and
BNN_MNIST_CACHE before running to use persistent locations. MNIST_RETRAIN=True
fits the actual experiment by default; False explicitly replays a matching
completed run. Data and model checkpoints are kept outside the repository.

Rebuilding
  python build_bnn_figures.py
  python build_bnn_seminar.py
  python execute_notebooks.py
The builder synchronizes student/solution cells and clears saved outputs.
Execution also requires nbformat and nbclient. Student execution is checked
in memory; successful solution outputs and guide figures are saved.
  python execute_notebooks.py student --appendices
  python execute_notebooks.py --skip-mnist
  python execute_notebooks.py --skip-ivon
  python execute_notebooks.py solutions --reuse-mnist

Compile the guide from this directory with either LaTeX or Tectonic:
  pdflatex -interaction=nonstopmode -halt-on-error bnn_teachers_guide.tex
  tectonic bnn_teachers_guide.tex

Sources
  build_bnn_seminar.py     Core theory, exercises, solutions and experiments.
  bnn_mnist_cells.py       GPU image practical, embedded in both notebooks.
  bnn_ivon_cells.py        Supplied IVON theory and interval comparison.
  bnn_ivon_mnist_cells.py  Supplied supervised GPU IVON/dropout comparison.
  build_bnn_figures.py     Three explanatory figures in PNG/vector PDF.
Explanatory figures are embedded in the notebooks as attachments, so they
remain visible when a notebook is copied independently of the assets folder.
Runtime code is also embedded; the builder modules are not notebook imports.
References and source attribution appear in the notebooks and guide.
