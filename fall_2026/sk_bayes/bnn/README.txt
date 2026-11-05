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
  a notebook interface such as Jupyter or VS Code. Install requirements.txt
  with matching PyTorch/torchvision builds. All small examples run on CPU.
  Task 4 uses CUDA; without it, CPU checks run and GPU training is skipped.

Run notebook cells in order. Unfinished exercises raise NotImplementedError;
their experiments are skipped. Incorrect completed solutions fail visibly.
RUN_MNIST=True enables the image practical. RUN_APPENDICES defaults to False
for students and True for solutions. SAVE_FIGURES defaults to True only for
solutions and exports figures into assets for the printed guide.

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
  python execute_notebooks.py solutions --reuse-mnist

Compile the guide from this directory with either LaTeX or Tectonic:
  pdflatex -interaction=nonstopmode -halt-on-error bnn_teachers_guide.tex
  tectonic bnn_teachers_guide.tex

Sources
  build_bnn_seminar.py     Core theory, exercises, solutions and experiments.
  bnn_mnist_cells.py       GPU image practical, embedded in both notebooks.
  build_bnn_figures.py     Three explanatory figures in PNG/vector PDF.
Explanatory figures are embedded in the notebooks as attachments, so they
remain visible when a notebook is copied independently of the assets folder.
Runtime code is also embedded; the builder modules are not notebook imports.
References and source attribution appear in the notebooks and guide.
