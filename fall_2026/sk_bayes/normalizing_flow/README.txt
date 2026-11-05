Normalizing flows: from change of variables to learned densities
Skoltech · Bayesian machine learning

Start here
  normalizing_flow_student.ipynb        Seminar notes and three coding tasks.
  normalizing_flow_solutions.ipynb      Worked answers, executed checks and plots.
  normalizing_flow_teachers_guide.pdf   Teaching plan, theory, solutions and hints.
  normalizing_flow_teachers_guide.tex   Editable LaTeX source of the guide.

The main lesson follows the MCMC and logistic/Laplace seminar structure:
motivation and derivation, a focused implementation, checks, an experiment,
and interpretation. Three tasks implement a scalar affine flow, an affine
coupling layer, and RealNVP composition/density/sampling. Supplied training
fits a six-layer model to locally generated noisy moons and compares it with
a Gaussian fitted to the same training observations. Separate validation and
test sets distinguish checkpoint selection from final evaluation.

The main route is designed for 90 minutes; a 60-minute route is in the guide.
Appendix A develops reverse-KL variational inference on a curved target and
compares the learned flow with Laplace and an analytic reference. Appendix B
examines repeated masks, support restrictions, and approximation versus
integration error. Neither appendix is required for the main seminar.

Running
  Python 3.9+, numpy, matplotlib, torch (2.0+), ipykernel, and a notebook
  interface such as Jupyter or VS Code. Install requirements.txt in your
  preferred environment. All data are generated locally; no dataset download
  or GPU is needed. Run notebook cells in order.

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
solution outputs are saved. To check optional student guards as well:
  python execute_notebooks.py student --appendices

Compile the guide with a standard LaTeX installation or Tectonic:
  pdflatex -interaction=nonstopmode -halt-on-error normalizing_flow_teachers_guide.tex
  tectonic normalizing_flow_teachers_guide.tex

build_normalizing_flow_figures.py maintains the explanatory figures in PNG
and vector PDF. The notebook builder maintains all lesson prose, task code,
checks, and worked interpretation answers. The guide is edited separately.

Seminar_RealNVP.ipynb is the supplied previous-year reference and is preserved.
Primary references and attribution appear at the end of the new materials.
