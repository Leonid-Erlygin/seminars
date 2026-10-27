Sampling: from independent draws to Markov chains
Skoltech · Bayesian machine learning

Files
  sampling_student.ipynb        Lecture notes and exercises, with clean outputs.
  sampling_solutions.ipynb      Worked exercises, executed checks, tables, and plots.
  sampling_teachers_guide.pdf   Theory, teaching notes, and worked answers.
  sampling_teachers_guide.tex   Editable LaTeX source.
  assets/                      Explanatory figures in PNG and vector PDF formats.

The notes develop posterior integration, independent Monte Carlo error, inverse
CDF sampling, rejection sampling, self-normalized importance sampling, and
Metropolis-Hastings. The three main implementation tasks use a common coin
posterior. Appendix A examines multimodality and the Hastings correction.

The six explanatory illustrations are embedded in both notebooks, so they are
visible before code is executed and remain available when the notebooks are
copied independently of the assets directory. The guide uses vector PDF figures.

Requirements
Python 3.9+, numpy, scipy, matplotlib, ipykernel, and a notebook interface such
as Jupyter or VS Code. All examples are generated locally. Run cells in order.
Unfinished exercises are skipped by the checks. RUN_APPENDIX defaults to False
in the student notebook and True in the solutions notebook.

Rebuilding
  python build_sampling_figures.py
  python build_sampling_seminar.py
The second command overwrites both notebooks and clears their outputs. Execute
the solutions notebook again to regenerate its saved outputs. Exercise code and
written answers are maintained together in build_sampling_seminar.py.

Compile the guide from this directory with either a standard LaTeX installation
or Tectonic:
  pdflatex -interaction=nonstopmode -halt-on-error sampling_teachers_guide.tex
  tectonic sampling_teachers_guide.tex

Academic references appear at the end of the notes and guide.
