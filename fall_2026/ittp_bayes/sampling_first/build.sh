#!/usr/bin/env bash
# Rebuild the notebook, execute all examples, and compile the teacher's guide.
set -euo pipefail
cd "$(dirname "$0")"

python build_sampling_notebook.py
python execute_notebook.py

SAMPLING_BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/sampling-first-latex.XXXXXX")
trap 'rm -rf "$SAMPLING_BUILD_DIR"' EXIT
if command -v pdflatex >/dev/null 2>&1; then
    for SAMPLING_PASS in 1 2 3; do
        if ! pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
            -output-directory="$SAMPLING_BUILD_DIR" sampling_first_teachers_guide.tex \
            > "$SAMPLING_BUILD_DIR/pass-${SAMPLING_PASS}.stdout" 2>&1; then
            cat "$SAMPLING_BUILD_DIR/pass-${SAMPLING_PASS}.stdout"
            exit 1
        fi
    done
elif command -v tectonic >/dev/null 2>&1; then
    tectonic --outdir "$SAMPLING_BUILD_DIR" sampling_first_teachers_guide.tex
else
    echo 'Install a LaTeX distribution with Russian support or Tectonic, then rerun build.sh.' >&2
    exit 1
fi
cp "$SAMPLING_BUILD_DIR/sampling_first_teachers_guide.pdf" sampling_first_teachers_guide.pdf
echo 'Built sampling_first_solutions.ipynb and sampling_first_teachers_guide.pdf.'
