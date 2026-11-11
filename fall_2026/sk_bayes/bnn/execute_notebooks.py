"""Execute and validate BNN notebooks in real Jupyter kernels.

Student execution is checked in memory. Solution outputs are saved only after
all cells complete successfully. Requires nbformat, nbclient, and ipykernel.
"""

from pathlib import Path
import argparse
import time

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def execute(edition, appendices=False, skip_mnist=False, reuse_mnist=False, timeout=1800, skip_ivon=False):
    path = HERE / f"bnn_{edition}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    originals = [cell.source for cell in notebook.cells]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if appendices:
                cell.source = cell.source.replace(
                    "RUN_APPENDICES = False", "RUN_APPENDICES = True"
                )
            if skip_mnist:
                cell.source = cell.source.replace("RUN_MNIST = True", "RUN_MNIST = False")
            if skip_ivon:
                cell.source = cell.source.replace("RUN_IVON = True", "RUN_IVON = False")
            if reuse_mnist:
                cell.source = cell.source.replace(
                    "MNIST_RETRAIN = True", "MNIST_RETRAIN = False"
                )
    started = time.perf_counter()
    print(f"Executing {edition}...", flush=True)
    NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    ).execute()
    nbformat.validate(notebook)
    if edition == "solutions":
        for cell, source in zip(notebook.cells, originals):
            cell.source = source
        nbformat.write(notebook, path)
    print(
        f"{edition}: all cells passed in {time.perf_counter()-started:.1f}s; "
        + ("outputs saved" if edition == "solutions" else "distributed outputs kept clean"),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "edition", choices=["all", "student", "solutions"], nargs="?", default="all"
    )
    parser.add_argument(
        "--appendices", action="store_true", help="Also check optional student guards"
    )
    parser.add_argument(
        "--skip-mnist",
        action="store_true",
        help="Run CPU checks and comparisons without either GPU experiment",
    )
    parser.add_argument(
        "--skip-ivon", action="store_true", help="Skip the supplied IVON comparison"
    )
    parser.add_argument(
        "--reuse-mnist",
        action="store_true",
        help="Reuse a matching completed GPU experiment explicitly",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    for edition in ["student", "solutions"] if args.edition == "all" else [args.edition]:
        execute(edition, args.appendices, args.skip_mnist, args.reuse_mnist, args.timeout, args.skip_ivon)
