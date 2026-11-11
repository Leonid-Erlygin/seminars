"""Run the filled seminar notebook in a fresh kernel and save its outputs."""
from argparse import ArgumentParser
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient


def execute(path, timeout=300):
    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    # Keep numerical work small and reproducible on a classroom CPU.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    start = time.monotonic()
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
        record_timing=False,
    )
    client.execute()
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells):
        raise RuntimeError("Some code cells were not executed.")
    figures = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.outputs
    )
    temporary = path.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    print(
        f"Executed {path.name}: {len(code_cells)} code cells, "
        f"{figures} figures, {time.monotonic() - start:.1f} seconds."
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook", nargs="?", type=Path,
        default=Path(__file__).with_name("sampling_first_solutions.ipynb"),
    )
    parser.add_argument("--timeout", type=int, default=300,
                        help="Maximum execution time per code cell, in seconds.")
    args = parser.parse_args()
    execute(args.notebook, args.timeout)
