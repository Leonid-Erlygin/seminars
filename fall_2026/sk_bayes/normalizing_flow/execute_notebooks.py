"""Execute and validate the seminar notebooks with a real Jupyter kernel.

Requires nbformat, nbclient, and ipykernel in addition to seminar dependencies.
Student execution is checked in memory, leaving its distributed outputs empty.
Solution execution is saved only after every cell completes successfully.
"""
from pathlib import Path
import argparse
import time

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def execute(edition, enable_appendices=False):
    path = HERE / f"normalizing_flow_{edition}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if enable_appendices:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace("RUN_APPENDICES = False", "RUN_APPENDICES = True")
    started = time.perf_counter()
    client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbformat.validate(notebook)
    if edition == "solutions":
        nbformat.write(notebook, path)
    print(f"{edition}: all cells passed in {time.perf_counter()-started:.1f}s"
          + ("; outputs saved" if edition == "solutions" else "; distributed notebook kept clean"),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edition", choices=["all", "student", "solutions"], nargs="?", default="all")
    parser.add_argument("--appendices", action="store_true",
                        help="Also exercise the unfinished student notebook's appendix guards")
    args = parser.parse_args()
    for edition in (["student", "solutions"] if args.edition == "all" else [args.edition]):
        execute(edition, enable_appendices=args.appendices)
