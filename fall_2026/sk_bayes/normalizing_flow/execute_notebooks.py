"""Execute and validate the seminar notebooks with a real Jupyter kernel.

Requires nbformat, nbclient, and ipykernel in addition to seminar dependencies.
Student execution is checked in memory, leaving its distributed outputs empty.
Solution execution is saved only after every cell completes successfully.
"""
from pathlib import Path
import argparse
import base64
import io
import time

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def execute(edition, enable_appendices=False, skip_mnist=False, timeout=1800):
    path = HERE / f"normalizing_flow_{edition}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if enable_appendices:
                cell.source = cell.source.replace("RUN_APPENDICES = False", "RUN_APPENDICES = True")
            if skip_mnist:
                cell.source = cell.source.replace("RUN_MNIST = True", "RUN_MNIST = False")
    started = time.perf_counter()
    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbformat.validate(notebook)
    if edition == "solutions":
        # Runtime switches do not change the next reader's default configuration.
        for original, executed in zip(nbformat.read(path, as_version=4).cells, notebook.cells):
            executed.source = original.source
        nbformat.write(notebook, path)
        # Keep the printed guide's image example synchronized with the notebook.
        for cell in notebook.cells:
            if cell.cell_type != "code" or "mnist_generated = " not in cell.source:
                continue
            for output in reversed(cell.outputs):
                png = output.get("data", {}).get("image/png")
                if png:
                    if isinstance(png, list):
                        png = "".join(png)
                    (HERE / "assets").mkdir(exist_ok=True)
                    payload = base64.b64decode(png)
                    (HERE / "assets" / "mnist_samples.png").write_bytes(payload)
                    # The repository tracks PDF figures; PNG exports are ignored.
                    from PIL import Image
                    with Image.open(io.BytesIO(payload)) as picture:
                        picture.convert("RGB").save(HERE / "assets" / "mnist_samples.pdf",
                                                    "PDF", resolution=150)
                    break
    print(f"{edition}: all cells passed in {time.perf_counter()-started:.1f}s"
          + ("; outputs saved" if edition == "solutions" else "; distributed notebook kept clean"),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edition", choices=["all", "student", "solutions"], nargs="?", default="all")
    parser.add_argument("--appendices", action="store_true",
                        help="Also exercise the unfinished student notebook's appendix guards")
    parser.add_argument("--skip-mnist", action="store_true",
                        help="Run Task 4 checks but skip MNIST download, GPU training, and image plots")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Maximum seconds per notebook cell (default: 1800)")
    args = parser.parse_args()
    for edition in (["student", "solutions"] if args.edition == "all" else [args.edition]):
        execute(edition, enable_appendices=args.appendices,
                skip_mnist=args.skip_mnist, timeout=args.timeout)
