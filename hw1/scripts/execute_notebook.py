"""Execute all notebook cells in a fresh kernel and save successful outputs."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

root = Path(__file__).resolve().parents[1]
install(user=True, kernel_name='recsys-hw1', display_name='Python (recsys-hw1)')
notebook = nbformat.read(root/'experiments.ipynb', as_version=4)
client = NotebookClient(notebook, timeout=None, kernel_name='recsys-hw1', resources={'metadata': {'path':str(root)}})
client.execute()
nbformat.write(notebook, root/'experiments.ipynb')
print('Все ячейки выполнены:', root/'experiments.ipynb')
