import os, sys, time, nbformat
from nbclient import NotebookClient
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["DATA_ROOT"] = os.path.abspath("data")
src, dst = "forest_fire_vit_detection.ipynb", "run_outputs/forest_fire_vit_detection_executed.ipynb"
nb = nbformat.read(src, as_version=4)
t0 = time.time()

def done(cell=None, cell_index=None, execute_reply=None):
    print(f"--- cell {cell_index} done [total {(time.time()-t0)/60:.1f} min]", flush=True)
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            print(o["text"][-1500:], flush=True)
        elif o.get("output_type") == "error":
            print("ERROR:", o["ename"], o["evalue"], flush=True)
    nbformat.write(nb, dst)

NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}},
               on_cell_executed=done).execute()
nbformat.write(nb, dst)
print("FINISHED", flush=True)
