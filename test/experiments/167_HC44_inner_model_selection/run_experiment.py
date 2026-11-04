import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[3] / "src" / "167_HC44_inner_model_selection" / "run_experiment.py"), run_name="__main__")
