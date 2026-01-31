import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, List, Optional, Tuple


def run_experiment(
    type: str,
    method: str,
    epochs: int,
):
    """
    Run a continual learning experiment with specified parameters

    Parameters
    ----------
    type : str
        The type of task and model (e.g. mlp_task)
    method : str
        The continual learning method to use (e.g. ewc)
    epochs : int
        The number of epochs for training

    Returns
    -------
    None

    Usage Example
    -------------
    run_experiment(type="mlp_task", method="ewc", epochs=10)
    """
    command = [sys.executable, "../src/main.py", "--type", type, "--method_r_l", method, "--epochs", str(epochs)]
    subprocess.run(command)


if __name__ == "__main__":
    epochs = 10
    # Run all model and taks configurations under every method
    for type in ["mlp_task", "mlp_domain", "mlp_class", "prog_net_task", "den_class"]:
        for method in ["none"]:#, "ewc", "replay", "replay_ewc"]:
            print(f"Running experiment: type={type}, method={method}, epochs={epochs}")
            run_experiment(type=type, method=method, epochs=epochs)

