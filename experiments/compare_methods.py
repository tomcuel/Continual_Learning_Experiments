from collections import defaultdict
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, List, Optional, Tuple
import yaml


experiment_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
os.makedirs(experiment_dir, exist_ok=True)
ewc_dir = os.path.join(experiment_dir, "ewc")
os.makedirs(ewc_dir, exist_ok=True)
none_dir = os.path.join(experiment_dir, "none")
os.makedirs(none_dir, exist_ok=True)
replay_dir = os.path.join(experiment_dir, "replay")
os.makedirs(replay_dir, exist_ok=True)
replay_ewc_dir = os.path.join(experiment_dir, "replay_ewc")
os.makedirs(replay_ewc_dir, exist_ok=True)


search_paths = [
    os.path.join(ewc_dir, "results"),
    os.path.join(none_dir, "results"),
    os.path.join(replay_dir, "results"),
    os.path.join(replay_ewc_dir, "results")
]

def load_results(
    task_type: str
):
    """
    Load results from YAML files for a specific task type

    Parameters
    ----------
    task_type : str
        The type of task (e.g. "task", "domain", "class")

    Returns
    -------
    List[Dict]
        List of results loaded from YAML files

    Usage Example
    -------------
    results = load_results(task_type="task")
    """
    results = []
    for path in search_paths:
        for file in os.listdir(path):
            if file.endswith(".yaml") and task_type in file:
                with open(os.path.join(path, file), 'r') as f:
                    data = yaml.safe_load(f)
                model = file.split("_")[1]
                method = os.path.basename(os.path.dirname(path)) # determine method from the parent directory name
                results.append((model, method, data.get("overall_accuracy"), data.get("forgetting")))
    return results

def analyze_and_print_results(results):
    """
    Analyze and print results comparing models, methods, and their metrics.
    """
    if not results:
        print("No results found")
        return

    # Organize results by model and method
    model_method_results = defaultdict(list)
    for model, method, acc, forgetting in results:
        model_method_results[(model, method)].append((acc, forgetting))

    print("\nResults Summary:")
    print(f"{'Model':<15} {'Method':<15} {'Avg Accuracy':<15} {'Avg Forgetting':<15}")
    print("-" * 60)
    for (model, method), vals in sorted(model_method_results.items()):
        accs = [v[0] for v in vals if v[0] is not None]
        forgets = [v[1] for v in vals if v[1] is not None]
        avg_acc = sum(accs) / len(accs) if accs else float('nan')
        avg_forget = sum(forgets) / len(forgets) if forgets else float('nan')
        print(f"{model:<15} {method:<15} {avg_acc:<15.4f} {avg_forget:<15.4f}")

    # Optionally, compare best models/methods
    best = max(results, key=lambda x: (x[2] if x[2] is not None else float('-inf')))
    print("\nBest result by accuracy:")
    print(f"Model: {best[0]}, Method: {best[1]}, Accuracy: {best[2]}, Forgetting: {best[3]}")

# Comparing the task-il continual learning methods and models
def compare_task_il():
    """
    Compare Task-IL continual learning methods and models

    Parameters
    ----------
    None

    Returns
    -------
    None

    Usage Example
    -------------
    compare_task_il()
    """
    results = load_results(task_type="task")
    analyze_and_print_results(results)
    

# Comparing the domain-il continual learning methods and models
def compare_domain_il():
    """
    Compare Domain-IL continual learning methods and models

    Parameters
    ----------
    None

    Returns
    -------
    None

    Usage Example
    -------------
    compare_domain_il()
    """
    results = load_results(task_type="domain")
    analyze_and_print_results(results)

# Comparing the class-il continual learning methods and models 
def compare_class_il():
    """
    Compare Class-IL continual learning methods and models

    Parameters
    ----------
    None

    Returns
    -------
    None

    Usage Example
    -------------
    compare_class_il()
    """
    results = load_results(task_type="class")
    analyze_and_print_results(results)


if __name__ == "__main__":
    print("Comparing Task-IL Methods")
    compare_task_il()
    print()
    print("Comparing Domain-IL Methods")
    compare_domain_il()
    print()
    print("Comparing Class-IL Methods")
    compare_class_il()

