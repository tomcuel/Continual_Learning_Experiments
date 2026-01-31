import argparse
import numpy as np
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from typing import Dict, List, Optional, Tuple
import yaml


from src.data.download_mnist import download_and_preprocess_mnist
from src.data.torch_utilities import df_to_tensor_dataset, make_dataloaders
from src.methods.replaybuffer import ReplayBuffer_ClassIL, ReplayBuffer_DomainIL, ReplayBuffer_TaskIL
from src.models.den import DEN
from src.models.mlp_tasks import DeepNN_TaskIL
from src.models.mlp import DeepNN
from src.models.prog_net import ProgressiveNet
from src.tasks.mnist_class_il import create_mnist_task_il
from src.tasks.mnist_domain_il import make_domain_il_tasks
from src.tasks.mnist_task_il import make_task_il_tasks
from src.training.evaluator import predict_loader_task_il, accuracy_score_task_il, predict_loader, accuracy_score, predict_loader_den, accuracy_score_den
from src.training.full_training import full_training_task_il, full_training, full_training_prog_net_task_il, full_training_den, calibrate_den
from src.training.metrics import average_accuracy, forgetting, backward_transfer


parser = argparse.ArgumentParser(description="Continual Learning Experiments")
parser.add_argument(
    "--type",
    type=str,
    choices=["mlp_task", "mlp_class", "mlp_domain", "prog_net_task", "den_class"],
    required=True,
    help="Type of continual learning scenario to run"
)
parser.add_argument(
    "--method_r_l",
    type=str,
    choices=["none", "ewc", "replay", "replay_ewc"],
    required=True,
    help="Continual learning method to use for replay and loss regularization"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="Number of epochs to train per task"
)
args = parser.parse_args()


# Determine task, method and model types
task_type = "task" if args.type.endswith("task") else "class" if args.type.endswith("class") else "domain"
method_type = args.method_r_l
model_type = "mlp" if args.type.startswith("mlp") else "prog_net" if args.type.startswith("prog_net") else "den"


# Paths for configuration and results with folder creation
experiment_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
os.makedirs(experiment_dir, exist_ok=True)
method_folder = os.path.abspath(os.path.join(experiment_dir, method_type))
os.makedirs(method_folder, exist_ok=True)
config_folder = os.path.abspath(os.path.join(method_folder, "configs"))
os.makedirs(config_folder, exist_ok=True)
config_path = f"{config_folder}/config_{model_type}_{task_type}.yaml"
results_folder = os.path.abspath(os.path.join(method_folder, "results"))
os.makedirs(results_folder, exist_ok=True)
results_path = f"{results_folder}/results_{model_type}_{task_type}.yaml"


# Generic training parameters
epochs_per_task = args.epochs
device = "cpu"
torch.set_num_threads(os.cpu_count())
RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)


# Load MNIST data
X_train, X_test, y_train, y_test, input_dim, output_dim = download_and_preprocess_mnist()


# Create tasks based on the specified continual learning scenario type
num_tasks = 5
if task_type == "task":
    tasks = make_task_il_tasks(X_train, y_train, num_tasks=num_tasks, is_print=True)
elif task_type == "domain":
    tasks = make_domain_il_tasks(X_train, y_train, num_tasks=num_tasks, is_print=True)
elif task_type == "class":
    tasks = create_mnist_task_il(X_train, y_train, num_tasks=num_tasks, is_print=True)
else:
    raise ValueError("Unknown task type")

# Load configuration for the model 
config = None
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
if config is None:
    config = {} # garantee we have a dict to get default values from


# Setup the model parameters based on the specified continual learning scenario type
if model_type == "mlp":
    config_dict = {
        "hidden_layers": config.get("hidden_layers", [216, 251, 197]),
        "activation_functions": config.get("activation_functions", ["leaky_relu", "tanh", "relu"]),
        "dropout_rates": config.get("dropout_rates", [0.4459, 0.1863, 0.1279]),
        "batch_size": config.get("batch_size", 16),
        "gamma": config.get("gamma", 0.4855325340941379), 
        "learning_rate": config.get("learning_rate", 0.0004980564413343545),
        "step_size": config.get("step_size", 14.161634448046335),
        "weight_decay": config.get("weight_decay", 8.560817132298919e-06)
    }
elif model_type == "prog_net":
    config_dict = {
        "hidden_dims": config.get("hidden_dims", [256, 256]),
        "batch_size": config.get("batch_size", 16),
        "gamma": config.get("gamma", 0.5), 
        "learning_rate": config.get("learning_rate", 0.001),
        "step_size": config.get("step_size", 20),
        "weight_decay": config.get("weight_decay", 1e-5)
    }
elif model_type == "den":
    config_dict = {
        "hidden_dims": config.get("hidden_dims", [256, 256]),
        "batch_size": config.get("batch_size", 16),
        "gamma": config.get("gamma", 0.5), 
        "learning_rate": config.get("learning_rate", 0.001),
        "step_size": config.get("step_size", 20),
        "weight_decay": config.get("weight_decay", 1e-4), 
        "lambda_sparse": config.get("lambda_sparse", 0.001),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "grad_threshold": config.get("grad_threshold", 0.001),
        "percentile": config.get("percentile", 20.0)
    }


# Create the model based on the specified continual learning scenario type
if model_type == "mlp" and task_type == "task":
    model = DeepNN_TaskIL( # default value taken from the DeepNeuralNetwork from scratch repo 
        input_dim=input_dim,
        hidden_layers=config_dict["hidden_layers"], 
        num_tasks=num_tasks,
        output_dim_per_task=2,  # binary per task
        activations=config_dict["activation_functions"],
        dropout_rates=config_dict["dropout_rates"],
        use_batchnorm=True
    )
elif model_type == "mlp" and task_type != "task":
    model = DeepNN(
        input_dim=input_dim,
        hidden_layers=config_dict["hidden_layers"],
        output_dim=output_dim,
        activations=config_dict["activation_functions"],
        dropout_rates=config_dict["dropout_rates"],
        use_batchnorm=True
    )
elif model_type == "prog_net" and task_type == "task":
    model = ProgressiveNet(
        input_dim=input_dim, 
        hidden_dims=config_dict["hidden_dims"], 
        output_dim=output_dim
    )
elif model_type == "den" and task_type == "class":
    model = DEN(
        input_dim=input_dim,
        hidden_dims=config_dict["hidden_dims"],
        output_dim=output_dim
    )
else:
    raise ValueError("Invalid model and task type combination")
model.to(device)


# Create the ewc regularization and replay buffer 
# won't be called if method is "none", so no problem setting them to real values
config_dict["lambda_ewc"] = config.get("lambda_ewc", 0.4)
config_dict["replay_weight"] = config.get("replay_weight", 1.0)

if method_type == "ewc" or method_type == "replay_ewc":
    is_ewc = True
else:
    is_ewc = False
if method_type == "replay" or method_type == "replay_ewc":
    if task_type == "task":
        config_dict["max_size_per_task"] = config.get("max_size_per_task", 1000)
        replay_buffer = ReplayBuffer_TaskIL(
            max_size_per_task=config_dict["max_size_per_task"]
        )
    elif task_type == "class":
        config_dict["max_size_per_class"] = config.get("max_size_per_class", 1000)
        replay_buffer = ReplayBuffer_ClassIL(
            max_size_per_class=config_dict["max_size_per_class"],
            num_classes=output_dim
        )
    elif task_type == "domain":
        config_dict["capacity"] = config.get("capacity", 1000)
        replay_buffer = ReplayBuffer_DomainIL(
            capacity=config_dict["capacity"]
        )
else:
    replay_buffer = None


# Train the model on the specified continual learning scenario
if model_type == "mlp" and task_type == "task":
    model, accuracy_matrix = full_training_task_il(
        model=model,
        device=device,
        tasks=tasks,
        replay_buffer=replay_buffer if method_type in ["replay", "replay_ewc"] else None,
        replay_weight=config_dict["replay_weight"],
        is_ewc=is_ewc,
        lambda_ewc=config_dict["lambda_ewc"],
        batch_size=config_dict["batch_size"],
        epochs=epochs_per_task,
        gamma=config_dict["gamma"],
        learning_rate=config_dict["learning_rate"],
        step_size=config_dict["step_size"],
        weight_decay=config_dict["weight_decay"]
    )
elif model_type == "mlp" and task_type != "task":
    if task_type == "class":
        replay_buffer_class = replay_buffer if method_type in ["replay", "replay_ewc"] else None
        replay_buffer_domain = None
    elif task_type == "domain":
        replay_buffer_class = None
        replay_buffer_domain = replay_buffer if method_type in ["replay", "replay_ewc"] else None
    else :
        replay_buffer_class = None
        replay_buffer_domain = None
    model, accuracy_matrix = full_training(
        model=model,
        device=device,
        tasks=tasks,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_domain=replay_buffer_domain,
        replay_weight=config_dict["replay_weight"],
        is_ewc=is_ewc,
        lambda_ewc=config_dict["lambda_ewc"],
        batch_size=config_dict["batch_size"],
        epochs=epochs_per_task,
        gamma=config_dict["gamma"],
        learning_rate=config_dict["learning_rate"],
        step_size=config_dict["step_size"],
        weight_decay=config_dict["weight_decay"]
    )
elif model_type == "prog_net" and task_type == "task":
    model, accuracy_matrix = full_training_prog_net_task_il(
        model=model,
        device=device,
        tasks=tasks,
        replay_buffer=replay_buffer if method_type in ["replay", "replay_ewc"] else None,
        replay_weight=config_dict["replay_weight"],
        is_ewc=is_ewc,
        lambda_ewc=config_dict["lambda_ewc"],
        batch_size=config_dict["batch_size"],
        epochs=epochs_per_task,
        gamma=config_dict["gamma"],
        learning_rate=config_dict["learning_rate"],
        step_size=config_dict["step_size"],
        weight_decay=config_dict["weight_decay"]
    )
elif model_type == "den" and task_type == "class":
    model, accuracy_matrix = full_training_den(
        model=model,
        device=device,
        tasks=tasks,
        ouptut_dim=output_dim,
        replay_buffer=replay_buffer if method_type in ["replay", "replay_ewc"] else None,
        replay_weight=config_dict["replay_weight"],
        is_ewc=is_ewc,
        lambda_ewc=config_dict["lambda_ewc"],
        batch_size=config_dict["batch_size"],
        epochs=epochs_per_task,
        gamma=config_dict["gamma"],
        learning_rate=config_dict["learning_rate"],
        step_size=config_dict["step_size"],
        weight_decay=config_dict["weight_decay"],
        lambda_sparse=config_dict["lambda_sparse"],
        max_grad_norm=config_dict["max_grad_norm"],
        grad_threshold=config_dict["grad_threshold"],
        percentile=config_dict["percentile"]
    )
    model = calibrate_den(
        den_model=model,
        device=device,
        X_train=X_train,
        y_train=y_train,
        batch_size=config_dict["batch_size"],
        epochs=int(epochs_per_task / 2),
        learning_rate=config_dict["learning_rate"], 
        weight_decay=config_dict["weight_decay"]
    )


# Matrix based on the accuracy after each task of the training 
A = accuracy_matrix.get()
results_dict = {
    "train_average_accuracy": round(float(average_accuracy(A)), 4),
    "forgetting": round(float(forgetting(A)), 4),
    "backward_transfer": round(float(backward_transfer(A)), 4)
}
print(f"Train Average Accuracy: {results_dict['train_average_accuracy']:.4f}")
print(f"Train Forgetting: {results_dict['forgetting']:.4f}")
print(f"Train Backward Transfer: {results_dict['backward_transfer']:.4f}")


# Predictions and evaluation on test set 
if model_type == "mlp" and task_type != "task":
    test_dataset = df_to_tensor_dataset(X_test, y_test)
    test_loader = make_dataloaders(test_dataset, batch_size=config_dict["batch_size"], shuffle=False)
    y_pred, y_probs, y_true = predict_loader(model, test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    results_dict["overall_accuracy"] = round(float(test_acc), 4)
    print(f"Overall Test Accuracy: {test_acc:.4f}")
elif (model_type == "mlp" and task_type == "task") or (model_type == "prog_net" and task_type == "task"):
    overall_accuracies = []
    for t, task_dataset in enumerate(tasks):
        test_loader = make_dataloaders(task_dataset, batch_size=config_dict["batch_size"], shuffle=False)
        y_pred, _, y_true = predict_loader_task_il(model, test_loader, task_id=t, device=device)
        acc = accuracy_score_task_il(y_true, y_pred)
        overall_accuracies.append(acc)
    results_dict["overall_accuracy"] = round(float(np.mean(overall_accuracies)), 4)
    print(f"Overall Test Accuracy (mean over tasks): {np.mean(overall_accuracies):.4f}")
elif model_type == "den" and task_type == "class":
    test_dataset = df_to_tensor_dataset(X_test, y_test)
    test_loader = make_dataloaders(test_dataset, batch_size=config_dict["batch_size"], shuffle=False)
    seen_classes = list(range(output_dim))
    y_pred, y_true = predict_loader_den(model, test_loader, seen_classes, device)
    test_acc = accuracy_score_den(y_true, y_pred)
    results_dict["overall_accuracy"] = round(float(test_acc), 4)
    print(f"Overall Test Accuracy: {test_acc:.4f}")


# Save the configuration of the model trained
with open(config_path, "w") as f:
    yaml.dump(config_dict, f)

# Save the results dictionary
with open(results_path, "w") as f:
    yaml.dump(results_dict, f)

