# Continual Learning Experiments
> A research-oriented continual learning project exploring **Task-IL**, **Domain-IL**, and **Class-IL** scenarios using modern neural architectures and mitigation strategies against catastrophic forgetting.
This repository implements and compares **baseline training**, **Elastic Weight Consolidation (EWC)**, **Replay Buffers**, and **Replay + EWC** across multiple models including **MLPs**, **Progressive Neural Networks**, and **Dynamically Expandable Networks (DEN)**, primarily evaluated on **MNIST**.

The project is designed as an **experimental framework**, emphasizing:
* clean separation between training, evaluation, and comparison
* standardized continual learning metrics
* extensibility for new methods and models

#### Tables of contents
* [Path tree](#path-tree)
* [Direct links to folders](#direct-links-to-folders) 
* [Installation](#installation)
* [Continual learning overview](#continual-learning-overview)
* [Implemented scenarios](#implemented-scenarios)
* [Models](#models)
* [Methods](#methods)
* [Training & evaluation pipeline](#training--evaluation-pipeline)
* [Running experiments](#running-experiments)
* [Experimental results](#experimental-results)
* [Results comparison](#results-comparison)


## Path tree
```
Continual_Learning_Experiments/
├── experiments/
│   ├── ewc/
│   ├── none/
│   ├── replay/
│   ├── replay_ewc/
│   ├── compare_methods.py
│   └── launcher.py
│   
├── src/
│   ├── data/
│   ├── methods/
│   ├── models/
│   ├── tasks/
│   ├── training/
│   └── main.py
│   
├── tests/
│   
├── README.md
└── requirements.txt
```


## Direct links to folders
* [experiments/](./experiments/): Contains experiment scripts for different methods and comparison
    * [ewc/](./experiments/ewc/): Experiments using Elastic Weight Consolidation
    * [none/](./experiments/none/): Baseline experiments without any continual learning method
    * [replay/](./experiments/replay/): Experiments using Replay Buffers
    * [replay_ewc/](./experiments/replay_ewc/): Experiments combining Replay Buffers and EWC
    * [compare_methods.py](./experiments/compare_methods.py): Script to compare results across different methods
    * [launcher.py](./experiments/launcher.py): Utility to launch experiments with different configurations
* [src/](./src/): Source code for data handling, models, methods, tasks, and training
    * [data/](./src/data/): Data loading and preprocessing utilities
    * [methods/](./src/methods/): Implementation of continual learning methods (EWC, Replay)
    * [models/](./src/models/): Neural network architectures (MLP, ProgNet, DEN)
    * [tasks/](./src/tasks/): Task definitions for different continual learning scenarios
    * [training/](./src/training/): Training and evaluation pipelines
    * [main.py](./src/main.py): Main entry point for running experiments
* [tests/](./tests/): Unit tests for various components of the project


## Installation
1. Clone the project:
```
git clone git@github.com:tomcuel/Continual_Learning_Experiments.git
cd Continual_Learning_Experiments
```
2. Create a python virtual environment: 
```
python -m venv venv
source venv/bin/activate  # macOS / Linux
```
3. Install the requirements:
```
python -m pip -r requirements.txt
```
4. Make sure to have Jupyter Notebook installed to run the `.ipynb` files


## Continual learning overview
Continual Learning aims to train models on a sequence of tasks while:
* retaining knowledge of previous tasks
* minimizing **catastrophic forgetting**
* avoiding retraining from scratch

This project focuses on **supervised continual learning** and systematically evaluates how different mitigation strategies behave under controlled experimental settings.


## Implemented scenarios
The project supports the three standard continual learning settings:

#### Task-Incremental Learning (Task-IL)
* Task identity is known at inference time
* Separate output heads per task
* Simplest setting, used as a baseline

#### Domain-Incremental Learning (Domain-IL)
* Same labels, different input distributions
* No task identity at inference
* Requires feature robustness

#### Class-Incremental Learning (Class-IL)
* New classes appear over time
* Single shared output head
* Hardest and most realistic setting


## Models
Implemented architectures include:

#### Multi-Layer Perceptron (MLP)
* Shared backbone
* Optional task-specific heads
* BatchNorm & Dropout support

#### Progressive Neural Network (ProgNet)
* One column per task
* Lateral connections to previous columns
* No forgetting by design, but parameter growth

#### Dynamically Expandable Network (DEN)
* Selective neuron expansion
* Sparse regularization
* Balances plasticity and stability


## Methods
The following continual learning strategies are implemented:

#### None (Naive fine-tuning)
* Baseline without forgetting mitigation

#### Elastic Weight Consolidation (EWC)
* Fisher Information Matrix
* Penalizes changes to important parameters

#### Replay Buffer
* Stores past samples
* Mixed with current task batches

#### Replay + EWC
* Combines data-based and regularization-based methods

## Training & Evaluation Pipeline
Each experiment follows a strict pipeline:
1. Task sequence creation (Task-IL / Domain-IL / Class-IL)
2. Sequential task training
3. Optional replay sampling
4. Optional EWC penalty
5. Evaluation on **all previously seen tasks**
6. Metrics aggregation
**Standard metrics:**
* **Average Accuracy (AA)**
* **Forgetting (F)**
* **Backward Transfer (BWT)**

All metrics are computed from the **accuracy matrix** produced during training.


## Running experiments
Run a single experiment with:
```
python src/main.py --type mlp_task --method replay_ewc --epochs 10
```
**Arguments:**
* `--type`: Model and task type (`mlp_task`, `mlp_domain`, `mlp_class`, `prog_net_task`, `den_class`)
* `--method`: Continual learning method (`none`, `ewc`, `replay`, `replay_ewc`)
* `--epochs`: Number of epochs per task

Batch experiments can be launched using:
```
python experiments/launcher.py
python experiments/compare_methods.py
```

## Experimental results
Each run automatically saves configs and results to the corresponding method folder under `experiments/`.


#### Example for an MLP with Replay + EWC in Task-IL
**Configs:**
```yaml
activation_functions:
- leaky_relu
- tanh
- relu
batch_size: 16
dropout_rates:
- 0.4459
- 0.1863
- 0.1279
gamma: 0.4855325340941379
hidden_layers:
- 216
- 251
- 197
lambda_ewc: 0.4
learning_rate: 0.0004980564413343545
replay_weight: 1.0
step_size: 14.161634448046335
weight_decay: 8.560817132298919e-06
```
**Results:**
```
backward_transfer: -0.9994
forgetting: 0.9994
overall_accuracy: 0.1958
train_average_accuracy: 0.1997
```

## Results comparison
The repository includes a dedicated comparison script that:
* loads all experiment results
* groups by scenario, model, and method
* prints clean comparison tables

Example output (fictional data):
```markdown
Model        Method        AA     F     BWT
-------------------------------------------
MLP          none        0.62   0.41  -0.32
MLP          replay      0.78   0.19  -0.08
ProgNet      ewc         0.81   0.12  -0.05
DEN          replay_ewc  0.84   0.10  -0.03
```