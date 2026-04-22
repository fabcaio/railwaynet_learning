# Description

Repository containing the data and code of the paper entitled “Learning-based model predictive control for passenger-oriented train rescheduling with flexible train composition” available as a preprint [here](https://arxiv.org/abs/2502.15544). Corresponds to Chapter 3 of the thesis (the part related to the results without state reduction).

This repository only contains the Python scripts and notebooks. The full repository with the dataset is available as the file "chapter3.rar" (folder "without_state_reduction") [here](https://doi.org/10.4121/277b2054-94e1-4d6a-b16c-db448bd8c4c5).

Acknowledgement: This research has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 101018826 - CLariNet).
# Setup    
  
This project uses [`uv`] for reproducible dependency management.  
  
Developed and tested on Windows 11 with `uv` version `0.11.6` and `Python 3.11.15`.  
  
Please install `uv` first by following the official documentation.    
## Installation

Download "chapter3.rar" from [this link](https://doi.org/10.4121/277b2054-94e1-4d6a-b16c-db448bd8c4c5), and extract the folder "without_state_reduction". Then run the following commands to install the necessary Python version and packages.
  
```powershell  
cd <PATH:project_folder>  
uv python install  
uv sync --locked
```
# Folders

data_optimal: contains the data for the training of the machine learning models. In the following the naming of the files is explained:
- milp: data obtained by the solution of a mixed-integer linear program
- minlp: data obtained by the solution of a mixed-integer nonlinear program
- cl: data obtained by closed-loop simulations
- ol: data obtained by open-loop simulations
- N20: prediction horizon set to 20
- N40: prediction horizon set to 40
Check the "rail_gen_optimal_data_original.py" for more details about data generation.

data_railway: contains the file "training_sets.npy" that stores historical data about the rail network used in the case study.

tests: the outcome of the tests scripts described below.

training_data: trained weights and hyperparameters of several neural networks. Example: the file "milp_cl_N40_006_weight" stores the weights of the PyTorch model and the file "milp_cl_N40_006_info.npy" stores a Python dictionary with the corresponding hyperparameters. The last three digits in the name simply index the model.
# Files

analysis_networks.ipynb: analyses the results obtained from the scripts "tests_networks_cl.py" and "tests_networks_ol.py". The goal is to assess the performance of the individual model to form the ensemble of neural networks.

analysis_tests_learning_cl.ipynb: analyses the results obtained from the script "tests_learning_cl_heuristic.py" (for the ensemble).

analysis_tests_learning_ol.ipynb: analyses the results obtained from the script "tests_learning_ol.py" (for the ensemble).

analysis_tests_minlp_milp.ipynb: analyses the results obtained from the script "tests_minlp_milp.py".

rail_data_preprocess_original.py: defines functions for data pre-processing.

rail_fun.py: creates auxiliary functions for data pre-processing, computing the step cost, and retrieving information from the system.

rail_gen_optimal_data_original.py: generates the data used for the training of the supervised learning approach by solving the underlying mixed-integer nonlinear (or linear) program. The flag "testing=False" sets the script to be run in a computing cluster. Alternatively, "testing=True" is used for local small tests.

rail_learning_cluster.py: trains the neural networks and saves the resulting weights and hyperparameters. The flag "testing=False" sets the script to be run in a computing cluster. Alternatively, "testing=True" is used for local small tests.

rail_rl_env.py: defines the railway network system (dynamics, parameters, etc...). It has a very similar architecture to environments in the library Gymnasium.

rail_training.py: creates classes which represent several neural network architectures, creates a function which updates the neural network weights by backpropagation, and defines auxiliary functions for data pre-processing and testing.

tests_learning_cl_heuristic.py: tests the closed-loop performance of the ensemble of the selected neural networks. It applies the feasibility recovery policy described in the paper when the learning-based solution is not feasible.

tests_learning_ol.py: tests the open-loop performance of the ensemble of the selected neural networks.

tests_minlp_milp.py: compares the performance of the neural networks trained with the models trained on the dataset based on mixed-integer nonlinear program solutions and on the models based on the dataset based on mixed-integer linear programs (approximation). It tests single models.

tests_networks_cl.py: tests the closed-loop performance of single models.

tests_networks_ol.py: tests the open-loop performance of single models.

tests_ntbk.ipynb: plots the training and validation losses for one trained model.

The uv project is defined by the following files:
- .python-version
- pyproject.toml
- uv.lock
