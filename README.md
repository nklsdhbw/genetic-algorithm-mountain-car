# 🚗 Solving OpenAI's Gymnasiums Mountain Car Problem with a generic algorithm

## Project Overview

This project implements a genetic algorithm to solve the MountainCar-v0 environment from OpenAI's Gymnasium. The algorithm uses two types of models: a linear model and a neural network model. The genetic algorithm optimizes the models to achieve the best performance in the MountainCar-v0 environment.

## 📂 Directory Structure

```
project-root/
│
├── models/
│   └── (Directory where trained models will be saved)
│
├── __init__.py
│
├── geneticalgorithm.py
│
├── nn.py
│
├── requirements.txt
│
├── run.py
│
└── train.py
```

## 📄 Files Description

1. **geneticalgorithm.py**:

   - Contains the main genetic algorithm functions, including initialization, fitness evaluation, selection, crossover, and mutation.
2. **nn.py**:

   - Defines the `NeuralNetwork` class used for the neural network model in the genetic algorithm.
3. **requirements.txt**:

   - Lists the required Python packages for the project.
4. **run.py**:

   - Loads and runs the trained models to evaluate their performance in the MountainCar-v0 environment. It supports both `linear`  and `nn` (neural network)  models.
5. **train.py**:

   - Trains the models using the genetic algorithm and saves the best models in the `models` directory. You can choose between `linear`or `nn` as model type. This will also be discussed later.
6. **__init__.py**:

   - Initialization file for the project module.

7. **compare.py**:

   - Compares the results of the nn and the linear model, if both were trained.

---

## ⚙️ Installation

### 🐍 Install Python 3.12

Download and install [Python3.12](https://www.python.org/downloads/)

### ⬇️ Clone the repository to your local machine

```python
git clone https://github.com/nklsdhbw/genetic-algorithm-mountain-car.git
```

and change to the project directory

### 🔨 Create a virtual environment using `venv`:

```sh
python3.12 -m venv .venv
```

### 🚀 Activate the virtual environment:

- On Windows:

```sh
   .venv\Scripts\activate
```

- On Unix or MacOS:

```sh
   source .venvbin/activate
```

### 📦 Install the required packages using pip:

```sh
pip install -r requirements.txt
```

---

## Usage

1. ### Training the Models:

To train the models using the genetic algorithm, run `python train.py --model` with the desired parameters. You need to pass `linear` or `nn` as argument value.
Other hyperparameters can also be set as shown in the following example
Example:

```sh
python train.py --model nn --population_size 150 --num_generations 200 --mutation_rate 0.05 --crossover_rate 0.7 --elite_size 10 --hidden_size 20
```

**Arguments for `train.py`:**

If the hyperparameters are not set, the default values will be used.

- `--model`: Specify the model type to run (`linear` or `nn`).
- `--population_size`: Specify the population size (default: 100).
- `--num_generations`: Specify the number of generations (default: 100).
- `--mutation_rate`: Specify the mutation rate (default: 0.1).
- `--crossover_rate`: Specify the crossover rate (default: 0.5).
- `--elite_size`: Specify the elite size (default: 5).
- `--hidden_size`: Specify the hidden size for the neural network (default: 10).
- `--grid_search`: Specify whether grid search for hyperparameter optimization will be used (default: False)

2. ### Running Grid Search:

To perform a grid search for hyperparameter optimization, use the  `--grid_search`  flag. Example:

    `python train.py --model nn --grid_search`

This will run a grid search over a predefined set of hyperparameters in the code and save the best model.
**Note:** All other flags set when setting `--grid_search` flag will be ignored.

3. ### Running the Models:

To run and evaluate the trained models, use `run.py` with the model type `('nn' or 'linear')`. Example:

```sh
python run.py --model nn
```

4. ### Compare the Models:

If both models were trained, you can compare the results of them via the following command:

````
python compare.py
````

## Acknowledgments

This project uses the MountainCar-v0 environment from OpenAI's Gymnasium and relies on several Python libraries, including numpy, gymnasium, and pygame.
