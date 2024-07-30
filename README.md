

#### Project Overview

This project implements a genetic algorithm to solve the MountainCar-v0 environment from OpenAI's Gymnasium. The algorithm uses two types of models: a linear model and a neural network model. The genetic algorithm optimizes the models to achieve the best performance in the MountainCar-v0 environment.

---

#### Directory Structure

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

---

#### Files Description

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

---

#### Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment using `venv`:

```sh
python -m venv .venv
```

4. Activate the virtual environment:

   - On Windows:

   ```sh
   .venv\Scripts\activate
   ```

   - On Unix or MacOS:

   ```sh
   source .venvbin/activate
   ```

5. Install the required packages using pip:

```sh
pip install -r requirements.txt
```

---

#### Usage


1. **Training the Models:**

To train the models using the genetic algorithm, run `train.py` with the desired parameters. Example:

```sh
python train.py --model nn --population_size 150 --num_generations 200 --mutation_rate 0.05 --crossover_rate 0.7 --elite_size 10 --hidden_size 20
```

Note: The following arguments except `--model` are only required when `--model` is set to `nn`. 
If `--model` is `linear`, no additional arguments need to be provided.

**Arguments for `train.py`:**

- `--model`: Specify the model type to run (`linear` or `nn`).
- `--population_size`: Specify the population size (default: 100).
- `--num_generations`: Specify the number of generations (default: 100).
- `--mutation_rate`: Specify the mutation rate (default: 0.1).
- `--crossover_rate`: Specify the crossover rate (default: 0.5).
- `--elite_size`: Specify the elite size (default: 5).
- `--hidden_size`: Specify the hidden size for the neural network (default: 10).

**Default Values:**

If these arguments are not set for the `nn` model, the default values will be used.





2. **Running the Models:**

To run and evaluate the trained models, use `run.py` with the model type `('nn' or 'linear')`. Example:

```sh
python run.py --model nn
```

---

#### Acknowledgments

This project uses the MountainCar-v0 environment from OpenAI's Gymnasium and relies on several Python libraries, including numpy, gymnasium, and pygame.
