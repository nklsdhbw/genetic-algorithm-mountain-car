import numpy as np
import gymnasium as gym
import pickle
import argparse
import os
import re
from typing import Union, List, Dict
from nn import NeuralNetwork



def run(model: str) -> float:
    models: List[str] = os.listdir("./models")
    if not models:
        red: str = "\033[91m"
        reset: str = "\033[0m"
        print(f"{red}ERROR: No models found in the models directory. Please run 'train.py --model' first.{reset}")
        exit(1)
    print(f"Running the best {model} model...")
    best_chromosome: Union[np.ndarray, NeuralNetwork] = load_model(f"./models/best_chromosome_{model}.pkl")
    total_reward: float = run_model(best_chromosome, model_type=model)
    print(f"Total reward using the best {model} model: {total_reward}")
    return total_reward



def load_model(filename: str) -> Union[np.ndarray, NeuralNetwork]:
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run_model(chromosome: Union[np.ndarray, NeuralNetwork], model_type: str = 'linear') -> float:
    env = gym.make('MountainCar-v0', render_mode='human')
    state, _ = env.reset()
    total_reward: float = 0
    for _ in range(200):
        env.render()
        if model_type == 'linear':
            action: int = np.argmax(np.dot(state, chromosome))
        else:
            action_values: np.ndarray = chromosome.forward(state)
            action: int = np.argmax(action_values)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    env.close()
    return total_reward  # Total reward (negative value) reflects the number of steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the trained model for MountainCar.')
    
    models: List[str] = os.listdir("./models")
    if not models:
        red: str = "\033[91m"
        reset: str = "\033[0m"
        print(f"{red}ERROR: No models found in the models directory. Please run 'train.py --model' first.{reset}")
        exit(1)
    models: List[List[str]] = [re.findall(r"best_chromosome_(\w+).pkl", model) for model in models]
    models: List[str] = sum(models, [])
    models_string: str = f"('{models[0]}'" + "".join([f", '{model}'" for model in models[1:]]) + ")"
    parser.add_argument("--model", type=str, choices=models, required=True, help=f"Specify the model type to run {models_string}.")
    args = parser.parse_args()

    total_reward = run(args.model)