import numpy as np
import gymnasium as gym
import pickle
import argparse
import os
import re
from typing import Union
from nn import NeuralNetwork

# Environment
env = gym.make('MountainCar-v0', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

def load_model(filename: str) -> Union[np.ndarray, NeuralNetwork]:
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run_model(chromosome: Union[np.ndarray, NeuralNetwork], model_type: str = 'linear') -> float:
    state, _ = env.reset()
    total_reward = 0
    for _ in range(200):
        env.render()
        if model_type == 'linear':
            action = np.argmax(np.dot(state, chromosome))
        else:
            action_values = chromosome.forward(state)
            action = np.argmax(action_values)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    env.close()
    return total_reward  # Total reward (negative value) reflects the number of steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the trained model for MountainCar.')
    models = os.listdir("./models")
    if not models:
        red = "\033[91m"
        reset = "\033[0m"
        print(f"{red}ERROR: No models found in the models directory. Please run 'train.py --model' first.{reset}")
        exit(1)
    models = [re.findall(r"best_chromosome_(\w+).pkl", model) for model in models]
    models = sum(models, [])
    models_string = f"('{models[0]}'" + "".join([f", '{model}'" for model in models[1:]]) + ")"
    parser.add_argument("--model", type=str, choices=models, required=True, help=f"Specify the model type to run {models_string}.")
    args = parser.parse_args()

    print(f"Running the best {args.model} model...")
    best_chromosome = load_model(f"./models/best_chromosome_{args.model}.pkl")
    total_reward = run_model(best_chromosome, model_type=args.model)
    print(f"Total reward using the best {args.model} model: {total_reward}")