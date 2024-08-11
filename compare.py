import os
from run import run
from typing import List
import re
import numpy as np
import pygame

def compare() -> None:
    models: List[str] = os.listdir("./models")
    models: List[List[str]] = [re.findall(r"best_chromosome_(\w+).pkl", model) for model in models]
    models: List[str] = sum(models, [])

    if not models:
        print("No models found in the models directory. Please run 'train.py --model ('nn', 'linear')' first.")
        return
    
    if len(models) == 1:
        print(f"Only {models[0]} model found. Please train also the {"linear" if models[0]=="nn" else "nn"} model to compare.")
        return

    rewards: List[float] = []
    for model in models:
        total_reward = run(model)
        rewards.append(total_reward)
        pygame.quit()

    best: int = np.argmax(rewards)
    print(f"Model {models[best]} outperformed the {[model for model in models if model != models[best]][0]} with a total reward of {max(rewards)} (vs. {min(rewards)})")

if __name__ == "__main__":
    compare()