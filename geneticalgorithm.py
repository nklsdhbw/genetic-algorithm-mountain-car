import numpy as np
import gymnasium as gym
import random
from nn import NeuralNetwork
from typing import List, Union

def initialize_population(population_size: int, model_type: str = 'linear', hidden_size: int = None, state_size: int = None, action_size: int = None) -> List[Union[np.ndarray, NeuralNetwork]]:
    if model_type == 'linear':
        return [np.random.randn(state_size, action_size) for _ in range(population_size)]
    else:
        return [NeuralNetwork(hidden_size=hidden_size, state_size=state_size, action_size=action_size) for _ in range(population_size)]

def fitness(chromosome: Union[np.ndarray, NeuralNetwork], env: gym.Env, model_type: str = 'linear') -> float:
    state, _ = env.reset()
    total_reward = 0
    for _ in range(200):
        if model_type == 'linear':
            action = np.argmax(np.dot(state, chromosome))
        else:
            action_values = chromosome.forward(state)
            action = np.argmax(action_values)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward  # Note: total_reward is negative

def selection(population: List[Union[np.ndarray, NeuralNetwork]], fitnesses: List[float], elite_size: int) -> List[Union[np.ndarray, NeuralNetwork]]:
    min_fitness = min(fitnesses)
    offset = abs(min_fitness) + 1 if min_fitness < 0 else 0
    adjusted_fitnesses = [f + offset for f in fitnesses]
    selected = random.choices(population, weights=adjusted_fitnesses, k=len(population) - elite_size)
    return selected

def crossover(parent1: Union[np.ndarray, NeuralNetwork], parent2: Union[np.ndarray, NeuralNetwork], crossover_rate: float, state_size: int, model_type: str = 'linear') -> Union[np.ndarray, NeuralNetwork]:
    if model_type == 'linear':
        if random.random() < crossover_rate:
            point = random.randint(1, state_size - 1)
            child1 = np.vstack((parent1[:point], parent2[point:]))
            child2 = np.vstack((parent2[:point], parent1[point:]))
            return child1, child2
        else:
            return parent1, parent2
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
        for key in parent1.__dict__.keys():
            if random.random() < crossover_rate:
                point = random.randint(1, getattr(parent1, key).shape[0] - 1)
                if getattr(parent1, key).ndim == 1:
                    setattr(child1, key, np.concatenate((getattr(parent1, key)[:point], getattr(parent2, key)[point:])))
                    setattr(child2, key, np.concatenate((getattr(parent2, key)[:point], getattr(parent1, key)[point:])))
                else:
                    setattr(child1, key, np.vstack((getattr(parent1, key)[:point, :], getattr(parent2, key)[point:, :])))
                    setattr(child2, key, np.vstack((getattr(parent2, key)[:point, :], getattr(parent1, key)[point:, :])))
        return child1, child2

def mutate(chromosome: Union[np.ndarray, NeuralNetwork], mutation_rate: float, state_size: int, action_size: int, model_type: str = 'linear') -> Union[np.ndarray, NeuralNetwork]:
    if model_type == 'linear':
        if random.random() < mutation_rate:
            index = random.randint(0, state_size - 1)
            chromosome[index] = np.random.randn(action_size)
    else:
        for key in chromosome.__dict__.keys():
            if random.random() < mutation_rate:
                index = random.randint(0, getattr(chromosome, key).shape[0] - 1)
                if getattr(chromosome, key).ndim == 1:
                    getattr(chromosome, key)[index] = np.random.randn()
                else:
                    getattr(chromosome, key)[index, :] = np.random.randn(getattr(chromosome, key).shape[1])
    return chromosome