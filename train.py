import pickle
import argparse
from geneticalgorithm import initialize_population, fitness, selection, crossover, mutate
import numpy as np
import gymnasium as gym
from typing import List, Union
from nn import NeuralNetwork

# Environment
env = gym.make('MountainCar-v0')
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n

def run_genetic_algorithm(
    model_type: str, 
    population_size: int, 
    num_generations: int, 
    mutation_rate: float, 
    crossover_rate: float, 
    elite_size: int, 
    hidden_size: int
) -> None:
    print(f"Running genetic algorithm for {model_type} model...")
    population: List[Union[np.ndarray, NeuralNetwork]] = initialize_population(
        population_size=population_size, 
        model_type=model_type, 
        hidden_size=hidden_size, 
        state_size=STATE_SIZE, 
        action_size=ACTION_SIZE
    )
    best_fitness: float = -float('inf')
    best_chromosome: Union[np.ndarray, NeuralNetwork] = None

    for generation in range(num_generations):
        fitnesses: List[float] = [fitness(chromosome=chromosome, env=env, model_type=model_type) for chromosome in population]
        max_fitness: float = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_chromosome = population[np.argmax(fitnesses)].copy()
        print(f'Generation {generation} | Best fitness: {max_fitness}')

        elite_indices: np.ndarray = np.argsort(fitnesses)[-elite_size:]
        elites: List[Union[np.ndarray, NeuralNetwork]] = [population[i] for i in elite_indices]

        selected_population: List[Union[np.ndarray, NeuralNetwork]] = selection(population=population, fitnesses=fitnesses, elite_size=elite_size)
        next_population: List[Union[np.ndarray, NeuralNetwork]] = []
        for i in range(0, len(selected_population) - 1, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1=parent1, parent2=parent2, crossover_rate=crossover_rate, state_size=STATE_SIZE, model_type=model_type)
            next_population.extend([mutate(chromosome=child1, mutation_rate=mutation_rate, state_size=STATE_SIZE, action_size=ACTION_SIZE, model_type=model_type), mutate(chromosome=child2, mutation_rate=mutation_rate, state_size=STATE_SIZE, action_size=ACTION_SIZE, model_type=model_type)])

        next_population.extend(elites)

        if len(selected_population) % 2 != 0:
            next_population.append(selected_population[-1])

        population = next_population

    best_filename: str = f'./models/best_chromosome_{model_type}.pkl'
    with open(best_filename, 'wb') as f:
        pickle.dump(best_chromosome, f)
    print('Best solution found:', best_chromosome)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the genetic algorithm for MountainCar.')
    parser.add_argument('--model', type=str, choices=['linear', 'nn'], required=True, help='Specify the model type to run (linear or nn).')
    parser.add_argument('--population_size', type=int, default=100, help='Specify the population size.')
    parser.add_argument('--num_generations', type=int, default=100, help='Specify the number of generations.')
    parser.add_argument('--mutation_rate', type=float, default=0.1, help='Specify the mutation rate.')
    parser.add_argument('--crossover_rate', type=float, default=0.5, help='Specify the crossover rate.')
    parser.add_argument('--elite_size', type=int, default=5, help='Specify the elite size.')
    parser.add_argument('--hidden_size', type=int, default=10, help='Specify the hidden size for the neural network.')

    args = parser.parse_args()

    defaults = {
        'population_size': 100,
        'num_generations': 100,
        'mutation_rate': 0.1,
        'crossover_rate': 0.5,
        'elite_size': 5,
        'hidden_size': 10
    }

    yellow = "\033[93m"
    reset = "\033[0m"
    
    if args.model != "nn":
        defaults.pop("hidden_size")
    warnings: List[str] = []
    for param, default_value in defaults.items():
        if getattr(args, param) == default_value:
            warnings.append(f"{param} not set, using default value: {default_value}")

    if warnings:
        print(f"{yellow}WARNING: The following parameters were not set and will use default values:{reset}")
        for warning in warnings:
            print(f"{yellow}{warning}{reset}")

    run_genetic_algorithm(
        model_type=args.model, 
        population_size=args.population_size, 
        num_generations=args.num_generations, 
        mutation_rate=args.mutation_rate, 
        crossover_rate=args.crossover_rate, 
        elite_size=args.elite_size, 
        hidden_size=args.hidden_size
    )