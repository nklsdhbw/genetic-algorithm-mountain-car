import pickle
from geneticalgorithm import initialize_population, fitness, selection, crossover, mutate
import numpy as np
import gymnasium as gym

# Environment
env = gym.make('MountainCar-v0')

# Hyperparameter
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
ELITE_SIZE = 5
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n
HIDDEN_SIZE = 10

def run_genetic_algorithm(model_type='linear'):
    print(f"Running genetic algorithm for {model_type} model...")
    population = initialize_population(population_size=POPULATION_SIZE, model_type=model_type, hidden_size=HIDDEN_SIZE, state_size=STATE_SIZE, action_size=ACTION_SIZE)
    best_fitness = -float('inf')
    best_chromosome = None

    for generation in range(NUM_GENERATIONS):
        fitnesses = [fitness(chromosome=chromosome, env=env, model_type=model_type) for chromosome in population]
        max_fitness = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_chromosome = population[np.argmax(fitnesses)].copy()  # Ensure we copy the best chromosome
        print(f'Generation {generation} | Best fitness: {max_fitness}')

        elite_indices = np.argsort(fitnesses)[-ELITE_SIZE:]
        elites = [population[i] for i in elite_indices]

        selected_population = selection(population=population, fitnesses=fitnesses, elite_size=ELITE_SIZE)
        next_population = []
        for i in range(0, len(selected_population) - 1, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1=parent1, parent2=parent2, crossover_rate=CROSSOVER_RATE, state_size=STATE_SIZE, model_type=model_type)
            next_population.extend([mutate(chromosome=child1, mutation_rate=MUTATION_RATE, state_size=STATE_SIZE, action_size=ACTION_SIZE, model_type=model_type), mutate(chromosome=child2, mutation_rate=MUTATION_RATE, state_size=STATE_SIZE, action_size=ACTION_SIZE, model_type=model_type)])

        next_population.extend(elites)

        if len(selected_population) % 2 != 0:
            next_population.append(selected_population[-1])

        population = next_population

    best_filename = f'./models/best_chromosome_{model_type}.pkl'
    with open(best_filename, 'wb') as f:
        pickle.dump(best_chromosome, f)
    print('Best solution found:', best_chromosome)

if __name__ == "__main__":
    run_genetic_algorithm(model_type='nn')