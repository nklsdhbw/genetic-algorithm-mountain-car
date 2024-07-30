import pickle
from geneticalgorithm import initialize_population, fitness, selection, crossover, mutate
import numpy as np

# Hyperparameter
population_size = 100
num_generations = 100
mutation_rate = 0.1
crossover_rate = 0.5
elitism = True
elite_size = 5

def run_genetic_algorithm(model_type):
    print(f"Running genetic algorithm for {model_type} model...")
    population = initialize_population(population_size=population_size, model_type=model_type)
    best_fitness = -float('inf')
    best_chromosome = None

    for generation in range(num_generations):
        fitnesses = [fitness(chromosome=chromosome, model_type=model_type) for chromosome in population]
        max_fitness = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_chromosome = population[np.argmax(fitnesses)].copy()  # Ensure we copy the best chromosome
        print(f'Generation {generation} | Best fitness: {max_fitness}')

        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elites = [population[i] for i in elite_indices]

        selected_population = selection(population=population, fitnesses=fitnesses, elite_size=elite_size)
        next_population = []
        for i in range(0, len(selected_population) - 1, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1=parent1, parent2=parent2, crossover_rate=crossover_rate, model_type=model_type)
            next_population.extend([mutate(chromosome=child1, mutation_rate=mutation_rate, model_type=model_type), mutate(chromosome=child2, mutation_rate=mutation_rate, model_type=model_type)])

        next_population.extend(elites)

        if len(selected_population) % 2 != 0:
            next_population.append(selected_population[-1])

        population = next_population

    best_filename = f'best_chromosome_{model_type}.pkl'
    with open(best_filename, 'wb') as f:
        pickle.dump(best_chromosome, f)
    print('Best solution found:', best_chromosome)

if __name__ == "__main__":
    run_genetic_algorithm(model_type='nn')