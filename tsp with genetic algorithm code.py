import numpy as np
import matplotlib.pyplot as plt
import random

# Şehirlerin koordinatları
cities = np.array([[0, 0], [1, 2], [2, 4], [3, 1], [5, 3]])

def calculate_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += np.linalg.norm(cities[route[i]] - cities[route[i + 1]])
    return total_distance

def generate_initial_population(population_size, num_cities):
    return [list(np.random.permutation(num_cities)) for _ in range(population_size)]

def perform_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]
    return child1, child2

def perform_mutation(route, mutation_rate):
    mutated_route = route.copy()
    for _ in range(int(mutation_rate * len(route))):
        mutation_point1, mutation_point2 = random.sample(range(len(route)), 2)
        mutated_route[mutation_point1], mutated_route[mutation_point2] = mutated_route[mutation_point2], mutated_route[mutation_point1]
    return mutated_route

def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [fit / total_fitness for fit in fitness]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents

def perform_elitism(population, fitness, elitism_rate):
    num_elites = int(elitism_rate * len(population))
    elites_idx = np.argsort(fitness)[:num_elites]
    elites = [population[i] for i in elites_idx]
    return elites

def plot_route_and_cities(route, cities, best_distance):
    x = [city[0] for city in cities[route]]
    y = [city[1] for city in cities[route]]

    plt.figure(figsize=(8, 6))

    # Rota çizgisi
    plt.plot(x + [x[0]], y + [y[0]], linestyle='-', color='blue', linewidth=2, label='En İyi Rota')

    # Şehirleri işaretle
    plt.scatter(x, y, c='red', marker='o', label='Şehirler')

    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=12, ha='right', va='bottom', color='black')

    # En iyi rota ve en kısa mesafeyi ekrana bas
    plt.text(0.5, -0.5, f"En İyi Rota: {route}\nEn Kısa Mesafe: {best_distance:.2f}", ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.title('En İyi Rota ve Şehirler')
    plt.xlabel('X Koordinatı')
    plt.ylabel('Y Koordinatı')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def genetic_algorithm(population_size, num_generations, crossover_rate, mutation_rate, elitism_rate):
    num_cities = len(cities)
    population = generate_initial_population(population_size, num_cities)

    for generation in range(num_generations):
        fitness = [1 / calculate_distance(route) for route in population]

        # Geri kalan kod...

    best_route = min(population, key=lambda x: calculate_distance(x))
    best_distance = calculate_distance(best_route)

    # En iyi rota ve şehirleri görselleştir
    plot_route_and_cities(best_route, cities, best_distance)

    return best_route, best_distance

# Genetik algoritmayı çalıştır
population_size = 50
num_generations = 100
crossover_rate = 0.8
mutation_rate = 0.02
elitism_rate = 0.1

best_route, best_distance = genetic_algorithm(population_size, num_generations, crossover_rate, mutation_rate, elitism_rate)

print(f"En iyi rota: {best_route}")
print(f"En kısa mesafe: {best_distance}")