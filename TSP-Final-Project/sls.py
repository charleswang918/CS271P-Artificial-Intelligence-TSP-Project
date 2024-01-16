import random
import math
import time

import numpy as np


def generate_neighbor(solution):
    modified_solution = solution.copy()
    city_index1, city_index2 = random.sample(range(len(solution)), 2)
    modified_solution[city_index1], modified_solution[city_index2] = modified_solution[city_index2], modified_solution[city_index1]
    return modified_solution

def calculate_cost(graph, solution):
    total_cost = 0
    n = len(solution)
    for i in range(n - 1):
        total_cost += graph[solution[i]][solution[i + 1]]
    total_cost += graph[solution[-1]][solution[0]]  # Cost to return to the starting city
    return total_cost

def accept_worse_solution(cost_difference, temperature):
    probability = math.exp(-cost_difference / temperature)
    return random.random() < probability

def simulated_annealing(graph, initial_solution, initial_temperature, cooling_rate):
    current_solution = initial_solution
    best_solution = initial_solution
    best_cost = calculate_cost(graph, best_solution)
    current_temperature = initial_temperature

    while current_temperature > 0.1:
        new_solution = generate_neighbor(current_solution)
        new_cost = calculate_cost(graph, new_solution)
        cost_difference = new_cost - calculate_cost(graph, current_solution)

        if cost_difference < 0 or accept_worse_solution(cost_difference, current_temperature):
            current_solution = new_solution
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

        current_temperature *= cooling_rate

    return best_solution, best_cost

def random_dfs_tsp(graph, start_city):
    n = len(graph)
    visited = set([start_city])
    path = [start_city]

    def dfs(current_city):
        if len(visited) == n:
            # Check if a return to the start_city is possible.
            if graph[current_city][start_city] > 0:
                path.append(start_city)
                return True
            return False

        neighbors = [i for i in range(n) if graph[current_city][i] > 0]
        random.shuffle(neighbors)  # Shuffle neighbors to introduce randomness

        for next_city in neighbors:
            if next_city not in visited:
                visited.add(next_city)
                path.append(next_city)

                if dfs(next_city):
                    return True

                # Backtrack if the path does not lead to a solution
                visited.remove(next_city)
                path.pop()

        return False

    dfs(start_city)
    path.pop()

    return path

# Testing the simulated annealing algorithm
graph_example = [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
]


graph_1 = [[0, 10, 15, 20],
                  [10, 0, 35, 25],
                  [15, 35, 0, 30],
                  [20, 25, 30, 0]]
graph_2 = [[0, 10, 15, 20, 25],
                    [10, 0, 12, 8, 17],
                    [15, 12, 0, 19, 21],
                    [20, 8, 19, 0, 13],
                    [25, 17, 21, 13, 0]]

graph_3  = [[0, 5, 12, 8, 15, 10],
                    [5, 0, 7, 6, 9, 11],
                    [12, 7, 0, 10, 14, 20],
                    [8, 6, 10, 0, 11, 13],
                    [15, 9, 14, 11, 0, 18],
                    [10, 11, 20, 13, 18, 0]]


graph_4 = [[0, 8, 12, 15, 10, 9, 11],
                    [8, 0, 6, 10, 7, 5, 14],
                    [12, 6, 0, 9, 16, 13, 20],
                    [15, 10, 9, 0, 14, 11, 17],
                    [10, 7, 16, 14, 0, 8, 18],
                    [9, 5, 13, 11, 8, 0, 7],
                    [11, 14, 20, 17, 18, 7, 0]]
graph_5 = [[0, 9, 15, 20, 12, 10, 8, 14],
                    [9, 0, 14, 13, 8, 7, 5, 11],
                    [15, 14, 0, 11, 18, 10, 13, 17],
                    [20, 13, 11, 0, 16, 8, 15, 9],
                    [12, 8, 18, 16, 0, 6, 10, 12],
                    [10, 7, 10, 8, 6, 0, 9, 15],
                    [8, 5, 13, 15, 10, 9, 0, 7],
                    [14, 11, 17, 9, 12, 15, 7, 0]]
graph_large = [
    [0, 2, 3, 1, 4, 2, 5, 6, 3, 7, 8, 4],
    [2, 0, 5, 2, 1, 6, 4, 3, 5, 9, 2, 6],
    [3, 5, 0, 6, 7, 1, 2, 4, 6, 8, 3, 7],
    [1, 2, 6, 0, 3, 5, 7, 2, 4, 6, 1, 4],
    [4, 1, 7, 3, 0, 4, 6, 1, 3, 5, 2, 8],
    [2, 6, 1, 5, 4, 0, 3, 7, 2, 4, 6, 5],
    [5, 4, 2, 7, 6, 3, 0, 2, 6, 8, 3, 1],
    [6, 3, 4, 2, 1, 7, 2, 0, 4, 9, 2, 7],
    [3, 5, 6, 4, 3, 2, 6, 4, 0, 5, 3, 2],
    [7, 9, 8, 6, 5, 4, 8, 9, 5, 0, 7, 1],
    [8, 2, 3, 1, 2, 6, 3, 2, 3, 7, 0, 5],
    [4, 6, 7, 4, 8, 5, 1, 7, 2, 1, 5, 0]
]

def generate_tsp_matrix(num_cities):
    # Seed for reproducibility
    np.random.seed(42)

    # Generate a random symmetric matrix with distances between cities
    tsp_matrix = np.random.randint(1, 100, size=(num_cities, num_cities))
    tsp_matrix = (tsp_matrix + tsp_matrix.T) // 2  # Make the matrix symmetric

    # Set diagonal elements to zero (distance from a city to itself)
    np.fill_diagonal(tsp_matrix, 0)

    return tsp_matrix

num_cities = 100
tsp_matrix = generate_tsp_matrix(num_cities)

start_time = time.time()
initial_start_city=0
test_graph=tsp_matrix

initial_solution_example = random_dfs_tsp(test_graph, initial_start_city)
initial_solution_example1 = [0, 1, 2, 3]
initial_temperature_example = 10000
cooling_rate_example = 0.995

sa_solution, sa_cost = simulated_annealing(test_graph, initial_solution_example, initial_temperature_example, cooling_rate_example)

print("Optimal Tour Order:", sa_solution)
print("Optimal Tour Distance:", sa_cost)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

