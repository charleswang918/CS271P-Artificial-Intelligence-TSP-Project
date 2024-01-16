import heapq


def BranchAndBound_DFS(graph, start_city):
    best_path = []
    best_weight = float('inf')

    def dfs(path, current_city, current_weight):
        nonlocal best_path, best_weight

        if len(path) == len(graph) and graph[current_city][start_city] > 0:
            path_cost = current_weight + graph[current_city][start_city]
            if path_cost < best_weight:
                best_path = path + [start_city]
                best_weight = path_cost
            return

        for next_city in range(len(graph)):
            if graph[current_city][next_city] > 0 and next_city not in path:
                next_weight = current_weight + graph[current_city][next_city]
                if next_weight + calculate_bound(graph, path, next_city) < best_weight:
                    dfs(path + [next_city], next_city, next_weight)

    dfs([start_city], start_city, 0)
    return best_path, best_weight


def calculate_bound(graph, path, current_city):
    mst_weight = prim_mst_weight(graph, path + [current_city])
    min_edge_weight = min_edge_weight_to_unvisited(path, current_city, graph)
    return mst_weight + min_edge_weight


def prim_mst_weight(graph, path):
    mst_weight = 0
    visited = set(path)
    edges = [(0, path[-1], next_city) for next_city in range(len(graph)) if
             next_city not in visited and graph[path[-1]][next_city] > 0]
    heapq.heapify(edges)

    while edges:
        weight, from_city, to_city = heapq.heappop(edges)
        if to_city not in visited:
            visited.add(to_city)
            mst_weight += weight
            for next_city in range(len(graph)):
                if next_city not in visited and graph[to_city][next_city] > 0:
                    heapq.heappush(edges, (graph[to_city][next_city], to_city, next_city))
    return mst_weight


def min_edge_weight_to_unvisited(path, current_city, graph):
    min_weight = float('inf')
    for next_city in range(len(graph)):
        if next_city not in path and graph[current_city][next_city] > 0:
            min_weight = min(min_weight, graph[current_city][next_city])
    return min_weight if min_weight != float('inf') else 0


# 测试用例
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

graph_2 = [[0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]]

graph_1 = [[0, 10, 15, 20],
                  [10, 0, 35, 25],
                  [15, 35, 0, 30],
                  [20, 25, 30, 0]]
graph_5 = [[0, 10, 15, 20, 25],
                    [10, 0, 12, 8, 17],
                    [15, 12, 0, 19, 21],
                    [20, 8, 19, 0, 13],
                    [25, 17, 21, 13, 0]]
graph_6 = [[0, 5, 12, 8, 15, 10],
                    [5, 0, 7, 6, 9, 11],
                    [12, 7, 0, 10, 14, 20],
                    [8, 6, 10, 0, 11, 13],
                    [15, 9, 14, 11, 0, 18],
                    [10, 11, 20, 13, 18, 0]]
graph_7 = [[0, 8, 12, 15, 10, 9, 11],
                    [8, 0, 6, 10, 7, 5, 14],
                    [12, 6, 0, 9, 16, 13, 20],
                    [15, 10, 9, 0, 14, 11, 17],
                    [10, 7, 16, 14, 0, 8, 18],
                    [9, 5, 13, 11, 8, 0, 7],
                    [11, 14, 20, 17, 18, 7, 0]]
graph_8 = [[0, 9, 15, 20, 12, 10, 8, 14],
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
start_city = 0
test_graph=graph_large
path, weight = BranchAndBound_DFS(test_graph, start_city)
print("Path:", path)
print("Total Weight:", weight)
