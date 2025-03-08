import heapq
import random

from collections import deque
from attributes import CELL_HEIGHT, CELL_WIDTH, ROWS, COLS, ball_speed

path = []
total_obstacles = 0  # Initialize globally

def go_to_target(current_pos, target_pos, obstacle_list, algorithm):
    global path, total_obstacles

    cx, cy = current_pos
    
    # Pop waypoint if reached
    if path and (cx, cy) == path[0]:
        path.pop(0)
    
    # Recalculate if no path or next step is blocked
    if not path or (path and path[0] in obstacle_list):
        match algorithm:
            case "a_star":
                path = a_star(current_pos, target_pos, obstacle_list)
            case "bfs":
                path = bfs(current_pos, target_pos, obstacle_list)
            case "dfs":
                path = brute_force(current_pos, target_pos, obstacle_list)
            case "r_dfs":
                path = random_brute_force(current_pos, target_pos, obstacle_list)
            case "bell_ford":
                path = bellman_ford(current_pos, target_pos, obstacle_list)
        if not path:
            # print("Hi")
            # return -1, -1
            return cx, cy, False  # No path, stay put
    
    # Update total_obstacles after recalculation (not before)
    if len(obstacle_list) > total_obstacles:
        total_obstacles = len(obstacle_list)
    
    if path:
        next_pos = path[0]
        # Move towards next position
        if cx < next_pos[0]:
            cx += ball_speed
        elif cx > next_pos[0]:
            cx -= ball_speed
        if cy < next_pos[1]:
            cy += ball_speed
        elif cy > next_pos[1]:
            cy -= ball_speed
        
        # Snap to position if close enough
        if abs(cx - next_pos[0]) < ball_speed:
            cx = next_pos[0]
        if abs(cx - next_pos[1]) < ball_speed:
            cy = next_pos[1]
    
    return cx, cy, True



## A-Star Algorithm
def a_star(start, goal, obstacle_list):
    start = (start[0] // CELL_WIDTH, start[1] // CELL_HEIGHT)  # Convert to grid coordinates
    goal = (goal[0] // CELL_WIDTH, goal[1] // CELL_HEIGHT)
    
    open_set = [(0, start)]  # Priority queue with (f_score, position)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    obstacle_grid = set((x // CELL_WIDTH, y // CELL_HEIGHT) for x, y in obstacle_list)
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append((current[0] * CELL_WIDTH, current[1] * CELL_HEIGHT))
                current = came_from[current]
            path.append((start[0] * CELL_WIDTH, start[1] * CELL_HEIGHT))
            return path[::-1]  # Return reversed path
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Four directions
            neighbor = (current[0] + dx, current[1] + dy)
            
            if (0 <= neighbor[0] < COLS and 0 <= neighbor[1] < ROWS and 
                neighbor not in obstacle_grid):
                tentative_g_score = g_score[current] + 1
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # Return empty path if no path found

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance



## BFS
def bfs(start, goal, obstacle_list):
    start = (start[0] // CELL_WIDTH, start[1] // CELL_HEIGHT)  # Convert to grid coordinates
    goal = (goal[0] // CELL_WIDTH, goal[1] // CELL_HEIGHT)
    
    queue = deque([start])  # Queue for BFS
    came_from = {start: None}  # Track where we came from
    visited = {start}  # Track visited nodes
    
    obstacle_grid = set((x // CELL_WIDTH, y // CELL_HEIGHT) for x, y in obstacle_list)
    
    while queue:
        current = queue.popleft()
        
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append((current[0] * CELL_WIDTH, current[1] * CELL_HEIGHT))
                current = came_from[current]
            return path[::-1]  # Return reversed path
        
        # Explore four directions: right, down, left, up
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if (0 <= neighbor[0] < COLS and 
                0 <= neighbor[1] < ROWS and 
                neighbor not in obstacle_grid and 
                neighbor not in visited):
                queue.append(neighbor)
                visited.add(neighbor)
                came_from[neighbor] = current
    
    return []  # Return empty path if no path found

# DFS (Brute Force)
def brute_force(start, goal, obstacle_list):
    start = (start[0] // CELL_WIDTH, start[1] // CELL_HEIGHT)
    goal = (goal[0] // CELL_WIDTH, goal[1] // CELL_HEIGHT)
    obstacle_grid = set((x // CELL_WIDTH, y // CELL_HEIGHT) for x, y in obstacle_list)
    visited = set()
    
    def dfs(current, path_so_far):
        if current == goal:
            return path_so_far
        
        visited.add(current)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Explore: down, right, up, left
            next_pos = (current[0] + dx, current[1] + dy)
            if (0 <= next_pos[0] < COLS and 
                0 <= next_pos[1] < ROWS and 
                next_pos not in obstacle_grid and 
                next_pos not in visited):
                result = dfs(next_pos, path_so_far + [next_pos])
                if result:
                    return result
        return None
    
    path = dfs(start, [start])
    if path:
        return [(x * CELL_WIDTH, y * CELL_HEIGHT) for x, y in path]
    return []


# Random DFS
def random_brute_force(start, goal, obstacle_list):
    start = (start[0] // CELL_WIDTH, start[1] // CELL_HEIGHT)
    goal = (goal[0] // CELL_WIDTH, goal[1] // CELL_HEIGHT)
    obstacle_grid = set((x // CELL_WIDTH, y // CELL_HEIGHT) for x, y in obstacle_list)
    visited = set()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # down, right, up, left
    
    def dfs_random(current, path_so_far, max_attempts=1000):
        if current == goal:
            return path_so_far
        
        if len(path_so_far) > max_attempts:  # Limit to prevent infinite recursion
            return None
        
        visited.add(current)
        # Randomly shuffle directions to explore
        random.shuffle(directions)
        
        for dx, dy in directions:
            next_pos = (current[0] + dx, current[1] + dy)
            if (0 <= next_pos[0] < COLS and 
                0 <= next_pos[1] < ROWS and 
                next_pos not in obstacle_grid and 
                next_pos not in visited):
                result = dfs_random(next_pos, path_so_far + [next_pos])
                if result:
                    return result
        return None
    
    path = dfs_random(start, [start])
    if path:
        return [(x * CELL_WIDTH, y * CELL_HEIGHT) for x, y in path]
    return []


# Bellman Ford Algorithm
def bellman_ford(start, goal, obstacle_list):
    start = (start[0] // CELL_WIDTH, start[1] // CELL_HEIGHT)
    goal = (goal[0] // CELL_WIDTH, goal[1] // CELL_HEIGHT)
    obstacle_grid = set((x // CELL_WIDTH, y // CELL_HEIGHT) for x, y in obstacle_list)
    
    # Initialize distances and predecessors
    distances = {}
    predecessors = {}
    for y in range(ROWS):
        for x in range(COLS):
            node = (x, y)
            distances[node] = float('inf')
            predecessors[node] = None
    distances[start] = 0
    
    # Edges: all possible moves between grid cells
    edges = []
    for y in range(ROWS):
        for x in range(COLS):
            if (x, y) not in obstacle_grid:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < COLS and 0 <= ny < ROWS and (nx, ny) not in obstacle_grid:
                        edges.append(((x, y), (nx, ny), 1))  # Weight of 1 for each move
    
    # Bellman-Ford algorithm
    for _ in range(ROWS * COLS - 1):  # Relax edges V-1 times
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
    
    # Check for negative cycles (not applicable here, but included for completeness)
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return []  # Negative cycle detected (shouldn't happen in this grid)
    
    # Reconstruct path
    if distances[goal] == float('inf'):
        return []  # No path to goal
    
    path = []
    current = goal
    while current is not None:
        path.append((current[0] * CELL_WIDTH, current[1] * CELL_HEIGHT))
        current = predecessors[current]
    return path[::-1]