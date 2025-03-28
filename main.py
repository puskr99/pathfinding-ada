import pygame
import sys
import random
import time
import matplotlib.pyplot as plt
import heapq
from collections import deque

# from ball import go_to_target
# from obstacles import get_obstacles
from attributes import *
from plot import plot_simulation_results

# Initialize Pygame
pygame.init()

# Screen and grid dimensions
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pathfinding with Obstacles")

markers = []
lines = []
trail = []

# Buttons for algorithm selection
ASTAR_BUTTON_RECT = pygame.Rect(10, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
BFS_BUTTON_RECT = pygame.Rect(120, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
DFS_BUTTON_RECT = pygame.Rect(240, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
# RANDOM_DFS_BUTTON_RECT = pygame.Rect(360, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
BELL_BUTTON_RECT = pygame.Rect(360, 10, BUTTON_WIDTH, BUTTON_HEIGHT)

# Font for button texts
FONT = pygame.font.Font(None, 36)

# Simulation variables
simulation_count = 0
simulation_data = {
    "a_star": {"times": [], "path_lengths": [], "nodes": []},
    "bfs": {"times": [], "path_lengths": [], "nodes": []},
    "dfs": {"times": [], "path_lengths": [], "nodes": []},
    "r_dfs": {"times": [], "path_lengths": [], "nodes": []},
    "bell_ford": {"times": [], "path_lengths": [], "nodes": []}
}

ALGORITHMS = [ALGORITHM_ASTAR, ALGORITHM_BFS, ALGORITHM_BRUTE, ALGORITHM_BELLMAN_FORD]
ITERATIONS_PER_ALGO = MAX_SIMULATION_COUNT

# Main game loop variables
running = True
moving = False
target_pos = None
ball_x, ball_y = 0, 0  # Initial position
current_algorithm = ALGORITHM_ASTAR  # Default algorithm
path = []  # Ensure path is global or passed correctly

obstacles = []
def get_obstacles(ball_x = 0, ball_y = 0):
    obstacles.clear()
    while len(obstacles) < NUM_OBSTACLES:
        obstacle_col = random.randint(0, COLS - 1)
        obstacle_row = random.randint(0, ROWS - 1)
        obstacle_pos = (obstacle_col * CELL_WIDTH, obstacle_row * CELL_HEIGHT)
        
        # Ensure obstacle is not placed at the ball's position
        if obstacle_pos != (ball_x, ball_y) and obstacle_pos not in obstacles:
            obstacles.append(obstacle_pos)

    return obstacles

obstacles = get_obstacles(ball_x, ball_y)


# Function to draw obstacles
def draw_obstacles(obstacle_list):
    for obstacle_x, obstacle_y in obstacle_list:
        obstacle_rect = pygame.Rect(obstacle_x, obstacle_y, CELL_WIDTH, CELL_HEIGHT)
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle_rect)

# Function to draw the ball
def draw_ball(x, y):
    pygame.draw.circle(screen, BALL_COLOR, (x + CELL_WIDTH // 2, y + CELL_HEIGHT // 2), min(CELL_WIDTH, CELL_HEIGHT) // 3)

# Function to draw dotted lines
def draw_dotted_line(screen, start_pos, end_pos, color, dot_spacing=100):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dx = x2 - x1
    dy = y2 - y1
    distance = max(abs(dx), abs(dy))
    if distance != 0:
        step_x = dx / distance
        step_y = dy / distance
    else:
        step_x, step_y = 0, 0
    for i in range(0, int(distance), dot_spacing):
        dot_x = int(x1 + step_x * i)
        dot_y = int(y1 + step_y * i)
        pygame.draw.circle(screen, color, (dot_x, dot_y), 1)

# Function to draw algorithm buttons
def draw_buttons():
    astar_color = (0, 255, 0) if current_algorithm == ALGORITHM_ASTAR else (150, 150, 150)
    pygame.draw.rect(screen, astar_color, ASTAR_BUTTON_RECT)
    astar_text = FONT.render("A*", True, (0, 0, 0))
    screen.blit(astar_text, (ASTAR_BUTTON_RECT.x + 30, ASTAR_BUTTON_RECT.y + 5))
    
    bfs_color = (0, 255, 0) if current_algorithm == ALGORITHM_BFS else (150, 150, 150)
    pygame.draw.rect(screen, bfs_color, BFS_BUTTON_RECT)
    bfs_text = FONT.render("BFS", True, (0, 0, 0))
    screen.blit(bfs_text, (BFS_BUTTON_RECT.x + 20, BFS_BUTTON_RECT.y + 5))

    dfs_color = (0, 255, 0) if current_algorithm == ALGORITHM_BRUTE else (150, 150, 150)
    pygame.draw.rect(screen, dfs_color, DFS_BUTTON_RECT)
    dfs_text = FONT.render("DFS", True, (0, 0, 0))
    screen.blit(dfs_text, (DFS_BUTTON_RECT.x + 20, DFS_BUTTON_RECT.y + 5))

    # r_dfs_color = (0, 255, 0) if current_algorithm == ALGORITHM_RANDOM_BRUTE else (150, 150, 150)
    # pygame.draw.rect(screen, r_dfs_color, RANDOM_DFS_BUTTON_RECT)
    # r_dfs_text = FONT.render("R-DFS", True, (0, 0, 0))
    # screen.blit(r_dfs_text, (RANDOM_DFS_BUTTON_RECT.x + 20, RANDOM_DFS_BUTTON_RECT.y + 5))

    b_ford_color = (0, 255, 0) if current_algorithm == ALGORITHM_BELLMAN_FORD else (150, 150, 150)
    pygame.draw.rect(screen, b_ford_color, BELL_BUTTON_RECT)
    b_ford_text = FONT.render("BELL", True, (0, 0, 0))
    screen.blit(b_ford_text, (BELL_BUTTON_RECT.x + 20, BELL_BUTTON_RECT.y + 5))

# Function to randomize simulation parameters
def randomize_params():
    global WIDTH, HEIGHT, ROWS, COLS, CELL_WIDTH, CELL_HEIGHT, ball_x, ball_y, NUM_OBSTACLES
    
    # Randomize screen size and grid parameters
    random_width = random.randint(500, 1000)
    random_height = random.randint(200, 600)
    random_rows = random.randint(10, 50)
    random_cols = random.randint(10, 50)
    
    total_nodes = random_cols * random_rows
    # NUM_OBSTACLES = random.randint(int(0.2 * total_nodes), int(0.6 * total_nodes) )
    
    # Update global variables
    WIDTH, HEIGHT = random_width, random_height
    ROWS, COLS = random_rows, random_cols
    
    # Calculate new cell size
    CELL_WIDTH = WIDTH // COLS
    CELL_HEIGHT = HEIGHT // ROWS
    
    # Reinitialize the ball's position
    ball_x = 0
    ball_y = random.randint(0, ROWS - 1) * CELL_HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT))


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
        # Update total_obstacles before recalculation
        if len(obstacle_list) > total_obstacles:
            total_obstacles = len(obstacle_list)
        
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
            return cx, cy, False  # No path, stay put
    
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
        if abs(cy - next_pos[1]) < ball_speed:
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
    
    def dfs_random(current, path_so_far, max_attempts=10000):
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


# Main game loop
clock = pygame.time.Clock()
# IS_SIMULATION = False

while running:
    if IS_SIMULATION and simulation_count < MAX_SIMULATION_COUNT:
        algo_index = simulation_count % len(ALGORITHMS)
        current_algorithm = ALGORITHMS[algo_index]

        # Simulation mode
        if not moving:
            obstacles = get_obstacles(ball_x, ball_y)  # Static base
            trail.clear()
            lines.clear()
            markers.clear()
            target_pos = (random.randint(COLS//2, COLS-1) * CELL_WIDTH,
                          random.randint(ROWS//2, ROWS-1) * CELL_HEIGHT)
            markers.append(target_pos)
            moving = True
            path = []
            start_time = time.time()

        
        # Move ball and add random obstacles
        old_x, old_y = ball_x, ball_y
        ball_x, ball_y, movable = go_to_target((ball_x, ball_y), target_pos, obstacles, current_algorithm)
        if (ball_x, ball_y) != (old_x, old_y):
            trail.append((ball_x + CELL_WIDTH // 2, ball_y + CELL_HEIGHT // 2))

        if (ball_x, ball_y) == target_pos or not movable:
            moving = False
            end_time = time.time()
            sim_time = end_time - start_time
            path_len = len(trail)
            if path_len > 0:
                simulation_count += 1
                simulation_data[current_algorithm]["times"].append(sim_time)
                simulation_data[current_algorithm]["path_lengths"].append(path_len)
                simulation_data[current_algorithm]["nodes"].append(ROWS * COLS)
            print(f"Iteration {simulation_count}: {current_algorithm}, Time: {sim_time:.3f}s, Path Length: {path_len}")
            randomize_params()


        if simulation_count >= MAX_SIMULATION_COUNT:
            plot_simulation_results(simulation_data=simulation_data)
            IS_SIMULATION = False  # Exit simulation mode
        

    else:
        # Interactive mode
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if event.button == 1:
                    col = mouse_pos[0] // CELL_WIDTH
                    row = mouse_pos[1] // CELL_HEIGHT
                    obstacle_pos = (col * CELL_WIDTH, row * CELL_HEIGHT)
                    if obstacle_pos == target_pos:
                        continue
                    if ASTAR_BUTTON_RECT.collidepoint(mouse_pos):
                        current_algorithm = ALGORITHM_ASTAR
                        path = []
                    elif BFS_BUTTON_RECT.collidepoint(mouse_pos):
                        current_algorithm = ALGORITHM_BFS
                        path = []
                    elif DFS_BUTTON_RECT.collidepoint(mouse_pos):
                        current_algorithm = ALGORITHM_BRUTE
                        path = []
                    # elif RANDOM_DFS_BUTTON_RECT.collidepoint(mouse_pos):
                    #     current_algorithm = ALGORITHM_RANDOM_BRUTE
                    #     path = []
                    elif BELL_BUTTON_RECT.collidepoint(mouse_pos):
                        current_algorithm = ALGORITHM_BELLMAN_FORD
                        path = []
                    elif not moving:
                        mouse_x, mouse_y = mouse_pos
                        col = mouse_x // CELL_WIDTH
                        row = mouse_y // CELL_HEIGHT
                        markers.clear()
                        markers.append((col * CELL_WIDTH, row * CELL_HEIGHT))
                        target_pos = markers[-1]
                        moving = True
                        trail.clear()
                elif event.button == 3:
                    col = mouse_pos[0] // CELL_WIDTH
                    row = mouse_pos[1] // CELL_HEIGHT
                    obstacle_pos = (col * CELL_WIDTH, row * CELL_HEIGHT)
                    if obstacle_pos == target_pos:
                        continue
                    if obstacle_pos in obstacles:
                        obstacles.remove(obstacle_pos)
                    elif obstacle_pos != (ball_x, ball_y):
                        obstacles.append(obstacle_pos)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    lines.clear()
                    trail.clear()
                elif event.key == pygame.K_s:  # Press 'S' to start simulation
                    IS_SIMULATION = True
                    simulation_count = 0

        if moving:
            old_x, old_y = ball_x, ball_y
            ball_x, ball_y, movable = go_to_target((ball_x, ball_y), target_pos, obstacles, current_algorithm)
            if not movable:
                moving = False
            if (ball_x, ball_y) != (old_x, old_y):
                trail.append((ball_x + CELL_WIDTH // 2, ball_y + CELL_HEIGHT // 2))
            if (ball_x, ball_y) == target_pos:
                lines.append(((ball_x + CELL_WIDTH // 2, ball_y + CELL_HEIGHT // 2),
                              (target_pos[0] + CELL_WIDTH // 2, target_pos[1] + CELL_HEIGHT // 2)))
                moving = False

    # Rendering
    screen.fill((255, 255, 255))
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

    for marker_x, marker_y in markers:
        marker_rect = pygame.Rect(marker_x, marker_y, CELL_WIDTH, CELL_HEIGHT)
        pygame.draw.rect(screen, MARKER_COLOR, marker_rect)
    draw_ball(ball_x, ball_y)
    draw_obstacles(obstacles)
    draw_buttons()
    for i in range(1, len(trail)):
        draw_dotted_line(screen, trail[i-1], trail[i], TRAIL_COLOR)
    
    pygame.display.flip()
    clock.tick(60)  # Cap at 60 FPS

pygame.quit()
sys.exit()
