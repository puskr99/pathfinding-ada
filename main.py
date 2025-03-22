from ball import go_to_target
from obstacles import get_obstacles
from attributes import *
from plot import plot_simulation_results

import pygame
import sys
import random
import time
import matplotlib.pyplot as plt

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pathfinding with Obstacles")

markers = []
lines = []
trail = []

# Buttons
ASTAR_BUTTON_RECT = pygame.Rect(10, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
BFS_BUTTON_RECT = pygame.Rect(120, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
DFS_BUTTON_RECT = pygame.Rect(240, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
RANDOM_DFS_BUTTON_RECT = pygame.Rect(360, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
BELL_BUTTON_RECT = pygame.Rect(480, 10, BUTTON_WIDTH, BUTTON_HEIGHT)

FONT = pygame.font.Font(None, 36)

# Simulation variables
simulation_count = 0
simulation_data = {"a_star": {"times": [], "path_lengths": []},
                   "bfs": {"times": [], "path_lengths": []},
                   "dfs": {"times": [], "path_lengths": []},
                   "r_dfs": {"times": [], "path_lengths": []},
                   "bell_ford": {"times": [], "path_lengths": []}}
ALGORITHMS = [ALGORITHM_ASTAR, ALGORITHM_BFS, ALGORITHM_BELLMAN_FORD]
ITERATIONS_PER_ALGO = MAX_SIMULATION_COUNT

# Main game loop variables
running = True
moving = False
target_pos = None
ball_x, ball_y = 0, 0  # Initial position
obstacles = get_obstacles()
current_algorithm = ALGORITHM_ASTAR  # Default
path = []  # Ensure path is global or passed correctly

def draw_obstacles(obstacle_list):
    for obstacle_x, obstacle_y in obstacle_list:
        obstacle_rect = pygame.Rect(obstacle_x, obstacle_y, CELL_WIDTH, CELL_HEIGHT)
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle_rect)

def draw_ball(x, y):
    pygame.draw.circle(screen, BALL_COLOR, (x + CELL_WIDTH // 2, y + CELL_HEIGHT // 2), min(CELL_WIDTH, CELL_HEIGHT) // 3)

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

def draw_buttons():
    astar_color = (0, 255, 0) if current_algorithm == ALGORITHM_ASTAR else (150, 150, 150)
    pygame.draw.rect(screen, astar_color, ASTAR_BUTTON_RECT)
    astar_text = FONT.render("A*", True, (0, 0, 0))
    screen.blit(astar_text, (ASTAR_BUTTON_RECT.x + 30, ASTAR_BUTTON_RECT.y + 5))
    
    bfs_color = (0, 255, 0) if current_algorithm == ALGORITHM_BFS else (150, 150, 150)
    pygame.draw.rect(screen, bfs_color, BFS_BUTTON_RECT)
    bfs_text = FONT.render("BFS", True, (0, 0, 0))
    screen.blit(bfs_text, (BFS_BUTTON_RECT.x + 20, BFS_BUTTON_RECT.y + 5))

    dfs = (0, 255, 0) if current_algorithm == ALGORITHM_BRUTE else (150, 150, 150)
    pygame.draw.rect(screen, dfs, DFS_BUTTON_RECT)
    dfs_text = FONT.render("DFS", True, (0, 0, 0))
    screen.blit(dfs_text, (DFS_BUTTON_RECT.x + 20, DFS_BUTTON_RECT.y + 5))

    r_dfs = (0, 255, 0) if current_algorithm == ALGORITHM_RANDOM_BRUTE else (150, 150, 150)
    pygame.draw.rect(screen, r_dfs, RANDOM_DFS_BUTTON_RECT)
    r_dfs_text = FONT.render("R-DFS", True, (0, 0, 0))
    screen.blit(r_dfs_text, (RANDOM_DFS_BUTTON_RECT.x + 20, RANDOM_DFS_BUTTON_RECT.y + 5))

    b_ford = (0, 255, 0) if current_algorithm == ALGORITHM_BELLMAN_FORD else (150, 150, 150)
    pygame.draw.rect(screen, b_ford, BELL_BUTTON_RECT)
    b_ford_text = FONT.render("BELL", True, (0, 0, 0))
    screen.blit(b_ford_text, (BELL_BUTTON_RECT.x + 20, BELL_BUTTON_RECT.y + 5))

complexity_data = {
    "a_star": {"sizes": [], "times": []},
    "bfs": {"sizes": [], "times": []},
    "r_dfs": {"sizes": [], "times": []},
    "dfs": {"sizes": [], "times": []},
    "bell_ford": {"sizes": [], "times": []},
}

def run_complexity_experiment():
    grid_sizes = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]  # Varying grid sizes
    num_trials = 5  # Average over 5 runs per size for stability

    for rows, cols in grid_sizes:
        # Temporarily override attributes for this grid size
        global ROWS, COLS, WIDTH, HEIGHT
        ROWS, COLS = rows, cols
        CELL_WIDTH, CELL_HEIGHT = WIDTH // COLS, HEIGHT // ROWS
        WIDTH, HEIGHT = COLS * CELL_WIDTH, ROWS * CELL_HEIGHT
        
        # Reset screen size (optional, for visualization during experiment)
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        
        # Fixed start and target for consistency
        start_pos = (0, 0)
        target_pos = ((cols-1) * CELL_WIDTH, (rows-1) * CELL_HEIGHT)
        obstacles = [(CELL_WIDTH * i, CELL_HEIGHT * i) for i in range(min(rows, cols) // 2)]  # Diagonal obstacles
        
        for algo in ALGORITHMS:
            total_time = 0
            for _ in range(num_trials):
                start_time = time.time()
                go_to_target(start_pos, target_pos, obstacles, algo)  # Run algorithm
                total_time += time.time() - start_time
            avg_time = total_time / num_trials
            complexity_data[algo]["sizes"].append(rows * cols)  # Input size = number of nodes
            complexity_data[algo]["times"].append(avg_time)
            print(f"Grid {rows}x{cols}, {algo}: Avg Time = {avg_time:.4f}s")


# Main game loop
clock = pygame.time.Clock()
while running:
    if IS_SIMULATION and simulation_count < MAX_SIMULATION_COUNT:

        algo_index = simulation_count % len(ALGORITHMS)
        current_algorithm = ALGORITHMS[algo_index]

        # Simulation mode
        if not moving:
            # Reset for new iteration
            # ball_x, ball_y = 0, 0
            obstacles = get_obstacles()  # Static base
            trail.clear()
            lines.clear()
            markers.clear()
            # Random target in bottom-right quadrant
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
        # if random.random() < 0.01:  # 5% chance per frame to add obstacle
        #     new_obstacle = (random.randint(COLS//2, COLS-1) * CELL_WIDTH,
        #                   random.randint(ROWS//2, ROWS-1) * CELL_HEIGHT)
        #     if new_obstacle != (ball_x, ball_y) and new_obstacle != target_pos and new_obstacle not in obstacles:
        #         obstacles.append(new_obstacle)
        if (ball_x, ball_y) == target_pos or not movable:
            moving = False
            simulation_count += 1
            end_time = time.time()
            sim_time = end_time - start_time
            path_len = len(trail)
            simulation_data[current_algorithm]["times"].append(sim_time)
            simulation_data[current_algorithm]["path_lengths"].append(path_len)
            print(f"Iteration {simulation_count}: {current_algorithm}, Time: {sim_time:.3f}s, Path Length: {path_len}")

        if simulation_count >= MAX_SIMULATION_COUNT:
            # run_complexity_experiment()
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
                    elif RANDOM_DFS_BUTTON_RECT.collidepoint(mouse_pos):
                        current_algorithm = ALGORITHM_RANDOM_BRUTE
                        path = []
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