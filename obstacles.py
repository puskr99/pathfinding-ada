import random
from attributes import COLS, ROWS, CELL_HEIGHT, CELL_WIDTH, ball_x, ball_y, NUM_OBSTACLES

obstacles = []

def get_obstacles():
    obstacles.clear()
    while len(obstacles) < NUM_OBSTACLES:
        obstacle_col = random.randint(0, COLS - 1)
        obstacle_row = random.randint(0, ROWS - 1)
        obstacle_pos = (obstacle_col * CELL_WIDTH, obstacle_row * CELL_HEIGHT)
        if obstacle_pos != (ball_x, ball_y) and obstacle_pos not in obstacles:
            obstacles.append(obstacle_pos)

    return obstacles