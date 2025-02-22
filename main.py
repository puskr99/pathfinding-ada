from ball import go_to_target
from obstacles import get_obstacles
from attributes import *

import pygame
import sys

pygame.init()


# Create the Pygame screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Grid with Ball")

# List to store marker positions
markers = []

# List to store all dotted lines
lines = []

# List to store the ball's trail
trail = []

## buttons
ASTAR_BUTTON_RECT = pygame.Rect(10, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
BFS_BUTTON_RECT = pygame.Rect(1200, 1000, BUTTON_WIDTH, BUTTON_HEIGHT)
DFS_BUTTON_RECT = pygame.Rect(120, 10, BUTTON_WIDTH, BUTTON_HEIGHT)
RANDOM_DFS_BUTTON_RECT = pygame.Rect(240, 10, BUTTON_WIDTH, BUTTON_HEIGHT)


FONT = pygame.font.Font(None, 36)

# Main game loop variables
running = True
moving = False  # Whether the ball is currently moving
target_pos = None  # Target position for the ball

obstacles = get_obstacles()
def draw_obstacles(obstacle_list):
    for obstacle_x, obstacle_y in obstacle_list:  
        obstacle_rect = pygame.Rect(obstacle_x, obstacle_y, CELL_WIDTH, CELL_HEIGHT)
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle_rect)


# Draw the ball
def draw_ball(x, y):
    pygame.draw.circle(screen, BALL_COLOR, (x + CELL_WIDTH // 2, y + CELL_HEIGHT // 2), min(CELL_WIDTH, CELL_HEIGHT) // 3)


# Function to draw a dotted line
def draw_dotted_line(screen, start_pos, end_pos, color, dot_spacing=100):
    x1, y1 = start_pos
    x2, y2 = end_pos
    # Calculate the distance and direction vector
    dx = x2 - x1
    dy = y2 - y1
    distance = max(abs(dx), abs(dy))  # Use the larger dimension for spacing
    # Normalize the direction vector
    if distance != 0:
        step_x = dx / distance
        step_y = dy / distance
    else:
        step_x, step_y = 0, 0
    # Draw dots along the line
    for i in range(0, int(distance), dot_spacing):
        dot_x = int(x1 + step_x * i)
        dot_y = int(y1 + step_y * i)
        pygame.draw.circle(screen, color, (dot_x, dot_y), 1)  # Draw a small circle as a dot

def draw_buttons():
    # A* button
    astar_color = (0, 255, 0) if current_algorithm == ALGORITHM_ASTAR else (150, 150, 150)
    pygame.draw.rect(screen, astar_color, ASTAR_BUTTON_RECT)
    astar_text = FONT.render("A*", True, (0, 0, 0))
    screen.blit(astar_text, (ASTAR_BUTTON_RECT.x + 30, ASTAR_BUTTON_RECT.y + 5))
    
    # BFS button
    # bfs_color = (0, 255, 0) if current_algorithm == ALGORITHM_BFS else (150, 150, 150)
    # pygame.draw.rect(screen, bfs_color, BFS_BUTTON_RECT)
    # bfs_text = FONT.render("BFS", True, (0, 0, 0))
    # screen.blit(bfs_text, (BFS_BUTTON_RECT.x + 20, BFS_BUTTON_RECT.y + 5))

    # DFS
    dfs = (0, 255, 0) if current_algorithm == ALGORITHM_BRUTE else (150, 150, 150)
    pygame.draw.rect(screen, dfs, DFS_BUTTON_RECT)
    dfs_text = FONT.render("DFS", True, (0, 0, 0))
    screen.blit(dfs_text, (DFS_BUTTON_RECT.x + 20, DFS_BUTTON_RECT.y + 5))

    # Random DFS
    r_dfs = (0, 255, 0) if current_algorithm == ALGORITHM_RANDOM_BRUTE else (150, 150, 150)
    pygame.draw.rect(screen, r_dfs, RANDOM_DFS_BUTTON_RECT)
    r_dfs_text = FONT.render("R-DFS", True, (0, 0, 0))
    screen.blit(r_dfs_text, (RANDOM_DFS_BUTTON_RECT.x + 20, RANDOM_DFS_BUTTON_RECT.y + 5))


# Main game loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            if ASTAR_BUTTON_RECT.collidepoint(mouse_pos):
                current_algorithm = ALGORITHM_ASTAR
                path = []  # Reset path when changing algorithm
            elif BFS_BUTTON_RECT.collidepoint(mouse_pos):
                current_algorithm = ALGORITHM_BFS
                path = []  # Reset path when changing algorithm
            elif DFS_BUTTON_RECT.collidepoint(mouse_pos):
                current_algorithm = ALGORITHM_BRUTE
                path = []
            elif RANDOM_DFS_BUTTON_RECT.collidepoint(mouse_pos):
                current_algorithm = ALGORITHM_RANDOM_BRUTE
                path = []
            elif not moving:  # Only allow new movement if the ball isn't already moving
                # Get the mouse position and calculate grid cell
                mouse_x, mouse_y = event.pos
                col = mouse_x // CELL_WIDTH
                row = mouse_y // CELL_HEIGHT
                markers.clear()
                markers.append((col * CELL_WIDTH, row * CELL_HEIGHT))  # Store marker position in pixels
                target_pos = markers[-1]  # Set the target position
                moving = True  # Start moving the ball
                trail.clear() # Clear the trail on new movement.
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:  # Press 'E' to erase all lines
                lines.clear()  # Clear all stored lines
                trail.clear() # Clear the trail as well.

    # Move the ball if there's a target position
    if moving:
        old_x, old_y = ball_x, ball_y
        ball_x, ball_y = go_to_target((ball_x, ball_y), target_pos, obstacle_list=obstacles, algorithm=current_algorithm)
        if (ball_x, ball_y) != (old_x, old_y):
            trail.append((ball_x + CELL_WIDTH // 2, ball_y + CELL_HEIGHT // 2))
        # else:
        #     target_pos = ball_x, ball_y

        if (ball_x, ball_y) == target_pos:
            # When the ball reaches the target, store the line
            lines.append(((ball_x + CELL_WIDTH // 2, ball_y + CELL_HEIGHT // 2),
                          (target_pos[0] + CELL_WIDTH // 2, target_pos[1] + CELL_HEIGHT // 2)))
            moving = False  # Stop moving when the ball reaches the target

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the grid
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

    # Draw the markers
    for marker_x, marker_y in markers:
        marker_rect = pygame.Rect(marker_x, marker_y, CELL_WIDTH, CELL_HEIGHT)
        pygame.draw.rect(screen, MARKER_COLOR, marker_rect)

    # Draw the ball
    draw_ball(ball_x, ball_y)
    draw_obstacles(obstacle_list= obstacles)
    draw_buttons()

    # Draw the ball's trail
    for i in range(1, len(trail)):
        draw_dotted_line(screen, trail[i-1], trail[i], TRAIL_COLOR)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()