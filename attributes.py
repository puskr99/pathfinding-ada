# Screen dimensions
WIDTH, HEIGHT = 800, 600
ROWS, COLS = 20, 20  # Number of rows and columns in the grid
GRID_COLOR = (200, 200, 200)
BALL_COLOR = (6, 35, 48)#062330
MARKER_COLOR = (0, 255, 0)
LINE_COLOR = (0, 0, 255)
TRAIL_COLOR = (231, 231, 10) # Color of the trail line  
OBSTACLE_COLOR = (255, 0, 0)

# Calculate the size of each grid cell
CELL_WIDTH = WIDTH // COLS
CELL_HEIGHT = HEIGHT // ROWS

# Initialize the ball's position
ball_x = 0
ball_y = HEIGHT - CELL_HEIGHT

NUM_OBSTACLES = 50  # Number of obstacles

ball_speed = 2 # Speed of the ball

ALGORITHM_ASTAR = "a_star"
ALGORITHM_BFS = "bfs"
ALGORITHM_BRUTE = "dfs"
ALGORITHM_RANDOM_BRUTE = "r_dfs"
ALGORITHM_BELLMAN_FORD = "bell_ford"
current_algorithm = ALGORITHM_RANDOM_BRUTE  # Default to brute

# Button properties
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 40
