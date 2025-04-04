import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_simulation_results(simulation_data, output_csv_path="simulation_summary.csv"):
    # Filter algorithms with data to avoid ZeroDivisionError
    valid_algorithms = [algo for algo in simulation_data.keys()
                        if simulation_data[algo]["times"] and simulation_data[algo]["path_lengths"]]

    if not valid_algorithms:
        print("No simulation data to plot.")
        return

    # Check if CSV file exists
    csv_exists = os.path.exists(output_csv_path)

    # Open CSV file and write (or append) data
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header if file does not exist
        if not csv_exists:
            writer.writerow(["algorithm", "time", "path_length", "nodes", "obstacles_percent"])

        # Write data rows
        for algo in valid_algorithms:
            times = simulation_data[algo]["times"]
            lengths = simulation_data[algo]["path_lengths"]
            nodes = simulation_data[algo]["nodes"]
            obstacles_percent = simulation_data[algo]["obstacles_percent"]

            for t, l, n, op in zip(times, lengths, nodes, obstacles_percent):
                writer.writerow([algo, t, l, n, op])

    print(f"Simulation data saved to {os.path.abspath(output_csv_path)}")

def plot_summary_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"No data found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV file is empty.")
        return

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Unique obstacle percentages
    obstacle_percentages = df['obstacles_percent'].unique()

    # Loop through each obstacle percentage and plot separately
    for op in obstacle_percentages:
        # Filter data for the current obstacle percentage
        op_df = df[df['obstacles_percent'] == op]

        # Group by algorithm and calculate stats
        grouped = op_df.groupby("algorithm")
        avg_times = grouped["time"].mean()
        std_times = grouped["time"].std()
        avg_lengths = grouped["path_length"].mean()
        std_lengths = grouped["path_length"].std()

        valid_algorithms = avg_times.index.tolist()

        # Subplot setup
        plt.figure(figsize=(15, 5))

        # Subplot 1: Avg Computation Time
        plt.subplot(1, 3, 1)
        plt.bar(valid_algorithms, avg_times, yerr=std_times, color='skyblue', capsize=5)
        plt.title(f"Avg Computation Time (Obstacles {op}%)")
        plt.ylabel("Time (s)")
        plt.xticks(rotation=45)

        # Subplot 2: Avg Path Length
        plt.subplot(1, 3, 2)
        plt.bar(valid_algorithms, avg_lengths, yerr=std_lengths, color='lightgreen', capsize=5)
        plt.title(f"Avg Path Length (Obstacles {op}%)")
        plt.ylabel("Steps")
        plt.xticks(rotation=45)

        # Subplot 3: Smoothed Time vs Nodes (log scale)
        plt.subplot(1, 3, 3)
        for algo in valid_algorithms:
            sub_df = op_df[op_df["algorithm"] == algo]
            sub_df = sub_df.sort_values("nodes")
            x = sub_df["nodes"].values
            y = sub_df["time"].values

            # Smooth data
            nodes_smooth, times_smooth = smooth_data(x, y)
            plt.plot(nodes_smooth, times_smooth, label=algo, linewidth=2)

        plt.title(f"Computation Time vs. Nodes (Obstacles {op * 100}%)")
        plt.xlabel("Number of Nodes (rows * cols)")
        plt.ylabel("Time (s)")
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()

        # Adjust layout for clarity
        plt.tight_layout()
        # Show plot for this obstacle percentage
        plt.show()

def smooth_data(x, y, frac=0.2):
    """
    Smooths the data using LOWESS (locally weighted regression).
    
    Parameters:
        x: 1D array-like, independent variable (e.g., nodes)
        y: 1D array-like, dependent variable (e.g., time)
        frac: float, between 0 and 1 — amount of smoothing (higher = smoother)
    
    Returns:
        Tuple of (smoothed_x, smoothed_y)
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 3:
        return x, y  # not enough data to smooth

    # Sort by x to ensure proper line
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Apply LOWESS
    smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)

    smoothed_x = smoothed[:, 0]
    smoothed_y = smoothed[:, 1]

    return smoothed_x, smoothed_y

# Example usage
# plot_summary_from_csv("simulation_summary.csv")
