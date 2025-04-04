import numpy as np
import matplotlib.pyplot as plt
import csv
import os

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
            writer.writerow(["algorithm", "time", "path_length", "nodes"])

        # Write data rows
        for algo in valid_algorithms:
            times = simulation_data[algo]["times"]
            lengths = simulation_data[algo]["path_lengths"]
            nodes = simulation_data[algo]["nodes"]

            for t, l, n in zip(times, lengths, nodes):
                writer.writerow([algo, t, l, n])

    print(f"Simulation data saved to {os.path.abspath(output_csv_path)}")

    # Proceed with plotting summary from new data
    # plot_summary_from_csv(output_csv_path)


def plot_summary_from_csv(csv_path):
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"No data found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV file is empty.")
        return

    # Group by algorithm and calculate stats
    grouped = df.groupby("algorithm")
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
    plt.title("Average Computation Time")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45)

    # Subplot 2: Avg Path Length
    plt.subplot(1, 3, 2)
    plt.bar(valid_algorithms, avg_lengths, yerr=std_lengths, color='lightgreen', capsize=5)
    plt.title("Average Path Length")
    plt.ylabel("Steps")
    plt.xticks(rotation=45)

    # Subplot 3: Smoothed Time vs Nodes (log scale)
    plt.subplot(1, 3, 3)
    for algo in valid_algorithms:
        sub_df = df[df["algorithm"] == algo]
        sub_df = sub_df.sort_values("nodes")
        x = sub_df["nodes"].values
        y = sub_df["time"].values

        # Smooth data
        nodes_smooth, times_smooth = smooth_data(x, y)
        plt.plot(nodes_smooth, times_smooth, label=algo, linewidth=2)

    plt.title("Smoothed Computation Time vs. Nodes (Log Scale)")
    plt.xlabel("Number of Nodes (rows * cols)")
    plt.ylabel("Time (s)")
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.tight_layout()
    plt.show()


def smooth_data(x, y, window_size=5):
    if len(x) < window_size:
        return x, y  # Not enough data to smooth
    smoothed_x = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
    smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    return smoothed_x, smoothed_y

plot_summary_from_csv("simulation_summary.csv")