import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

def save_simulation_data(simulation_data, output_csv_path="simulation_summary.csv"):
    """
    Saves simulation data to a CSV file.

    Parameters:
        simulation_data (dict): Nested dictionary containing simulation results.
        output_csv_path (str): Path to the CSV file.
    """
    valid_algorithms = [
        algo for algo in simulation_data.keys()
        if simulation_data[algo]["times"] and simulation_data[algo]["path_lengths"]
    ]

    if not valid_algorithms:
        print("No valid simulation data to save.")
        return

    csv_exists = os.path.exists(output_csv_path)

    # Write or append simulation data
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["algorithm", "time", "path_length", "nodes", "obstacles_percent", "memory_mb"])

        for algo in valid_algorithms:
            algo_data = simulation_data[algo]
            memory_mb = algo_data.get("memory_mb", [0] * len(algo_data["times"]))
            for t, l, n, op, m in zip(
                algo_data["times"], algo_data["path_lengths"],
                algo_data["nodes"], algo_data["obstacles_percent"], memory_mb
            ):
                writer.writerow([algo, t, l, n, op, m])

    print(f"Simulation data saved to {os.path.abspath(output_csv_path)}")

def load_and_prepare_csv(csv_path):
    """
    Loads and cleans simulation data from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned and prepared data.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path).drop_duplicates()
    if df.empty:
        print("CSV file is empty.")
        return df

    if "memory_mb" not in df.columns:
        df["memory_mb"] = 0

    # Remove rows with invalid data
    numeric_columns = ["time", "path_length", "nodes", "obstacles_percent", "memory_mb"]
    df = df.dropna(subset=numeric_columns)
    df = df[np.all(np.isfinite(df[numeric_columns]), axis=1)]
    return df

def plot_data(df, obstacle_percent):
    """
    Plots simulation data for a given obstacle percentage.

    Parameters:
        df (pd.DataFrame): Filtered simulation data.
        obstacle_percent (float): Obstacle percentage to plot.
    """
    grouped = df.groupby("algorithm")
    algorithms = grouped.groups.keys()

    avg_metrics = grouped[["time", "path_length", "memory_mb"]].mean()
    std_metrics = grouped[["time", "path_length", "memory_mb"]].std()

    plt.figure(figsize=(15, 10))

    # Average Metrics
    for idx, (metric, title, ylabel, color) in enumerate(
        [("path_length", "Avg Path Length", "Steps", "lightgreen"),
         ("time", "Avg Computation Time", "Time (s)", "skyblue"),
         ("memory_mb", "Avg Memory Usage", "Memory (MB)", "lightcoral")]
    ):
        plt.subplot(2, 3, idx + 1)
        plt.bar(algorithms, avg_metrics[metric], color=color, capsize=5)
        plt.title(f"{title} (Obstacles {obstacle_percent * 100:.0f}%)")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")

    # Nodes vs Metrics (Log Scale)
    for idx, (metric, ylabel, title) in enumerate(
        [("time", "Time (s)", "Computation Time vs. Nodes"),
         ("memory_mb", "Memory (MB)", "Memory Usage vs. Nodes")]
    ):
        plt.subplot(2, 3, idx + 4)
        for algo in algorithms:
            algo_df = df[df["algorithm"] == algo].sort_values("nodes")
            nodes, metric_values = algo_df["nodes"], algo_df[metric]
            smoothed_nodes, smoothed_values = smooth_data(nodes, metric_values)
            plt.plot(smoothed_nodes, smoothed_values, label=algo, linewidth=2)
        plt.title(f"{title} (Obstacles {obstacle_percent * 100:.0f}%)")
        plt.xlabel("Number of Nodes (rows * cols)")
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.grid(which="both", linestyle="--", linewidth=0.5)
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_summary_from_csv(csv_path):
    """
    Generates plots from simulation data stored in a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.
    """
    df = load_and_prepare_csv(csv_path)
    if df.empty:
        return

    obstacle_percentages = sorted(df["obstacles_percent"].unique())
    for op in obstacle_percentages:
        plot_data(df[df["obstacles_percent"] == op], op)

import numpy as np
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter

def smooth_data(x, y, method="lowess", frac=0.3, spline_smooth_factor=None, window_size=5, clamp_range=None, remove_outliers=True):
    """
    Smooths data using LOWESS, cubic spline, or Savitzky-Golay filter with robust preprocessing.

    Parameters:
        x (array-like): Independent variable (e.g., nodes).
        y (array-like): Dependent variable (e.g., path_length, time, memory_mb).
        method (str): Smoothing method ('lowess', 'spline', 'savgol').
        frac (float): Smoothing factor for LOWESS (0 < frac <= 1, default=0.3).
        spline_smooth_factor (float, optional): Smoothness for spline (lower = smoother, default auto).
        window_size (int): Window size for Savitzky-Golay filter (must be odd, default=5).
        clamp_range (tuple, optional): (min, max) values to clamp y.
        remove_outliers (bool): Remove outliers using IQR method before smoothing (default=True).

    Returns:
        Tuple of (smoothed_x, smoothed_y) arrays.
    """
    # Convert inputs to numpy arrays
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)

    # Handle edge cases
    if len(x) < 3 or len(np.unique(x)) < 3:
        print(f"Insufficient data for smoothing: len(x)={len(x)}, unique_x={len(np.unique(x))}")
        return x, y

    # Remove NaN and infinite values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid_mask):
        print("All data points are invalid (NaN or infinite).")
        return x, y
    x, y = x[valid_mask], y[valid_mask]

    # Remove outliers using IQR method
    if remove_outliers:
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        outlier_mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
        if not np.any(outlier_mask):
            print("No valid data after outlier removal.")
            return x, y
        x, y = x[outlier_mask], y[outlier_mask]

    # Sort data by x
    sorted_indices = np.argsort(x)
    x_sorted, y_sorted = x[sorted_indices], y[sorted_indices]

    # Remove duplicate x values (keep first occurrence)
    unique_mask = np.concatenate(([True], np.diff(x_sorted) != 0))
    if not np.any(unique_mask):
        print("No unique x values after preprocessing.")
        return x_sorted, y_sorted
    x_sorted, y_sorted = x_sorted[unique_mask], y_sorted[unique_mask]

    smoothed_y = None

    if method == "lowess":
        # LOWESS smoothing with optimized frac
        try:
            frac = min(max(frac, 0.1), 1.0)  # Ensure frac is valid
            smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
            smoothed_y = smoothed[:, 1]
        except Exception as e:
            print(f"LOWESS smoothing failed: {e}. Falling back to raw data.")
            smoothed_y = y_sorted

    elif method == "spline":
        # Cubic spline with automatic smoothness
        try:
            if spline_smooth_factor is None:
                # Auto-calculate smoothness based on data size
                spline_smooth_factor = len(x_sorted) * np.var(y_sorted)
            spline = UnivariateSpline(x_sorted, y_sorted, s=spline_smooth_factor, k=3)
            smoothed_y = spline(x_sorted)
        except Exception as e:
            print(f"Spline smoothing failed: {e}. Falling back to raw data.")
            smoothed_y = y_sorted

    elif method == "savgol":
        # Savitzky-Golay filter for smooth, robust smoothing
        try:
            window_size = min(max(window_size, 3), len(x_sorted))  # Ensure valid window
            if window_size % 2 == 0:
                window_size += 1  # Must be odd
            smoothed_y = savgol_filter(y_sorted, window_size, polyorder=2)
        except Exception as e:
            print(f"Savitzky-Golay smoothing failed: {e}. Falling back to raw data.")
            smoothed_y = y_sorted

    else:
        print(f"Invalid smoothing method: {method}. Using raw data.")
        smoothed_y = y_sorted

    # Apply clamping if specified
    if clamp_range:
        smoothed_y = np.clip(smoothed_y, clamp_range[0], clamp_range[1])

    # Ensure output is finite
    if np.any(~np.isfinite(smoothed_y)):
        print("Smoothing produced invalid values. Using raw data.")
        return x_sorted, y_sorted

    return x_sorted, smoothed_y



if __name__ == "__main__":
    plot_summary_from_csv("simulation_summary.csv")
