import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import os

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
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(x) < 3 or len(np.unique(x)) < 3:
        print(f"Insufficient data for smoothing: len(x)={len(x)}, unique_x={len(np.unique(x))}")
        return x, y
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid_mask):
        print("All data points are invalid (NaN or infinite).")
        return x, y
    x, y = x[valid_mask], y[valid_mask]
    if remove_outliers:
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        outlier_mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
        if not np.any(outlier_mask):
            print("No valid data after outlier removal.")
            return x, y
        x, y = x[outlier_mask], y[outlier_mask]
    sorted_indices = np.argsort(x)
    x_sorted, y_sorted = x[sorted_indices], y[sorted_indices]
    unique_mask = np.concatenate(([True], np.diff(x_sorted) != 0))
    if not np.any(unique_mask):
        print("No unique x values after preprocessing.")
        return x_sorted, y_sorted
    x_sorted, y_sorted = x_sorted[unique_mask], y_sorted[unique_mask]
    smoothed_y = None
    if method == "lowess":
        try:
            frac = min(max(frac, 0.1), 1.0)
            smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
            smoothed_y = smoothed[:, 1]
        except Exception as e:
            print(f"LOWESS smoothing failed: {e}. Falling back to raw data.")
            smoothed_y = y_sorted
    elif method == "spline":
        from scipy.interpolate import UnivariateSpline
        try:
            if spline_smooth_factor is None:
                spline_smooth_factor = len(x_sorted) * np.var(y_sorted)
            spline = UnivariateSpline(x_sorted, y_sorted, s=spline_smooth_factor, k=3)
            smoothed_y = spline(x_sorted)
        except Exception as e:
            print(f"Spline smoothing failed: {e}. Falling back to raw data.")
            smoothed_y = y_sorted
    elif method == "savgol":
        from scipy.signal import savgol_filter
        try:
            window_size = min(max(window_size, 3), len(x_sorted))
            if window_size % 2 == 0:
                window_size += 1
            smoothed_y = savgol_filter(y_sorted, window_size, polyorder=2)
        except Exception as e:
            print(f"Savitzky-Golay smoothing failed: {e}. Falling back to raw data.")
            smoothed_y = y_sorted
    else:
        print(f"Invalid smoothing method: {method}. Using raw data.")
        smoothed_y = y_sorted
    if clamp_range:
        smoothed_y = np.clip(smoothed_y, clamp_range[0], clamp_range[1])
    if np.any(~np.isfinite(smoothed_y)):
        print("Smoothing produced invalid values. Using raw data.")
        return x_sorted, y_sorted
    return x_sorted, smoothed_y

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
    numeric_columns = ["time", "path_length", "nodes", "obstacles_percent", "memory_mb"]
    df = df.dropna(subset=numeric_columns)
    df = df[np.all(np.isfinite(df[numeric_columns]), axis=1)]
    return df

def plot_metric_subplots(df, metric, title, ylabel, color, obstacle_percents, save_path, is_line_plot=False):
    """
    Plots a metric for multiple obstacle percentages in a 2x2 grid and saves the figure.

    Parameters:
        df (pd.DataFrame): Simulation data.
        metric (str): Metric to plot ('time', 'path_length', 'memory_mb').
        title (str): Plot title prefix.
        ylabel (str): Y-axis label.
        color (str): Color for bars or lines.
        obstacle_percents (list): List of obstacle percentages (e.g., [0.05, 0.1, 0.25, 0.4]).
        save_path (str): Path to save the figure.
        is_line_plot (bool): If True, plot nodes vs. metric; else, plot average metric bars.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, obst in enumerate(obstacle_percents):
        df_obst = df[df['obstacles_percent'] == obst]
        if df_obst.empty:
            print(f"No data for obstacle percentage {obst*100:.0f}%")
            continue
        grouped = df_obst.groupby('algorithm')
        algorithms = grouped.groups.keys()
        
        if is_line_plot:
            # Line plot: metric vs. nodes
            for algo in algorithms:
                algo_df = df_obst[df_obst['algorithm'] == algo].sort_values('nodes')
                nodes, metric_values = algo_df['nodes'], algo_df[metric]
                smoothed_nodes, smoothed_values = smooth_data(nodes, metric_values)
                axes[idx].plot(smoothed_nodes, smoothed_values, label=algo, linewidth=2)
            axes[idx].set_title(f"{title} (Obstacles {obst*100:.0f}%)")
            axes[idx].set_xlabel("Number of Nodes (rows * cols)")
            axes[idx].set_ylabel(ylabel)
            axes[idx].set_yscale('log')
            axes[idx].grid(which="both", linestyle="--", linewidth=0.5)
            axes[idx].legend()
        else:
            # Bar plot: average metric
            avg_metrics = grouped[metric].mean()
            axes[idx].bar(algorithms, avg_metrics, color=color, capsize=5)
            axes[idx].set_title(f"{title} (Obstacles {obst*100:.0f}%)")
            axes[idx].set_ylabel(ylabel)
            axes[idx].tick_params(axis='x', rotation=45)
        
        # Adjust x-axis labels for bottom subplots only
        if idx in [2, 3]:
            axes[idx].set_xlabel("Algorithm")
        else:
            axes[idx].set_xticklabels([])

    plt.tight_layout()
    # plt.savefig(save_path)
    # plt.close()
    plt.show()

def plot_summary_from_csv(csv_path, obstacle_percents=[0.05, 0.1, 0.25, 0.4], output_dir='./'):
    """
    Generates five images, each with four subplots for specified obstacle percentages.

    Parameters:
        csv_path (str): Path to the CSV file.
        obstacle_percents (list): List of obstacle percentages (default: [0.05, 0.1, 0.25, 0.4]).
        output_dir (str): Directory to save the output images (default: './').
    """
    df = load_and_prepare_csv(csv_path)
    if df.empty:
        print("No data to plot.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    if not output_dir.endswith('/'):
        output_dir += '/'

    # Plot 1: Average Computation Time
    plot_metric_subplots(
        df,
        metric='time',
        title='Avg Computation Time',
        ylabel='Time (s)',
        color='skyblue',
        obstacle_percents=obstacle_percents,
        save_path=f'{output_dir}avg_computation_time.png',
        is_line_plot=False
    )

    # Plot 2: Average Memory Usage
    plot_metric_subplots(
        df,
        metric='memory_mb',
        title='Avg Memory Usage',
        ylabel='Memory (MB)',
        color='lightcoral',
        obstacle_percents=obstacle_percents,
        save_path=f'{output_dir}avg_memory_usage.png',
        is_line_plot=False
    )

    # Plot 3: Average Path Length
    plot_metric_subplots(
        df,
        metric='path_length',
        title='Avg Path Length',
        ylabel='Steps',
        color='lightgreen',
        obstacle_percents=obstacle_percents,
        save_path=f'{output_dir}avg_path_length.png',
        is_line_plot=False
    )

    # Plot 4: Time vs. Nodes
    plot_metric_subplots(
        df,
        metric='time',
        title='Computation Time vs. Nodes',
        ylabel='Time (s)',
        color=None,  # Colors assigned per algorithm in line plot
        obstacle_percents=obstacle_percents,
        save_path=f'{output_dir}time_vs_nodes.png',
        is_line_plot=True
    )

    # Plot 5: Memory vs. Nodes
    plot_metric_subplots(
        df,
        metric='memory_mb',
        title='Memory Usage vs. Nodes',
        ylabel='Memory (MB)',
        color=None,  # Colors assigned per algorithm in line plot
        obstacle_percents=obstacle_percents,
        save_path=f'{output_dir}memory_vs_nodes.png',
        is_line_plot=True
    )

    print(f"Plots saved to {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    plot_summary_from_csv("simulation_summary.csv")