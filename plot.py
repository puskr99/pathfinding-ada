import matplotlib.pyplot as plt
import numpy as np

def smooth_data(x, y, window_size=5):
    """Apply a simple moving average to smooth data."""
    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    x_smooth = x[window_size-1:]  # Adjust x to match smoothed y length
    return x_smooth, y_smooth

def plot_simulation_results(simulation_data):
    # Filter algorithms with data to avoid ZeroDivisionError
    valid_algorithms = [algo for algo in simulation_data.keys() if simulation_data[algo]["times"] and simulation_data[algo]["path_lengths"]]
    
    if not valid_algorithms:
        print("No simulation data to plot.")
        return
    
    # Calculate averages and standard deviations for valid algorithms
    avg_times = [sum(simulation_data[algo]["times"]) / len(simulation_data[algo]["times"]) for algo in valid_algorithms]
    avg_lengths = [sum(simulation_data[algo]["path_lengths"]) / len(simulation_data[algo]["path_lengths"]) for algo in valid_algorithms]
    
    # Calculate standard deviations for error bars
    std_times = [np.std(simulation_data[algo]["times"]) for algo in valid_algorithms]
    std_lengths = [np.std(simulation_data[algo]["path_lengths"]) for algo in valid_algorithms]

    # Create figure with 3 subplots side by side
    plt.figure(figsize=(15, 5))  # Wider figure for 3 plots

    # Subplot 1: Average Computation Time with error bars
    plt.subplot(1, 3, 1)
    plt.bar(valid_algorithms, avg_times, yerr=std_times, color='skyblue', capsize=5)
    plt.title("Average Computation Time")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45)

    # Subplot 2: Average Path Length with error bars
    plt.subplot(1, 3, 2)
    plt.bar(valid_algorithms, avg_lengths, yerr=std_lengths, color='lightgreen', capsize=5)
    plt.title("Average Path Length")
    plt.ylabel("Steps")
    plt.xticks(rotation=45)

    # Subplot 3: Smoothed Computation Time vs. Number of Nodes
    plt.subplot(1, 3, 3)
    for algo in valid_algorithms:
        times = np.array(simulation_data[algo]["times"])
        nodes = np.array(simulation_data[algo]["nodes"])
        
        # Sort data by nodes to ensure a proper trend
        sorted_indices = np.argsort(nodes)
        nodes_sorted = nodes[sorted_indices]
        times_sorted = times[sorted_indices]
        
        # Smooth the data
        nodes_smooth, times_smooth = smooth_data(nodes_sorted, times_sorted, window_size=5)
        
        # Plot smoothed data
        plt.plot(nodes_smooth, times_smooth, label=algo, linewidth=2)
    
    plt.title("Smoothed Computation Time vs. Nodes (Log Scale)")
    plt.xlabel("Number of Nodes (rows * cols)")
    plt.ylabel("Time (s)")
    plt.yscale('log')  # Apply log scale explicitly
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()