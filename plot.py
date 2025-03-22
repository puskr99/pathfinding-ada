import matplotlib.pyplot as plt

def plot_simulation_results(simulation_data):
    # Filter algorithms with data to avoid ZeroDivisionError
    valid_algorithms = [algo for algo in simulation_data.keys() if simulation_data[algo]["times"] and simulation_data[algo]["path_lengths"]]
    if not valid_algorithms:
        print("No simulation data to plot.")
        return
    
    # Calculate averages for valid algorithms
    avg_times = [sum(simulation_data[algo]["times"]) / len(simulation_data[algo]["times"]) for algo in valid_algorithms]
    avg_lengths = [sum(simulation_data[algo]["path_lengths"]) / len(simulation_data[algo]["path_lengths"]) for algo in valid_algorithms]
    
    # Create figure with 3 subplots side by side
    plt.figure(figsize=(15, 5))  # Wider figure for 3 plots

    # Subplot 1: Average Computation Time
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, position 1
    plt.bar(valid_algorithms, avg_times, color='skyblue')
    plt.title("Average Computation Time")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45)  # Rotate x-labels for readability

    # Subplot 2: Average Path Length
    plt.subplot(1, 3, 2)  # Position 2
    plt.bar(valid_algorithms, avg_lengths, color='lightgreen')
    plt.title("Average Path Length")
    plt.ylabel("Steps")
    plt.xticks(rotation=45)

    # Subplot 3: Computation Time per Iteration (Time Complexity Proxy)
    plt.subplot(1, 3, 3)  # Position 3
    for algo in valid_algorithms:
        times = simulation_data[algo]["times"]
        plt.plot(range(1, len(times) + 1), times, label=algo, marker='o', markersize=4)
    plt.title("Time per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()