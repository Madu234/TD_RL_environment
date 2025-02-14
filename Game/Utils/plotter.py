import os
import matplotlib.pyplot as plt

def read_values_from_file(filepath):
    with open(filepath, 'r') as file:
        values = [float(line.strip()) for line in file]
    return values

def plot_values(filepaths, labels, title, xlabel, ylabel, output_filename):
    plt.figure(figsize=(10, 6))
    
    for filepath, label in zip(filepaths, labels):
        values = read_values_from_file(filepath)
        plt.plot(values, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    # Define the file paths
    filepaths = [
        "c:/Users/Madu/Desktop/Disertatie/TD_RL_environment/Game/DQN64_1000episodes/average_DQN64_1000episodes4.txt",
        "c:/Users/Madu/Desktop/Disertatie/TD_RL_environment/Game/random_1000episodes/average_random_1000episodes1.txt",
        "C:/Users/Madu/Desktop/Disertatie/TD_RL_environment/Game/DQN16_1000episodes/average_DQN16_1000episodes2.txt"
    ]
    
    # Define the labels for the plots
    labels = ["DQN64_agent", "random_agent", "DQN16_agent"]
    
    # Plot the values
    plot_values(
        filepaths=filepaths,
        labels=labels,
        title="Comparison of DQN and Random Strategy",
        xlabel="Average of 10 episodes",
        ylabel="Reward",
        output_filename="comparison_plot.png"
    )