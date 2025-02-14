import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_average_files(folder):
    files = glob.glob(os.path.join(folder, 'average_*.txt'))
    all_averages = []

    for file in files:
        with open(file, 'r') as f:
            averages = [float(line.strip()) for line in f]
            all_averages.append(averages)

    return all_averages

def calculate_medians(all_averages):
    max_length = max(len(avg) for avg in all_averages)
    medians = []

    for i in range(max_length):
        values_at_index = [avg[i] for avg in all_averages if i < len(avg)]
        median_value = np.median(values_at_index)
        medians.append(median_value)

    return medians

def plot_medians(medians):
    plt.plot(medians)
    plt.xlabel('Index')
    plt.ylabel('Median Value')
    plt.title('Median of Average Rewards')
    plt.show()

def main():
    folder = 'C:\\Users\\Madu\\Desktop\\disertatie\\random_agent'  # Change this to the folder containing your average_*.txt files
    all_averages = read_average_files(folder)
    medians = calculate_medians(all_averages)
    plot_medians(medians)

if __name__ == "__main__":
    main()