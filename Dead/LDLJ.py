import matplotlib.pyplot as plt
import numpy as np


def plot_ldlj_values(results):
    """Plot all LDLJ values across all files with a best fit line."""
    all_LDLJ_values = []
    for file_result in results.values():
        all_LDLJ_values.extend(file_result['parameters']['LDLJ_values'])

    x_values = np.arange(1, len(all_LDLJ_values) + 1)
    coefficients = np.polyfit(x_values, all_LDLJ_values, 1)
    best_fit_line = np.poly1d(coefficients)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, all_LDLJ_values, color='b', label='LDLJ Values', alpha=0.7)
    plt.plot(x_values, best_fit_line(x_values), color='r', linestyle='--', label='Best Fit Line')
    plt.title('LDLJ Values Across All Files with Best Fit Line')
    plt.xlabel('Index')
    plt.ylabel('LDLJ Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_averaged_ldlj_values(results):
    """Plot averaged LDLJ values for each trial with a best fit line."""
    averaged_LDLJ_values = []
    for file_result in results.values():
        LDLJ_values = file_result['parameters']['LDLJ_values']
        averaged_LDLJ_values.append(np.mean(LDLJ_values))

    x_values = np.arange(1, len(averaged_LDLJ_values) + 1)
    coefficients = np.polyfit(x_values, averaged_LDLJ_values, 1)
    best_fit_line = np.poly1d(coefficients)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, averaged_LDLJ_values, color='b', label='Averaged LDLJ Values', alpha=0.7)
    plt.plot(x_values, best_fit_line(x_values), color='r', linestyle='--', label='Best Fit Line')
    plt.title('Averaged LDLJ Values Across All Trials with Best Fit Line')
    plt.xlabel('Trial Index')
    plt.ylabel('Averaged LDLJ Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ldlj_subplots(results):
    """Create 16 subplots for LDLJ values across trials."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    fig.suptitle('LDLJ Values Across Trials', fontsize=16)

    for i in range(16):
        LDLJ_values = [file_result['parameters']['LDLJ_values'][i] for file_result in results.values()]
        x_values = np.arange(1, len(LDLJ_values) + 1)
        coefficients = np.polyfit(x_values, LDLJ_values, 1)
        best_fit_line = np.poly1d(coefficients)

        ax = axes[i // 4, i % 4]
        ax.scatter(x_values, LDLJ_values, color='b', label='LDLJ Values', alpha=0.7)
        ax.plot(x_values, best_fit_line(x_values), color='r', linestyle='--', label='Best Fit Line')
        ax.set_title(f'LDLJ Index {i + 1}')
        ax.set_xlabel('Trial')
        ax.set_ylabel('LDLJ Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
