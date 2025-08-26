# Funtions 
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

def scatter_plot_duration_distance_by_choice(updated_metrics, overlay_hands=True, selected_subjects=None, special_indices=None):
    """
    Plots a scatter plot of all durations vs all distances across selected subjects, hands, and trials,
    and fits a hyperbolic curve. Optionally overlays both hands or separates them.
    Allows highlighting specific reach indices.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        overlay_hands (bool): If True, overlays both hands in the same plot. If False, separates them.
        selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
        special_indices (list or None): List of reach indices to include. If None, all indices are included.
    """
    def extract_data(metrics, hand, indices):
        durations = [[] for _ in range(16)]
        distances = [[] for _ in range(16)]
        for trial, trial_durations in metrics[hand]['durations'].items():
            for reach_index, duration in enumerate(trial_durations):
                if not pd.isna(duration) and (indices is None or reach_index in indices):
                    durations[reach_index].append(duration)
        for trial, trial_distances in metrics[hand]['distance'].items():
            for reach_index, distance in enumerate(trial_distances):
                if not pd.isna(distance) and (indices is None or reach_index in indices):
                    distances[reach_index].append(distance)
        return durations, distances

    def _plot_scatter_with_fit(durations, distances, title, special_indices=None):
        """
        Helper function to plot scatter and fit a hyperbolic curve.

        Parameters:
            durations (list of lists): List of durations for each reach index.
            distances (list of lists): List of distances for each reach index.
            title (str): Title of the plot.
            special_indices (list or None): List of reach indices to include. If None, all indices are included.
        """
        # Define color groups for reach indices
        color_groups = {
            0: 'red', 4: 'red', 8: 'red', 12: 'red',
            1: 'blue', 5: 'blue', 9: 'blue', 13: 'blue',
            2: 'green', 6: 'green', 10: 'green', 14: 'green',
            3: 'purple', 7: 'purple', 11: 'purple', 15: 'purple'
        }

        # Flatten durations and distances for correlation calculation
        flat_durations = [d for i, sublist in enumerate(durations) for d in sublist if special_indices is None or i in special_indices]
        flat_distances = [d for i, sublist in enumerate(distances) for d in sublist if special_indices is None or i in special_indices]

        # Calculate Spearman correlation
        if len(flat_durations) > 1 and len(flat_distances) > 1:
            spearman_corr, p_value = spearmanr(flat_durations, flat_distances)
        else:
            spearman_corr, p_value = np.nan, np.nan

        # Scatter plot
        plt.figure(figsize=(8, 6))
        for i in range(16):
            if special_indices is None or i in special_indices:
                alpha = 1.0 if special_indices and i in special_indices else 0.7
                size = 100 if special_indices and i in special_indices else 50
                if i in [0, 1, 2, 3]:  # Label only for these indices
                    plt.scatter(durations[i], distances[i], alpha=alpha, color=color_groups[i], s=size, label=f"{i}, {i+4}, {i+8}, {i+12}")
                else:
                    plt.scatter(durations[i], distances[i], alpha=alpha, color=color_groups[i], s=size)

                # Overlay median values
                if durations[i] and distances[i]:
                    median_duration = np.median(durations[i])
                    median_distance = np.median(distances[i])
                    plt.scatter(median_duration, median_distance, color='none', edgecolor='black', s=100, zorder=5)
                    plt.text(median_duration, median_distance, f"{i}", fontsize=7, color='black', ha="center", va="center",
                             bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle', linewidth=1.5))

        plt.title(f"Scatter Plot of Durations vs Distances ({title})")

        labels = {
            'distance': "Good → Bad (cm)",
            'durations': "Good / Fast → Bad / Slow (s)"
        }
        plt.xlabel(f"Durations ({labels.get('durations', '')})")
        plt.ylabel(f"Distance ({labels.get('distance', '')})")
        plt.grid(alpha=0.5)

        # Add Spearman correlation and number of data points to the plot
        plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.4f}\nP-value: {p_value:.4f}\nData Points: {len(flat_durations)}",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Fit a hyperbolic curve
        def hyperbolic_func(x, a, b):
            return a / (x + b)

        try:
            params, _ = curve_fit(hyperbolic_func, flat_durations, flat_distances)
            x_fit = np.linspace(min(flat_durations), max(flat_durations), 500)
            y_fit = hyperbolic_func(x_fit, *params)
            plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
        except Exception as e:
            print(f"Hyperbolic fit failed: {e}")

        plt.legend()
        plt.tight_layout()
        plt.show()

    if selected_subjects is None:
        selected_subjects = updated_metrics.keys()

    if overlay_hands:
        all_durations = [[] for _ in range(16)]
        all_distances = [[] for _ in range(16)]
        for subject in selected_subjects:
            if subject in updated_metrics:
                for hand in ['left', 'right']:
                    durations, distances = extract_data(updated_metrics[subject], hand, special_indices)
                    for i in range(16):
                        all_durations[i].extend(durations[i])
                        all_distances[i].extend(distances[i])

        # Plot overlayed hands
        _plot_scatter_with_fit(all_durations, all_distances, title=f"Overlayed Hands (Subjects: {', '.join(selected_subjects)})", special_indices=special_indices)
    else:
        for hand in ['left', 'right']:
            all_durations = [[] for _ in range(16)]
            all_distances = [[] for _ in range(16)]
            for subject in selected_subjects:
                if subject in updated_metrics:
                    durations, distances = extract_data(updated_metrics[subject], hand, special_indices)
                    for i in range(16):
                        all_durations[i].extend(durations[i])
                        all_distances[i].extend(distances[i])

            # Plot separate hands
            _plot_scatter_with_fit(all_durations, all_distances, title=f"{hand.capitalize()} Hand (Subjects: {', '.join(selected_subjects)})", special_indices=special_indices)



# 1.1	Does a subject show a speed–accuracy trade-off trial by trial?
# -------------------------------------------------------------------------------------------------------------------
# Scatter plot for all durations and all distances with hyperbolic fit per subject per hand (or overlay hands)
def duration_distance_trial_by_trial(updated_metrics, overlay_hands=True, selected_subjects=None):
    """
    Plots a scatter plot of all durations vs all distances across selected subjects, hands, and trials,
    and fits a hyperbolic curve. Optionally overlays both hands or separates them.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        overlay_hands (bool): If True, overlays both hands in the same plot. If False, separates them.
        selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
    """
    def extract_data(metrics, hand):
        durations = []
        distances = []
        for trial, trial_durations in metrics[hand]['durations'].items():
            durations.extend([duration for duration in trial_durations if not pd.isna(duration)])
        for trial, trial_distances in metrics[hand]['distance'].items():
            distances.extend([distance for distance in trial_distances if not pd.isna(distance)])
        return durations, distances

    if selected_subjects is None:
        selected_subjects = updated_metrics.keys()

    if overlay_hands:
        all_durations, all_distances = [], []
        for subject in selected_subjects:
            if subject in updated_metrics:
                for hand in ['left', 'right']:
                    durations, distances = extract_data(updated_metrics[subject], hand)
                    all_durations.extend(durations)
                    all_distances.extend(distances)

        # Determine title based on the number of selected subjects
        title = f"{len(selected_subjects)} Subjects" if len(selected_subjects) > 1 else f"Subject: {', '.join(selected_subjects)}"
        _plot_scatter_with_fit(all_durations, all_distances, title=f"Overlayed Hands ({title})")
    else:
        for hand in ['left', 'right']:
            all_durations, all_distances = [], []
            for subject in selected_subjects:
                if subject in updated_metrics:
                    durations, distances = extract_data(updated_metrics[subject], hand)
                    all_durations.extend(durations)
                    all_distances.extend(distances)

            # Determine title based on the number of selected subjects
            title = f"{len(selected_subjects)} Subjects" if len(selected_subjects) > 1 else f"Subject: {', '.join(selected_subjects)}"
            _plot_scatter_with_fit(all_durations, all_distances, title=f"{hand.capitalize()} Hand ({title})")

def _plot_scatter_with_fit(durations, distances, title):
    """
    Helper function to plot scatter and fit a hyperbolic curve.

    Parameters:
        durations (list): List of durations.
        distances (list): List of distances.
        title (str): Title of the plot.
    """
    # Calculate Spearman correlation
    if len(durations) > 1 and len(distances) > 1:
        spearman_corr, p_value = spearmanr(durations, distances)
    else:
        spearman_corr, p_value = np.nan, np.nan

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(durations, distances, alpha=0.7, color='blue')
    plt.title(f"Scatter Plot of Durations vs Distances ({title})")

    labels = {
        'distance': "Good → Bad (cm)",
        'durations': "Good / Fast → Bad / Slow (s)"
    }
    plt.xlabel(f"Durations ({labels.get('durations', '')})")
    plt.ylabel(f"Distance ({labels.get('distance', '')})")
    plt.grid(alpha=0.5)

    # Add Spearman correlation and number of data points to the plot
    plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.4f}\nP-value: {p_value:.4f}\nData Points: {len(durations)}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Fit a hyperbolic curve
    def hyperbolic_func(x, a, b):
        return a / (x + b)

    try:
        params, _ = curve_fit(hyperbolic_func, durations, distances)
        x_fit = np.linspace(min(durations), max(durations), 500)
        y_fit = hyperbolic_func(x_fit, *params)
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
    except Exception as e:
        print(f"Hyperbolic fit failed: {e}")

    plt.legend()
    plt.tight_layout()
    plt.show()

# Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject and hand
def calculate_duration_distance_trial_by_trial(updated_metrics, selected_subjects=None):
    """
    Calculates Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b)
    for durations vs distances for each subject and hand.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        selected_subjects (list or None): List of subjects to include. If None, includes all subjects.

    Returns:
        dict: Dictionary containing results for each subject and hand.
    """
    def extract_data(metrics, hand):
        durations = []
        distances = []
        for trial, trial_durations in metrics[hand]['durations'].items():
            durations.extend([duration for duration in trial_durations if not pd.isna(duration)])
        for trial, trial_distances in metrics[hand]['distance'].items():
            distances.extend([distance for distance in trial_distances if not pd.isna(distance)])
        return durations, distances

    if selected_subjects is None:
        selected_subjects = updated_metrics.keys()

    results = {}

    for subject in selected_subjects:
        if subject in updated_metrics:
            results[subject] = {}
            for hand in ['left', 'right']:
                durations, distances = extract_data(updated_metrics[subject], hand)

                # Calculate Spearman correlation
                if len(durations) > 1 and len(distances) > 1:
                    spearman_corr, p_value = spearmanr(durations, distances)
                else:
                    spearman_corr, p_value = np.nan, np.nan

                # Fit a hyperbolic curve
                def hyperbolic_func(x, a, b):
                    return a / (x + b)

                try:
                    params, _ = curve_fit(hyperbolic_func, durations, distances)
                    a, b = params
                except Exception:
                    a, b = np.nan, np.nan

                # Store results
                results[subject][hand] = {
                    "spearman_corr": spearman_corr,
                    "p_value": p_value,
                    "data_points": len(durations),
                    "hyperbolic_fit_a": a,
                    "hyperbolic_fit_b": b
                }

    return results

# Plot column chart of Spearman correlations and p-values for left and right hands
def plot_column_spearman_corr_pvalues_trial_by_trial(results):
    """
    Plots column charts of Spearman correlations and p-values for left and right hands across all subjects.

    Parameters:
        results (dict): Results containing Spearman correlations and p-values for each subject and hand.
    """
    subjects = list(results.keys())
    correlations_left = [results[subject]['left']['spearman_corr'] if 'left' in results[subject] else np.nan for subject in subjects]
    correlations_right = [results[subject]['right']['spearman_corr'] if 'right' in results[subject] else np.nan for subject in subjects]
    pvalues_left = [results[subject]['left']['p_value'] if 'left' in results[subject] else np.nan for subject in subjects]
    pvalues_right = [results[subject]['right']['p_value'] if 'right' in results[subject] else np.nan for subject in subjects]

    x = np.arange(len(subjects))  # the label locations
    width = 0.35  # the width of the bars

    # Create figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Spearman correlations
    bars_left_corr = axes[0].bar(x - width / 2, correlations_left, width, label='Left Hand', color='orange', alpha=0.7)
    bars_right_corr = axes[0].bar(x + width / 2, correlations_right, width, label='Right Hand', color='blue', alpha=0.7)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Spearman Correlation')
    axes[0].set_title('Spearman Correlations by Subject and Hand')
    axes[0].legend()

    # Annotate Spearman correlation values
    for bar in bars_left_corr:
        height = bar.get_height()
        if not np.isnan(height):
            axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars_right_corr:
        height = bar.get_height()
        if not np.isnan(height):
            axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Plot P-values
    bars_left_pval = axes[1].bar(x - width / 2, pvalues_left, width, label='Left Hand', color='orange', alpha=0.7)
    bars_right_pval = axes[1].bar(x + width / 2, pvalues_right, width, label='Right Hand', color='blue', alpha=0.7)
    axes[1].axhline(0.05, color='red', linewidth=0.8, linestyle='--', label='Significance Threshold (0.05)')
    axes[1].set_ylabel('P-value')
    axes[1].set_title('P-values by Subject and Hand')
    axes[1].legend()

    # Annotate P-value values only when > 0.05
    for bar in bars_left_pval:
        height = bar.get_height()
        if not np.isnan(height) and height > 0.05:
            axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars_right_pval:
        height = bar.get_height()
        if not np.isnan(height) and height > 0.05:
            axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Set x-axis labels for both subplots
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subjects, rotation=45, ha='right')
    axes[1].set_xlabel('Subjects')

    plt.tight_layout()
    plt.show()

# Plot histograms for Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
def plot_histogram_spearman_corr_with_stats_trial_by_trial(results):
    """
    Plots histograms of Spearman correlations for durations vs distances across all subjects,
    overlaying left and right hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
    """
    correlations_left = []
    correlations_right = []

    for subject, hands_data in results.items():
        for hand, metrics in hands_data.items():
            if 'spearman_corr' in metrics and not np.isnan(metrics['spearman_corr']):
                if hand == 'left':
                    correlations_left.append(metrics['spearman_corr'])
                elif hand == 'right':
                    correlations_right.append(metrics['spearman_corr'])

    # Calculate statistics
    median_left = np.median(correlations_left)
    iqr_left = np.percentile(correlations_left, 75) - np.percentile(correlations_left, 25)
    q1_left = np.percentile(correlations_left, 25)
    q3_left = np.percentile(correlations_left, 75)

    median_right = np.median(correlations_right)
    iqr_right = np.percentile(correlations_right, 75) - np.percentile(correlations_right, 25)
    q1_right = np.percentile(correlations_right, 25)
    q3_right = np.percentile(correlations_right, 75)

    # Perform Wilcoxon signed-rank test for each hand separately
    stat_left, p_value_left = wilcoxon(correlations_left)
    stat_right, p_value_right = wilcoxon(correlations_right)

    # Plot histogram for Spearman correlations
    plt.figure(figsize=(8, 6))
    plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
    plt.axvline(median_left, color='orange', linestyle='--', label=f"Median Left: {median_left:.2f}")
    plt.axvline(median_right, color='blue', linestyle='--', label=f"Median Right: {median_right:.2f}")
    plt.axvspan(q1_left, q3_left, color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
    plt.axvspan(q1_right, q3_right, color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")
    plt.title("Histogram of Spearman Correlations by Hand")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # # Print statistics
    # print(f"Left Hand: Median = {median_left:.2f}, IQR = {iqr_left:.2f}")
    # print(f"Right Hand: Median = {median_right:.2f}, IQR = {iqr_right:.2f}")
    # print(f"Wilcoxon Signed-Rank Test (Left Hand): Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}")
    # print(f"Wilcoxon Signed-Rank Test (Right Hand): Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}")

# Perform Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately
def wilcoxon_test_spearman_corr_separate_trial_by_trial(results):
    """
    Performs the Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Test statistics and p-values for left and right hands.
    """
    correlations_left = []
    correlations_right = []

    for subject, hands_data in results.items():
        if 'left' in hands_data and 'spearman_corr' in hands_data['left']:
            correlations_left.append(hands_data['left']['spearman_corr'])
        if 'right' in hands_data and 'spearman_corr' in hands_data['right']:
            correlations_right.append(hands_data['right']['spearman_corr'])

    # Perform Wilcoxon signed-rank test for left hand
    stat_left, p_value_left = wilcoxon(correlations_left)

    # Perform Wilcoxon signed-rank test for right hand
    stat_right, p_value_right = wilcoxon(correlations_right)

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results:")
    print(f"Left Hand: Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}, Data Points = {len(correlations_left)}")
    print(f"Right Hand: Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}, Data Points = {len(correlations_right)}")


    return {
        "left": {"statistic": stat_left, "p_value": p_value_left},
        "right": {"statistic": stat_right, "p_value": p_value_right}
    }

# -------------------------------------------------------------------------------------------------------------------
# 1.1	Does a subject show a speed–accuracy trade-off trial by trial?
def Check_SAT_in_trial_by_trial(updated_metrics, selected_subjects, sample_subject= ['07/22/HW'], overlay_hands=False):
    """
    Analyzes the speed-accuracy trade-off trial by trial for selected subjects.
    Generates scatter plots, calculates Spearman correlations, and performs statistical tests.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
        overlay_hands (bool): If True, overlays both hands in the same plot. If False, separates them.

    Returns:
        dict: Results containing Spearman correlations, p-values, and statistical test results.
    """
    # Scatter plot for all durations and distances
    # duration_distance_trial_by_trial(updated_metrics, overlay_hands=overlay_hands, selected_subjects=selected_subjects)
    duration_distance_trial_by_trial(updated_metrics, overlay_hands=overlay_hands, selected_subjects=sample_subject)

    # Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters
    results_trial_by_trial = calculate_duration_distance_trial_by_trial(updated_metrics, selected_subjects=selected_subjects)

    # Plot column chart of Spearman correlations and p-values
    plot_column_spearman_corr_pvalues_trial_by_trial(results_trial_by_trial)

    # Plot histograms for Spearman correlations and report statistics
    plot_histogram_spearman_corr_with_stats_trial_by_trial(results_trial_by_trial)

    # Perform Wilcoxon signed-rank test on Spearman correlation values
    wilcoxon_results_by_hand_trial_by_trial = wilcoxon_test_spearman_corr_separate_trial_by_trial(results_trial_by_trial)

    return {
        "results": results_trial_by_trial,
        "wilcoxon_results_by_hand": wilcoxon_results_by_hand_trial_by_trial
    }

# -------------------------------------------------------------------------------------------------------------------
# 1.2	Do reach types that are faster on average also tend to be less accurate on average?

# Box plot for all reach indices with median and IQR
def boxplot_trials_mean_median_of_reach_indices(updated_metrics, subject, hand, metric):
    """
    Plots a box plot for all reach indices with median and IQR.
    Y-axis represents reach indices, and X-axis represents the metric values.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric (str): Metric to plot ('durations' or 'distance').
    """

    # Collect data for each reach index
    reach_data = {reach_index: [] for reach_index in range(16)}

    trials = updated_metrics[subject][hand][metric].keys()

    for trial in trials:
        trial_values = np.array(updated_metrics[subject][hand][metric][trial])

        for reach_index in range(len(trial_values)):
            if not np.isnan(trial_values[reach_index]):
                reach_data[reach_index].append(trial_values[reach_index])

    # Convert to DataFrame for box plot
    reach_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in reach_data.items()]))

    # Plot box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=reach_df, orient="h", palette="coolwarm", showmeans=False, width=0.6)
    plt.title(f"Box Plot for {subject} - {hand.capitalize()} Hand - {metric.capitalize()}")
    plt.xlabel(f"{metric.capitalize()} Values")
    plt.ylabel("Reach Index")
    plt.grid(alpha=0.5)

    # Add median and IQR annotations
    medians = reach_df.median()
    q1 = reach_df.quantile(0.25)
    q3 = reach_df.quantile(0.75)
    for i, median in enumerate(medians):
        plt.text(median, i, f"{median:.2f}", verticalalignment='center', color='black', fontsize=8, horizontalalignment='right')

    plt.tight_layout()
    plt.show()

# Calculate average and median 'durations', and average and median 'distance' for each reach_index for all subjects and hands across trials
def calculate_trials_mean_median_of_reach_indices(updated_metrics, metric_x, metric_y):
    """
    Calculates average and median 'durations', and average and median 'distance' for each reach_index
    for all subjects and hands, and returns two dictionaries: one for mean and one for median.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.

    Returns:
        tuple: (mean_statistics, median_statistics) dictionaries containing statistics for all subjects and hands.
    """
    mean_statistics = {}
    median_statistics = {}

    for subject, hands_data in updated_metrics.items():
        mean_statistics[subject] = {}
        median_statistics[subject] = {}
        for hand, metrics in hands_data.items():
            mean_statistics[subject][hand] = []
            median_statistics[subject][hand] = []

            for reach_index in range(16):
                x_values = []
                y_values = []

                trials = metrics[metric_x].keys()

                for trial in trials:
                    trial_x = np.array(metrics[metric_x][trial])
                    trial_y = np.array(metrics[metric_y][trial])

                    # Collect data for the specified reach index
                    if reach_index < len(trial_x) and reach_index < len(trial_y):
                        x_values.append(trial_x[reach_index])
                        y_values.append(trial_y[reach_index])
                
                # Remove NaN values
                x_values = np.array(x_values)
                y_values = np.array(y_values)
                valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[valid_indices]
                y_values = y_values[valid_indices]

                # Calculate statistics
                avg_duration = np.mean(x_values) if len(x_values) > 0 else np.nan
                median_duration = np.median(x_values) if len(x_values) > 0 else np.nan
                avg_distance = np.mean(y_values) if len(y_values) > 0 else np.nan
                median_distance = np.median(y_values) if len(y_values) > 0 else np.nan

                mean_statistics[subject][hand].append({
                    "reach_index": reach_index,
                    "avg_duration": avg_duration,
                    "avg_distance": avg_distance
                })

                median_statistics[subject][hand].append({
                    "reach_index": reach_index,
                    "median_duration": median_duration,
                    "median_distance": median_distance
                })

    return mean_statistics, median_statistics

# Overlay scatter plots for all reach indices in a single plot using either mean or median statistics
def plot_trials_mean_median_of_reach_indices(stats, subject, hand, metric_x, metric_y, stat_type="avg", use_unique_colors=False):
    """
    Overlays scatter plots for all reach indices in a single plot using either mean or median statistics.
    Groups reach indices by 0, 4, 8, 12; 1, 5, 9, 13; 2, 6, 10, 14; 3, 7, 11, 15, and uses the same color for each group
    or assigns a unique color to each reach index if `use_unique_colors` is True.
    Calculates and returns the Spearman correlation for the overlayed points.

    Parameters:
        stats (dict): Statistics (mean or median) for all subjects and hands.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        stat_type (str): Type of statistics to use ("mean" or "median").
        use_unique_colors (bool): If True, assigns a unique color to each reach index. Otherwise, groups by color.

    Returns:
        tuple: Spearman correlation and p-value for the overlayed points.
    """
    # Initialize the plot
    plt.figure(figsize=(6, 6))

    x_values = []
    y_values = []

    # Define color groups for reach indices
    if use_unique_colors:
        color_palette = sns.color_palette("coolwarm", 16)
        color_groups = {i: color_palette[i] for i in range(16)}
    else:
        color_groups = {
            0: 'red', 4: 'red', 8: 'red', 12: 'red',
            1: 'blue', 5: 'blue', 9: 'blue', 13: 'blue',
            2: 'green', 6: 'green', 10: 'green', 14: 'green',
            3: 'purple', 7: 'purple', 11: 'purple', 15: 'purple'
        }

    group_indexes = {
        0: [0, 4, 8, 12],
        1: [1, 5, 9, 13],
        2: [2, 6, 10, 14],
        3: [3, 7, 11, 15]
    }

    for reach_index in range(16):
        # Get statistics for the current reach index
        duration = stats[subject][hand][reach_index][f"{stat_type}_duration"]
        distance = stats[subject][hand][reach_index][f"{stat_type}_distance"]

        # Overlay the points
        if not np.isnan(duration) and not np.isnan(distance):
            x_values.append(duration)
            y_values.append(distance)
            color = color_groups[reach_index]
            label = f"Group {reach_index % 4}, {group_indexes[reach_index % 4]}" if not use_unique_colors and reach_index < 4 else None
            plt.scatter(duration, distance, color=color, s=50, label=label, zorder=5)

            # Annotate the reach index
            plt.text(duration, distance, f"{reach_index}", fontsize=12, color=color, ha="right" if stat_type == "mean" else "left")

    # Calculate Spearman correlation
    if len(x_values) > 1 and len(y_values) > 1:
        spearman_corr, p_value = spearmanr(x_values, y_values)
    else:
        spearman_corr, p_value = np.nan, np.nan

    # Add labels and legend
    labels = {
        'distance': "Good → Bad (cm)",
        'accuracy': "Bad → Good (%)",
        'durations': "Good / Fast → Bad / Slow (s)",
        'speed': "Bad / Slow → Good / Fast (1/s)"
    }
    plt.xlabel(f"{metric_x.capitalize()} ({labels.get(metric_x, '')})")
    plt.ylabel(f"{metric_y.capitalize()} ({labels.get(metric_y, '')})")
    plt.title(f"Overlay of Reach Statistics ({subject}, {hand.capitalize()}, {stat_type.capitalize()})\nSpearman Corr: {spearman_corr:.2f}, P-value: {p_value:.2f}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

    return spearman_corr, p_value

# Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject and hand across reach indices
def calculate_duration_distance_trials_mean_median_of_reach_indices(stats, selected_subjects=None, stat_type="avg"):
    """
    Calculates Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b)
    for reach statistics (e.g., durations vs distances) for each subject and hand.

    Parameters:
        stats (dict): Statistics (mean or median) for all subjects and hands.
        selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
        stat_type (str): Type of statistics to use ("mean" or "median").

    Returns:
        dict: Dictionary containing results for each subject and hand.
    """
    if selected_subjects is None:
        selected_subjects = stats.keys()

    results = {}

    for subject in selected_subjects:
        if subject in stats:
            results[subject] = {}
            for hand in ['left', 'right']:
                x_values = []
                y_values = []

                for reach_index in range(16):
                    # Get statistics for the current reach index
                    duration = stats[subject][hand][reach_index].get(f"{stat_type}_duration", np.nan)
                    distance = stats[subject][hand][reach_index].get(f"{stat_type}_distance", np.nan)

                    if not np.isnan(duration) and not np.isnan(distance):
                        x_values.append(duration)
                        y_values.append(distance)

                # Calculate Spearman correlation
                if len(x_values) > 1 and len(y_values) > 1:
                    spearman_corr, p_value = spearmanr(x_values, y_values)
                else:
                    spearman_corr, p_value = np.nan, np.nan

                # Fit a hyperbolic curve
                def hyperbolic_func(x, a, b):
                    return a / (x + b)

                try:
                    params, _ = curve_fit(hyperbolic_func, x_values, y_values)
                    a, b = params
                except Exception:
                    a, b = np.nan, np.nan

                # Store results
                results[subject][hand] = {
                    "spearman_corr": spearman_corr,
                    "p_value": p_value,
                    "data_points": len(x_values),
                    "hyperbolic_fit_a": a,
                    "hyperbolic_fit_b": b
                }

    return results

# Plot column chart of Spearman correlations and p-values for left and right hands
def plot_column_spearman_corr_pvalues_trials_mean_median_of_reach_indices(results):
    """
    Plots column charts of Spearman correlations and p-values for left and right hands across all subjects.

    Parameters:
        results (dict): Results containing Spearman correlations and p-values for each subject and hand.
    """
    subjects = list(results.keys())
    correlations_left = [results[subject]['left']['spearman_corr'] if 'left' in results[subject] else np.nan for subject in subjects]
    correlations_right = [results[subject]['right']['spearman_corr'] if 'right' in results[subject] else np.nan for subject in subjects]
    pvalues_left = [results[subject]['left']['p_value'] if 'left' in results[subject] else np.nan for subject in subjects]
    pvalues_right = [results[subject]['right']['p_value'] if 'right' in results[subject] else np.nan for subject in subjects]

    x = np.arange(len(subjects))  # the label locations
    width = 0.35  # the width of the bars

    # Create figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Spearman correlations
    bars_left_corr = axes[0].bar(x - width / 2, correlations_left, width, label='Left Hand', color='orange', alpha=0.7)
    bars_right_corr = axes[0].bar(x + width / 2, correlations_right, width, label='Right Hand', color='blue', alpha=0.7)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Spearman Correlation')
    axes[0].set_title('Spearman Correlations by Subject and Hand')
    axes[0].legend()

    # Annotate Spearman correlation values
    for bar in bars_left_corr:
        height = bar.get_height()
        if not np.isnan(height):
            axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars_right_corr:
        height = bar.get_height()
        if not np.isnan(height):
            axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Plot P-values
    bars_left_pval = axes[1].bar(x - width / 2, pvalues_left, width, label='Left Hand', color='orange', alpha=0.7)
    bars_right_pval = axes[1].bar(x + width / 2, pvalues_right, width, label='Right Hand', color='blue', alpha=0.7)
    axes[1].axhline(0.05, color='red', linewidth=0.8, linestyle='--', label='Significance Threshold (0.05)')
    axes[1].set_ylabel('P-value')
    axes[1].set_title('P-values by Subject and Hand')
    axes[1].legend()

    # Annotate P-value values only when > 0.05
    for bar in bars_left_pval:
        height = bar.get_height()
        if not np.isnan(height) and height > 0.05:
            axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars_right_pval:
        height = bar.get_height()
        if not np.isnan(height) and height > 0.05:
            axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Set x-axis labels for both subplots
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subjects, rotation=45, ha='right')
    axes[1].set_xlabel('Subjects')

    plt.tight_layout()
    plt.show()

# Plot histograms for Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
def plot_histogram_spearman_corr_with_stats_trials_mean_median_of_reach_indices(results):
    """
    Plots histograms of Spearman correlations for durations vs distances across all subjects,
    overlaying left and right hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
    """
    correlations_left = []
    correlations_right = []

    for subject, hands_data in results.items():
        for hand, metrics in hands_data.items():
            if 'spearman_corr' in metrics and not np.isnan(metrics['spearman_corr']):
                if hand == 'left':
                    correlations_left.append(metrics['spearman_corr'])
                elif hand == 'right':
                    correlations_right.append(metrics['spearman_corr'])

    # Calculate statistics
    median_left = np.median(correlations_left)
    iqr_left = np.percentile(correlations_left, 75) - np.percentile(correlations_left, 25)
    q1_left = np.percentile(correlations_left, 25)
    q3_left = np.percentile(correlations_left, 75)

    median_right = np.median(correlations_right)
    iqr_right = np.percentile(correlations_right, 75) - np.percentile(correlations_right, 25)
    q1_right = np.percentile(correlations_right, 25)
    q3_right = np.percentile(correlations_right, 75)

    # Perform Wilcoxon signed-rank test for each hand separately
    stat_left, p_value_left = wilcoxon(correlations_left)
    stat_right, p_value_right = wilcoxon(correlations_right)

    # Plot histogram for Spearman correlations
    plt.figure(figsize=(8, 6))
    plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
    plt.axvline(median_left, color='orange', linestyle='--', label=f"Median Left: {median_left:.2f}")
    plt.axvline(median_right, color='blue', linestyle='--', label=f"Median Right: {median_right:.2f}")
    plt.axvspan(q1_left, q3_left, color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
    plt.axvspan(q1_right, q3_right, color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")
    plt.title("Histogram of Spearman Correlations by Hand")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # # Print statistics
    # print(f"Left Hand: Median = {median_left:.2f}, IQR = {iqr_left:.2f}")
    # print(f"Right Hand: Median = {median_right:.2f}, IQR = {iqr_right:.2f}")
    # print(f"Wilcoxon Signed-Rank Test (Left Hand): Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}")
    # print(f"Wilcoxon Signed-Rank Test (Right Hand): Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}")

# Perform Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately
def wilcoxon_test_spearman_corr_separate(results):
    """
    Performs the Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Test statistics and p-values for left and right hands.
    """
    correlations_left = []
    correlations_right = []

    for subject, hands_data in results.items():
        if 'left' in hands_data and 'spearman_corr' in hands_data['left']:
            correlations_left.append(hands_data['left']['spearman_corr'])
        if 'right' in hands_data and 'spearman_corr' in hands_data['right']:
            correlations_right.append(hands_data['right']['spearman_corr'])

    # Perform Wilcoxon signed-rank test for left hand
    stat_left, p_value_left = wilcoxon(correlations_left)

    # Perform Wilcoxon signed-rank test for right hand
    stat_right, p_value_right = wilcoxon(correlations_right)

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results:")
    print(f"Left Hand: Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}, Data Points = {len(correlations_left)}")
    print(f"Right Hand: Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}, Data Points = {len(correlations_right)}")

    return {
        "left": {"statistic": stat_left, "p_value": p_value_left},
        "right": {"statistic": stat_right, "p_value": p_value_right}
    }

# Compare left vs right hands at the subject level using Wilcoxon signed-rank test on paired subject Spearman correlations for durations and distances
def compare_left_vs_right_hands_trials_mean_median_of_reach_indices(results):
    """
    Compares the left and right hands at the subject level using the Wilcoxon signed-rank test
    on paired subject Spearman correlations for durations and distances.

    Parameters:
        results (dict): Dictionary containing Spearman correlation results for each subject.

    Returns:
        dict: Results of the Wilcoxon signed-rank test for durations and distances.
    """
    left_spearman_corrs = []
    right_spearman_corrs = []


    for subject, hands_data in results.items():
        if 'left' in hands_data and 'right' in hands_data:
            left_corr = hands_data['left']['spearman_corr']
            right_corr = hands_data['right']['spearman_corr']
            if not (pd.isna(left_corr) or pd.isna(right_corr)):
                left_spearman_corrs.append(left_corr)
                right_spearman_corrs.append(right_corr)

    left_data_points = len(left_spearman_corrs)
    right_data_points = len(right_spearman_corrs)

    # Perform Wilcoxon signed-rank test for Spearman correlations
    stat_corrs, p_value_corrs = wilcoxon(left_spearman_corrs, right_spearman_corrs)

    results = {
        "spearman_correlations": {"statistic": stat_corrs, "p_value": p_value_corrs},
        "data_points": {"left": left_data_points, "right": right_data_points}
    }

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results Compare Left And Right:")
    print(f"Spearman Correlations: Statistic = {stat_corrs:.2f}, P-value = {p_value_corrs:.4f}")
    print(f"Data Points: Left Hand = {left_data_points}, Right Hand = {right_data_points}")

    return results

# -------------------------------------------------------------------------------------------------------------------
def Check_SAT_in_trials_mean_median_of_reach_indices(updated_metrics, sample_subject, metric_x, metric_y, stat_type="median"):
    """
    Analyzes reach indices for a given subject and metric, including box plots, scatter plots, 
    Spearman correlation calculations, and statistical tests.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        sample_subject (str): Subject identifier.
        metric_x (str): Metric for x-axis (e.g., 'durations').
        metric_y (str): Metric for y-axis (e.g., 'distance').
        stat_type (str): Type of statistics to use ("mean" or "median").

    Returns:
        dict: Results containing Spearman correlations, p-values, and statistical test results.
    """
    # Box plot for all reach indices with median and IQR
    boxplot_trials_mean_median_of_reach_indices(updated_metrics, sample_subject, 'right', metric_y)
    boxplot_trials_mean_median_of_reach_indices(updated_metrics, sample_subject, 'left', metric_y)

    # Calculate average and median metrics for all reach indices for each reach_index for all subjects and hands across trials
    mean_stats, median_stats = calculate_trials_mean_median_of_reach_indices(updated_metrics, metric_x, metric_y)

    # Overlay scatter plots for all reach indices
    plot_trials_mean_median_of_reach_indices(median_stats, sample_subject, 'right', metric_x, metric_y, stat_type=stat_type, use_unique_colors=True)
    plot_trials_mean_median_of_reach_indices(median_stats, sample_subject, 'left', metric_x, metric_y, stat_type=stat_type, use_unique_colors=True)

    # Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters
    results = calculate_duration_distance_trials_mean_median_of_reach_indices(median_stats, stat_type=stat_type)

    # Plot column chart of Spearman correlations and p-values
    plot_column_spearman_corr_pvalues_trials_mean_median_of_reach_indices(results)

    # Plot histograms for Spearman correlations and p-values
    plot_histogram_spearman_corr_with_stats_trials_mean_median_of_reach_indices(results)

    # Perform Wilcoxon signed-rank test on Spearman correlation values
    wilcoxon_results_by_hand_trials_mean_median_of_reach_indices = wilcoxon_test_spearman_corr_separate(results)

    # Compare left vs right hands at the subject level using Wilcoxon signed-rank test on paired Spearman correlations
    wilcoxon_results_compare_hand_trials_mean_median_of_reach_indices = compare_left_vs_right_hands_trials_mean_median_of_reach_indices(results)

    return {
        "results": results,
        "wilcoxon_results_by_hand": wilcoxon_results_by_hand_trials_mean_median_of_reach_indices,
        "wilcoxon_results_compare_hands": wilcoxon_results_compare_hand_trials_mean_median_of_reach_indices
    }

# -------------------------------------------------------------------------------------------------------------------
# 1.3	Within one reach location, is there still a speed–accuracy trade-off across repetitions?
# Calculate median, mean, standard deviation, max, and min for durations and distances for each subject
def calculate_statistics_for_subjects(updated_metrics):
    """
    Calculates median, mean, standard deviation, max, and min for durations and distances for each subject.

    Parameters:
        updated_metrics (dict): Updated metrics data.

    Returns:
        dict: Dictionary containing statistics for each subject.
    """
    statistics = {}

    for subject, hands_data in updated_metrics.items():
        all_durations = []
        all_distances = []

        for hand, metrics in hands_data.items():
            for trial, durations in metrics['durations'].items():
                all_durations.extend([dur for dur in durations if not pd.isna(dur)])
            for trial, distances in metrics['distance'].items():
                all_distances.extend([dist for dist in distances if not pd.isna(dist)])

        # Calculate statistics
        statistics[subject] = {
            'median_durations': np.median(all_durations) if all_durations else np.nan,
            'mean_durations': np.mean(all_durations) if all_durations else np.nan,
            'std_durations': np.std(all_durations) if all_durations else np.nan,
            'max_durations': np.max(all_durations) if all_durations else np.nan,
            'min_durations': np.min(all_durations) if all_durations else np.nan,
            'median_distance': np.median(all_distances) if all_distances else np.nan,
            'mean_distance': np.mean(all_distances) if all_distances else np.nan,
            'std_distance': np.std(all_distances) if all_distances else np.nan,
            'max_distance': np.max(all_distances) if all_distances else np.nan,
            'min_distance': np.min(all_distances) if all_distances else np.nan,
        }

    return statistics

# Scatter plot for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation
def scatter_by_reach_indices(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics, show_all=True):
    """
    Plots scatter plots for all reach indices as subplots in a 4x4 layout with optional hyperbolic regression and Spearman correlation.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        subject_statistics (dict): Statistics containing max durations and max distances for the subject.
        show_all (bool): If True, shows hyperbolic regression, intersections, and diagonal line. If False, only shows scatter and Spearman correlation.
    """
    # Create subplots in a 4x4 layout
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes = axes.flatten()

    # Get max duration and max distance for the subject
    max_duration = subject_statistics[subject]['max_durations']
    min_duration = subject_statistics[subject]['min_durations']
    max_distance = subject_statistics[subject]['max_distance']
    min_distance = subject_statistics[subject]['min_distance']

    for reach_index in range(16):
        x_values = []
        y_values = []
        trial_colors = []

        trials = updated_metrics[subject][hand][metric_x].keys()
        color_palette = sns.color_palette("Reds", len(trials))  # Generate a color palette from light to dark

        for i, trial in enumerate(trials):
            trial_x = np.array(updated_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(updated_metrics[subject][hand][metric_y][trial])

            # Collect data for the specified reach index
            if reach_index < len(trial_x) and reach_index < len(trial_y):
                x_values.append(trial_x[reach_index])
                y_values.append(trial_y[reach_index])
                trial_colors.append(color_palette[i])  # Assign color based on trial index
        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        trial_colors = np.array(trial_colors)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]
        trial_colors = trial_colors[valid_indices]

        # Calculate Spearman correlation
        if len(x_values) > 1 and len(y_values) > 1:
            spearman_corr, p_value = spearmanr(x_values, y_values)
        else:
            spearman_corr, p_value = np.nan, np.nan

        # Scatter plot for the current reach index
        ax = axes[reach_index]
        ax.scatter(x_values, y_values, c=trial_colors, alpha=0.7)
        ax.set_title(f"Reach Index {reach_index}")
        labels = {
            'distance': "Good → Bad (cm)",
            'accuracy': "Bad → Good (%)",
            'durations': "Good / Fast → Bad / Slow (s)",
            'speed': "Bad / Slow → Good / Fast (1/s)"
        }
        ax.set_xlabel(f"{metric_x.capitalize()} ({labels.get(metric_x, '')})")
        if reach_index % cols == 0:
            ax.set_ylabel(f"{metric_y.capitalize()} ({labels.get(metric_y, '')})")

        # Set x and y limits for each subplot
        ax.set_xlim(min_duration * 0.9, max_duration * 1.1)
        ax.set_ylim(min_distance * 0.9, max_distance * 1.1)

        if show_all and metric_x == 'durations' and metric_y == 'distance':
            # Perform hyperbolic regression
            def hyperbolic_func(x, a, b):
                return a / (x + b)

            try:
                params, _ = curve_fit(hyperbolic_func, x_values, y_values)
                a, b = params
                x_fit = np.linspace(min(x_values), max(x_values), 500)
                y_fit = hyperbolic_func(x_fit, *params)
                ax.plot(x_fit, y_fit, color='blue', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")

                # ---- Find intersection with diagonal line ----
                A = max_distance / max_duration
                B = A * b
                C = -a

                discriminant = B**2 - 4*A*C
                if discriminant >= 0:
                    x_roots = [(-B + np.sqrt(discriminant)) / (2*A),
                               (-B - np.sqrt(discriminant)) / (2*A)]
                    intersections = [(x, A*x) for x in x_roots if x > 0]

                    # Plot intersections
                    for (xi, yi) in intersections:
                        ax.scatter(xi, yi, color="black", s=50, zorder=5, label="Intersection")
                        ax.text(xi, yi, f"({xi:.2f}, {yi:.2f})",
                                fontsize=12, color="black", ha="left", va="bottom")
            except Exception as e:
                print(f"Hyperbolic regression failed for Reach Index {reach_index}: {e}")

            # Add diagonal line from (0, 0) to max duration and max distance
            if not np.isnan(max_duration) and not np.isnan(max_distance):
                ax.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")


        # Add Spearman correlation to the plot
        ax.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Hide unused subplots
    for i in range(16, len(axes)):
        axes[i].axis('off')

    # Add a title for the entire figure
    fig.suptitle(f"{subject} - {hand.capitalize()}", fontsize=16, y=1)

    plt.tight_layout()
    plt.show()

# Overlay 16 hyperbolic regressions, intersections, and diagonal line in one figure
def overlay_hyperbolic_regressions_reach_indices(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
    """
    Overlays 16 hyperbolic regressions, intersections, and diagonal line in one figure.
    Calculates and returns Spearman correlation, p-values, and intersections.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        subject_statistics (dict): Statistics containing max durations and max distances for the subject.

    Returns:
        list: Results containing Spearman correlation, p-values, and intersections for each reach index.
    """
    # Initialize results storage
    results = []

    # Get max duration and max distance for the subject
    max_duration = subject_statistics[subject]['max_durations']
    max_distance = subject_statistics[subject]['max_distance']

    # Create a figure
    plt.figure(figsize=(6, 6))

    # Generate a color palette from coolwarm
    color_palette = sns.color_palette("coolwarm", 16)

    for reach_index in range(16):
        x_values = []
        y_values = []

        trials = updated_metrics[subject][hand][metric_x].keys()

        for trial in trials:
            trial_x = np.array(updated_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(updated_metrics[subject][hand][metric_y][trial])

            # Collect data for the specified reach index
            if reach_index < len(trial_x) and reach_index < len(trial_y):
                x_values.append(trial_x[reach_index])
                y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Calculate Spearman correlation
        if len(x_values) > 1 and len(y_values) > 1:
            spearman_corr, p_value = spearmanr(x_values, y_values)
        else:
            spearman_corr, p_value = np.nan, np.nan

        # Perform hyperbolic regression
        def hyperbolic_func(x, a, b):
            return a / (x + b)

        try:
            params, _ = curve_fit(hyperbolic_func, x_values, y_values)
            a, b = params
            x_fit = np.linspace(min(x_values), max(x_values), 500)
            y_fit = hyperbolic_func(x_fit, *params)
            plt.plot(x_fit, y_fit, label=f"Reach {reach_index}", color=color_palette[reach_index], alpha=0.7)

            # ---- Find intersection with diagonal line ----
            A = max_distance / max_duration
            B = A * b
            C = -a

            discriminant = B**2 - 4 * A * C
            intersections = []
            if discriminant >= 0:
                x_roots = [(-B + np.sqrt(discriminant)) / (2 * A),
                           (-B - np.sqrt(discriminant)) / (2 * A)]
                intersections = [(x, A * x) for x in x_roots if x > 0]

                # Plot intersections
                for idx, (xi, yi) in enumerate(intersections):
                    plt.scatter(xi, yi, color=color_palette[reach_index], edgecolors='black', s=50, zorder=5)
                    # plt.text(xi, yi, f"({xi:.2f}, {yi:.2f})", fontsize=8, color=color_palette[reach_index], ha="right", va="bottom")

            # Save results for this reach index
            results.append({
                "reach_index": reach_index,
                "spearman_corr": spearman_corr,
                "p_value": p_value,
                "intersections": intersections
            })

        except Exception as e:
            print(f"Hyperbolic regression failed for Reach Index {reach_index}: {e}")
            results.append({
                "reach_index": reach_index,
                "spearman_corr": spearman_corr,
                "p_value": p_value,
                "intersections": None
            })

    # Add diagonal line from (0, 0) to max duration and max distance
    if not np.isnan(max_duration) and not np.isnan(max_distance):
        plt.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")
    
    plt.title(f"Overlay of Hyperbolic Regressions and Intersections ({subject}, {hand.capitalize()})")
    plt.xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    plt.ylabel(f"{metric_y.capitalize()} (Bad → Good)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return results

# Calculate and return Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject, hand, and reach index
def calculate_duration_distance_reach_indices(updated_metrics):
    """
    Calculates Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b)
    for durations vs distances for each subject, hand, and reach index.

    Parameters:
        updated_metrics (dict): Updated metrics data.

    Returns:
        dict: Dictionary containing results for each subject, hand, and reach index.
    """
    results = {}

    for subject in updated_metrics.keys():
        results[subject] = {}
        for hand in ['left', 'right']:
            results[subject][hand] = {}
            for reach_index in range(16):
                x_values = []
                y_values = []

                trials = updated_metrics[subject][hand]['durations'].keys()

                for trial in trials:
                    trial_x = np.array(updated_metrics[subject][hand]['durations'][trial])
                    trial_y = np.array(updated_metrics[subject][hand]['distance'][trial])

                    if reach_index < len(trial_x) and reach_index < len(trial_y):
                        x_values.append(trial_x[reach_index])
                        y_values.append(trial_y[reach_index])

                # Remove NaN values
                x_values = np.array(x_values)
                y_values = np.array(y_values)
                valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[valid_indices]
                y_values = y_values[valid_indices]

                # Calculate Spearman correlation
                if len(x_values) > 1 and len(y_values) > 1:
                    spearman_corr, p_value = spearmanr(x_values, y_values)
                else:
                    spearman_corr, p_value = np.nan, np.nan

                # Fit a hyperbolic curve
                def hyperbolic_func(x, a, b):
                    return a / (x + b)

                try:
                    params, _ = curve_fit(hyperbolic_func, x_values, y_values)
                    a, b = params
                except Exception:
                    a, b = np.nan, np.nan

                # Store results
                results[subject][hand][reach_index] = {
                    "spearman_corr": spearman_corr,
                    "p_value": p_value,
                    "data_points": len(x_values),
                    "hyperbolic_fit_a": a,
                    "hyperbolic_fit_b": b
                }

    return results

# Plot heatmap for Spearman correlations for each reach indices (0 to 15) for each subject and hand
def heatmap_spearman_correlation_reach_indices(results, hand):
    """
    Plots a heatmap of Spearman correlations for the specified hand (right or left).
    X-axis represents reach indices (0 to 15), and Y-axis represents subjects.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
        hand (str): Hand to plot ('right' or 'left').
    """
    subjects = list(results.keys())
    reach_indices = range(16)

    # Prepare data for the heatmap
    correlation_data = []
    for subject in subjects:
        if hand in results[subject]:
            correlations = [
                results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                for reach_index in reach_indices
            ]
            correlation_data.append(correlations)

    # Convert to DataFrame for heatmap
    correlation_df = pd.DataFrame(correlation_data, index=subjects, columns=reach_indices)

    # Plot heatmap
    plt.figure(figsize=(12, len(subjects) * 0.5))
    sns.heatmap(
        correlation_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        xticklabels=reach_indices,
        yticklabels=subjects,
        vmin=-1,
        vmax=1,
    )
    plt.title(f"Spearman Correlation Heatmap ({hand.capitalize()} Hand)")
    plt.xlabel("Reach Index")
    plt.ylabel("Subjects")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------------------------------
## ## correlations grouped by hand 
# -------------------------------------------------------------------------------------------------------------------
# Plot histogram of Spearman correlations for left and right hands, overlaying them
def plot_histogram_spearman_corr_reach_indices(results):
    """
    Plots a histogram of Spearman correlations for left and right hands, overlaying them in different colors.
    X-axis represents Spearman correlations, and Y-axis represents frequency.
    Displays the overall median for each hand.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
    """
    correlations_left = []
    correlations_right = []

    # Collect data for left and right hands
    for subject in results.keys():
        if 'left' in results[subject]:
            for reach_index in range(16):
                corr = results[subject]['left'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr):
                    correlations_left.append(corr)
        if 'right' in results[subject]:
            for reach_index in range(16):
                corr = results[subject]['right'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr):
                    correlations_right.append(corr)
    
    # print(f"Total Left Hand Correlations: {len(correlations_left)}")
    # print(f"Total Right Hand Correlations: {len(correlations_right)}")

    # Calculate overall medians
    overall_median_left = np.median(correlations_left) if correlations_left else np.nan
    overall_median_right = np.median(correlations_right) if correlations_right else np.nan

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')

    # Add median lines
    plt.axvline(overall_median_left, color='orange', linestyle='--', label=f'Median Left: {overall_median_left:.2f}')
    plt.axvline(overall_median_right, color='blue', linestyle='--', label=f'Median Right: {overall_median_right:.2f}')
    # plt.axvline(0, color='red', linestyle='--', label='Correlation = 0')  # Add line at correlation = 0

    # Calculate and display IQR for left and right hands
    iqr_left = np.percentile(correlations_left, 75) - np.percentile(correlations_left, 25)
    iqr_right = np.percentile(correlations_right, 75) - np.percentile(correlations_right, 25)
    plt.axvspan(np.percentile(correlations_left, 25), np.percentile(correlations_left, 75), color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
    plt.axvspan(np.percentile(correlations_right, 25), np.percentile(correlations_right, 75), color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")

    # Add labels, title, and legend
    plt.title("Spearman Correlation Histogram (Overlayed Hands)")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Perform Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately
def wilcoxon_test_spearman_corr_separate_reach_indices(results):
    """
    Performs the Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately.
    Also calculates the median for each hand.

    Parameters:
        results_all (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Test statistics, p-values, and medians for left and right hands.
    """
    correlations_left = []
    correlations_right = []

    for subject, hands_data in results.items():
        if 'left' in hands_data:
            for reach_index in range(16):
                corr = hands_data['left'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr):
                    correlations_left.append(corr)
        if 'right' in hands_data:
            for reach_index in range(16):
                corr = hands_data['right'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr):
                    correlations_right.append(corr)

    # Perform Wilcoxon signed-rank test for left hand
    stat_left, p_value_left = wilcoxon(correlations_left)

    # Perform Wilcoxon signed-rank test for right hand
    stat_right, p_value_right = wilcoxon(correlations_right)

    # Calculate medians
    median_left = np.median(correlations_left) if correlations_left else np.nan
    median_right = np.median(correlations_right) if correlations_right else np.nan

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results:")
    print(f"Left Hand: Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}, Data Points = {len(correlations_left)}")
    print(f"Right Hand: Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}, Data Points = {len(correlations_right)}")

    return {
        "left": {"statistic": stat_left, "p_value": p_value_left, "median": median_left},
        "right": {"statistic": stat_right, "p_value": p_value_right, "median": median_right}
    }

# Compare left vs right hands at the subject level using Wilcoxon signed-rank test on Spearman correlation values for durations and distances for each subject each reach index
def compare_left_vs_right_hands_reach_indices(results):
    """
    Compares the left and right hands at the subject level using the Wilcoxon signed-rank test
    on paired subject Spearman correlations for durations and distances.

    Parameters:
        results (dict): Dictionary containing Spearman correlation results for each subject.

    Returns:
        dict: Results of the Wilcoxon signed-rank test for durations and distances.
    """
    left_spearman_corrs = []
    right_spearman_corrs = []

    for subject, hands_data in results.items():
        if 'left' in hands_data and 'right' in hands_data:
            left_corrs = [hands_data['left'][reach_index]['spearman_corr'] for reach_index in hands_data['left'] if not pd.isna(hands_data['left'][reach_index]['spearman_corr'])]
            right_corrs = [hands_data['right'][reach_index]['spearman_corr'] for reach_index in hands_data['right'] if not pd.isna(hands_data['right'][reach_index]['spearman_corr'])]
            
            if len(left_corrs) == len(right_corrs) and len(left_corrs) > 0:
                left_spearman_corrs.extend(left_corrs)
                right_spearman_corrs.extend(right_corrs)

    # Perform Wilcoxon signed-rank test for Spearman correlations
    stat_corrs, p_value_corrs = wilcoxon(left_spearman_corrs, right_spearman_corrs)

    results = {
        "spearman_correlations": {"statistic": stat_corrs, "p_value": p_value_corrs},
        "data_points": {"left": len(left_spearman_corrs), "right": len(right_spearman_corrs)}
    }

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results Compare Left And Right:")
    print(f"Spearman Correlations: Statistic = {stat_corrs:.2f}, P-value = {p_value_corrs:.4f}")
    print(f"Data Points: Left Hand = {len(left_spearman_corrs)}, Right Hand = {len(right_spearman_corrs)}")

    return results

# -------------------------------------------------------------------------------------------------------------------
## correlations grouped by hand by index
# -------------------------------------------------------------------------------------------------------------------
# Plot box plot for Spearman correlations for each reach indices (0 to 15) for each subject and hand
def plot_correlation_boxplot_reach_indices_by_index(results, hand):
    """
    Plots a box plot of Spearman correlations for each reach index, rotated 90 degrees to the left.
    Y-axis represents reach indices (0 to 15), and X-axis represents Spearman correlations for each subject.
    Displays the median value for each reach index and the overall median across all reach types.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
        hand (str): Hand to plot ('right' or 'left').
    """
    reach_indices = range(16)
    correlation_data = {reach_index: [] for reach_index in reach_indices}

    # Collect data for each reach index
    for subject in results.keys():
        if hand in results[subject]:
            for reach_index in reach_indices:
                corr = results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr):
                    correlation_data[reach_index].append(corr)

    # Convert to DataFrame for box plot
    correlation_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in correlation_data.items()]))

    # Calculate overall median across all reach types
    overall_median = correlation_df.median().median()

    # Plot box plot (rotated)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=correlation_df, palette="coolwarm", orient="h", showmeans=False, width=0.6)
    plt.axvline(0, color='red', linestyle='--', label='Correlation = 0')  # Add vertical line at correlation 0
    plt.axvline(overall_median, color='purple', linestyle='--', label=f'Overall Median: {overall_median:.2f}')  # Add overall median line
    plt.title(f"Spearman Correlation Box Plot ({hand.capitalize()} Hand)")
    plt.ylabel("Reach Index")
    plt.xlabel("Spearman Correlation")
    plt.grid(alpha=0.5)

    # Add median values to the plot
    medians = correlation_df.median()
    for i, median in enumerate(medians):
        plt.text(median, i, f"{median:.2f}", verticalalignment='center', color='black', fontsize=8, horizontalalignment='right')

    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot histograms for median Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
def plot_histogram_spearman_corr_with_stats_reach_indices_by_index(results):
    """
    Plots histograms of median Spearman correlations for each reach index,
    overlaying left and right hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
    """
    median_correlations_left = []
    median_correlations_right = []

    # Collect median correlation for each reach index
    for reach_index in range(16):
        correlations_left = []
        correlations_right = []
        for subject in results.keys():
            if 'left' in results[subject]:
                corr_left = results[subject]['left'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr_left):
                    correlations_left.append(corr_left)
            if 'right' in results[subject]:
                corr_right = results[subject]['right'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr_right):
                    correlations_right.append(corr_right)
        if correlations_left:
            median_correlations_left.append(np.median(correlations_left))
        if correlations_right:
            median_correlations_right.append(np.median(correlations_right))

    # Calculate statistics
    median_left = np.median(median_correlations_left)
    iqr_left = np.percentile(median_correlations_left, 75) - np.percentile(median_correlations_left, 25)
    q1_left = np.percentile(median_correlations_left, 25)
    q3_left = np.percentile(median_correlations_left, 75)

    median_right = np.median(median_correlations_right)
    iqr_right = np.percentile(median_correlations_right, 75) - np.percentile(median_correlations_right, 25)
    q1_right = np.percentile(median_correlations_right, 25)
    q3_right = np.percentile(median_correlations_right, 75)

    # Perform Wilcoxon signed-rank test for each hand separately
    stat_left, p_value_left = wilcoxon(median_correlations_left)
    stat_right, p_value_right = wilcoxon(median_correlations_right)

    # Plot histogram for median Spearman correlations
    plt.figure(figsize=(8, 6))
    plt.hist(median_correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(median_correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
    plt.axvline(median_left, color='orange', linestyle='--', label=f"Median Left: {median_left:.2f}")
    plt.axvline(median_right, color='blue', linestyle='--', label=f"Median Right: {median_right:.2f}")
    plt.axvspan(q1_left, q3_left, color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
    plt.axvspan(q1_right, q3_right, color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")
    plt.title("Histogram of Median Spearman Correlations by Hand")
    plt.xlabel("Median Spearman Correlation")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # # Print statistics
    # print(f"Left Hand: Median = {median_left:.2f}, IQR = {iqr_left:.2f}")
    # print(f"Right Hand: Median = {median_right:.2f}, IQR = {iqr_right:.2f}")
    # print(f"Wilcoxon Signed-Rank Test (Left Hand): Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}")
    # print(f"Wilcoxon Signed-Rank Test (Right Hand): Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}")

# Perform Wilcoxon signed-rank test on median Spearman correlation values for left and right hands separately
def wilcoxon_test_spearman_corr_separate_reach_indices_by_index(results):
    """
    Performs the Wilcoxon signed-rank test on median Spearman correlation values for left and right hands separately.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Test statistics, p-values, and median statistics for left and right hands.
    """
    median_correlations_left = []
    median_correlations_right = []

    # Collect median correlation for each reach index
    for reach_index in range(16):
        correlations_left = []
        correlations_right = []
        for subject in results.keys():
            if 'left' in results[subject]:
                corr_left = results[subject]['left'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr_left):
                    correlations_left.append(corr_left)
            if 'right' in results[subject]:
                corr_right = results[subject]['right'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr_right):
                    correlations_right.append(corr_right)
        if correlations_left:
            median_correlations_left.append(np.median(correlations_left))
        if correlations_right:
            median_correlations_right.append(np.median(correlations_right))


    # Calculate statistics
    median_left = np.median(median_correlations_left)
    iqr_left = np.percentile(median_correlations_left, 75) - np.percentile(median_correlations_left, 25)
    q1_left = np.percentile(median_correlations_left, 25)
    q3_left = np.percentile(median_correlations_left, 75)

    median_right = np.median(median_correlations_right)
    iqr_right = np.percentile(median_correlations_right, 75) - np.percentile(median_correlations_right, 25)
    q1_right = np.percentile(median_correlations_right, 25)
    q3_right = np.percentile(median_correlations_right, 75)

    # Perform Wilcoxon signed-rank test for each hand separately
    stat_left, p_value_left = wilcoxon(median_correlations_left)
    stat_right, p_value_right = wilcoxon(median_correlations_right)

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results:")
    print(f"Left Hand: Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}, Data Points = {len(median_correlations_left)}")
    print(f"Right Hand: Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}, Data Points = {len(median_correlations_right)}")

    return {
        "left": {
            "statistic": stat_left,
            "p_value": p_value_left,
            "median": median_left,
            "iqr": iqr_left,
            "q1": q1_left,
            "q3": q3_left,
        },
        "right": {
            "statistic": stat_right,
            "p_value": p_value_right,
            "median": median_right,
            "iqr": iqr_right,
            "q1": q1_right,
            "q3": q3_right,
        },
    }

# Compare left vs right hands at the subject level using Wilcoxon signed-rank test on median Spearman correlations for durations and distances for each reach index
def compare_left_vs_right_hands_reach_indices_by_index(results):
    """
    Compares the left and right hands at the subject level using the Wilcoxon signed-rank test
    on median Spearman correlations for durations and distances.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Results of the Wilcoxon signed-rank test, median correlations, and data points for left and right hands.
    """
    median_correlations_left = []
    median_correlations_right = []


    # Collect median correlation for each reach index
    for reach_index in range(16):
        correlations_left = []
        correlations_right = []
        for subject in results.keys():
            if 'left' in results[subject]:
                corr_left = results[subject]['left'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr_left):
                    correlations_left.append(corr_left)
            if 'right' in results[subject]:
                corr_right = results[subject]['right'].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr_right):
                    correlations_right.append(corr_right)
        if correlations_left:
            median_correlations_left.append(np.median(correlations_left))
        if correlations_right:
            median_correlations_right.append(np.median(correlations_right))

    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(median_correlations_left, median_correlations_right)

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results Compare Left And Right:")
    print(f"Spearman Correlations: Statistic = {stat:.2f}, P-value = {p_value:.4f}")
    print(f"Data Points: Left Hand = {len(median_correlations_left)}, Right Hand = {len(median_correlations_right)}")

    return {
        "median_correlations_left": median_correlations_left,
        "median_correlations_right": median_correlations_right,
        "data_points_left": len(median_correlations_left),
        "data_points_right": len(median_correlations_right),
        "wilcoxon_statistic": stat,
        "wilcoxon_p_value": p_value
    }

# -------------------------------------------------------------------------------------------------------------------
## correlations grouped by hand by subject 
# -------------------------------------------------------------------------------------------------------------------
# Box plot of Spearman correlations for each subject
def plot_correlation_boxplot_reach_indices_by_subject(results, hand):
    """
    Plots a box plot of Spearman correlations for each subject.
    X-axis represents subjects, and Y-axis represents Spearman correlations across all reach indices.
    Displays the median value for each subject and the overall median.

    Parameters:
        results_all (dict): Results containing Spearman correlations for each subject and hand.
        hand (str): Hand to plot ('right' or 'left').
    """
    subjects = list(results.keys())
    correlation_data = {subject: [] for subject in subjects}

    # Collect data for each subject
    for subject in results.keys():
        if hand in results[subject]:
            for reach_index in range(16):
                corr = results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                if not np.isnan(corr):
                    correlation_data[subject].append(corr)

    # Convert to DataFrame for box plot
    correlation_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in correlation_data.items()]))

    # Calculate overall median
    # overall_median = correlation_df.stack().median()
    overall_median = correlation_df.median().median()


    # Plot box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=correlation_df, palette="coolwarm", orient="h", showmeans=False, width=0.6)
    plt.axvline(0, color='red', linestyle='--', label='Correlation = 0')  # Add vertical line at correlation 0
    plt.axvline(overall_median, color='purple', linestyle='--', label=f'Overall Median: {overall_median:.2f}')  # Add overall median line
    plt.title(f"Spearman Correlation Box Plot ({hand.capitalize()} Hand)")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Subjects")
    plt.yticks(rotation=0)
    plt.grid(alpha=0.5)

    # Add median values to the plot
    medians = correlation_df.median()
    for i, median in enumerate(medians):
        plt.text(median, i, f"{median:.2f}", verticalalignment='center', color='black', fontsize=8, horizontalalignment='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot histograms for median Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
def plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(results):
    """
    Plots histograms of median Spearman correlations for each subject,
    overlaying left and right hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
    """
    median_correlations_left = []
    median_correlations_right = []

    # Collect median correlation for each subject
    for subject in results.keys():
        for hand in ['left', 'right']:
            if hand in results[subject]:
                correlations = [
                    results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                    for reach_index in range(16)
                ]
                correlations = [corr for corr in correlations if not np.isnan(corr)]
                if correlations:
                    if hand == 'left':
                        median_correlations_left.append(np.median(correlations))
                    elif hand == 'right':
                        median_correlations_right.append(np.median(correlations))


    # Calculate statistics
    median_left = np.median(median_correlations_left)
    iqr_left = np.percentile(median_correlations_left, 75) - np.percentile(median_correlations_left, 25)
    q1_left = np.percentile(median_correlations_left, 25)
    q3_left = np.percentile(median_correlations_left, 75)

    median_right = np.median(median_correlations_right)
    iqr_right = np.percentile(median_correlations_right, 75) - np.percentile(median_correlations_right, 25)
    q1_right = np.percentile(median_correlations_right, 25)
    q3_right = np.percentile(median_correlations_right, 75)

    # Perform Wilcoxon signed-rank test for each hand separately
    stat_left, p_value_left = wilcoxon(median_correlations_left)
    stat_right, p_value_right = wilcoxon(median_correlations_right)

    # Plot histogram for median Spearman correlations
    plt.figure(figsize=(8, 6))
    plt.hist(median_correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(median_correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
    plt.axvline(median_left, color='orange', linestyle='--', label=f"Median Left: {median_left:.2f}")
    plt.axvline(median_right, color='blue', linestyle='--', label=f"Median Right: {median_right:.2f}")
    plt.axvspan(q1_left, q3_left, color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
    plt.axvspan(q1_right, q3_right, color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")
    plt.title("Histogram of Median Spearman Correlations by Hand")
    plt.xlabel("Median Spearman Correlation")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # # Print statistics
    # print(f"Left Hand: Median = {median_left:.2f}, IQR = {iqr_left:.2f}")
    # print(f"Right Hand: Median = {median_right:.2f}, IQR = {iqr_right:.2f}")
    # print(f"Wilcoxon Signed-Rank Test (Left Hand): Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}")
    # print(f"Wilcoxon Signed-Rank Test (Right Hand): Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}")

# Perform Wilcoxon signed-rank test on median Spearman correlation values for left and right hands separately
def wilcoxon_test_spearman_corr_separate_reach_indices_by_subject(results):
    """
    Performs the Wilcoxon signed-rank test on median Spearman correlation values for left and right hands separately.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Test statistics, p-values, and median statistics for left and right hands.
    """
    median_correlations_left = []
    median_correlations_right = []

    # Collect median correlation for each subject
    for subject in results.keys():
        for hand in ['left', 'right']:
            if hand in results[subject]:
                correlations = [
                    results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                    for reach_index in range(16)
                ]
                correlations = [corr for corr in correlations if not np.isnan(corr)]
                if correlations:
                    if hand == 'left':
                        median_correlations_left.append(np.median(correlations))
                    elif hand == 'right':
                        median_correlations_right.append(np.median(correlations))

    # Calculate statistics
    median_left = np.median(median_correlations_left)
    iqr_left = np.percentile(median_correlations_left, 75) - np.percentile(median_correlations_left, 25)
    q1_left = np.percentile(median_correlations_left, 25)
    q3_left = np.percentile(median_correlations_left, 75)

    median_right = np.median(median_correlations_right)
    iqr_right = np.percentile(median_correlations_right, 75) - np.percentile(median_correlations_right, 25)
    q1_right = np.percentile(median_correlations_right, 25)
    q3_right = np.percentile(median_correlations_right, 75)

    # Perform Wilcoxon signed-rank test for each hand separately
    stat_left, p_value_left = wilcoxon(median_correlations_left)
    stat_right, p_value_right = wilcoxon(median_correlations_right)

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results:")
    print(f"Left Hand: Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}, Data Points = {len(median_correlations_left)}")
    print(f"Right Hand: Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}, Data Points = {len(median_correlations_right)}")

    return {
        "left": {
            "statistic": stat_left,
            "p_value": p_value_left,
            "median": median_left,
            "iqr": iqr_left,
            "q1": q1_left,
            "q3": q3_left,
        },
        "right": {
            "statistic": stat_right,
            "p_value": p_value_right,
            "median": median_right,
            "iqr": iqr_right,
            "q1": q1_right,
            "q3": q3_right,
        },
    }

# Compare left vs right hands using Wilcoxon signed-rank test on median Spearman correlations
def compare_left_vs_right_hands_reach_indices_by_subject(results):
    """
    Compares the left and right hands using the Wilcoxon signed-rank test
    on median Spearman correlations for durations and distances.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Results of the Wilcoxon signed-rank test, median correlations, and data points for left and right hands.
    """
    median_correlations_left = []
    median_correlations_right = []

    # Collect median correlation for each subject
    for subject in results.keys():
        for hand in ['left', 'right']:
            if hand in results[subject]:
                correlations = [
                    results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                    for reach_index in range(16)
                ]
                correlations = [corr for corr in correlations if not np.isnan(corr)]
                if correlations:
                    if hand == 'left':
                        median_correlations_left.append(np.median(correlations))
                    elif hand == 'right':
                        median_correlations_right.append(np.median(correlations))

    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(median_correlations_left, median_correlations_right)

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results Compare Left And Right:")
    print(f"Spearman Correlations: Statistic = {stat:.2f}, P-value = {p_value:.4f}")
    print(f"Data Points: Left Hand = {len(median_correlations_left)}, Right Hand = {len(median_correlations_right)}")

    return {
        "median_correlations_left": median_correlations_left,
        "median_correlations_right": median_correlations_right,
        "data_points_left": len(median_correlations_left),
        "data_points_right": len(median_correlations_right),
        "wilcoxon_statistic": stat,
        "wilcoxon_p_value": p_value
    }

# -------------------------------------------------------------------------------------------------------------------

# 1.3	Within one reach location, is there still a speed–accuracy trade-off across repetitions?

def Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics, sample_subject, grouping="hand", hyperbolic=False):
    """
    Analyze speed–accuracy trade-off in one reach location for a given subject by running scatter/hyperbolic
    regressions and computing correlations using one of three grouping methods.
    
    Parameters:
        updated_metrics (dict): The full updated metrics data.
        subject (str): Subject identifier (e.g. '07/22/HW').
        grouping (str): One of {"hand", "hand_by_index", "hand_by_subject"} to select the correlation grouping.
        hyperbolic (bool): If True, runs hyperbolic regressions (scatter overlays).
    
    Returns:
        dict: Dictionary of results from the chosen grouping including correlation results and statistics.
    """
    # 1. Compute subject statistics from durations and distances
    subject_statistics = calculate_statistics_for_subjects(updated_metrics)
    
    # 2. Create scatter plots (without hyperbolic regression) for left and right hands
    # (These functions display plots; you can comment them out if not needed)
    scatter_by_reach_indices(updated_metrics, sample_subject, 'right', 'durations', 'distance', subject_statistics, show_all=False)
    scatter_by_reach_indices(updated_metrics, sample_subject, 'left', 'durations', 'distance', subject_statistics, show_all=False)
    
    # 3. Optionally run overlay 16 hyperbolic regressions, intersections, and diagonal line in one figure
    if hyperbolic:
        R_hyper = overlay_hyperbolic_regressions_reach_indices(updated_metrics, sample_subject, 'right', 'durations', 'distance', subject_statistics)
        L_hyper = overlay_hyperbolic_regressions_reach_indices(updated_metrics, sample_subject, 'left', 'durations', 'distance', subject_statistics)
    else:
        R_hyper = L_hyper = None

    # 4. Calculate and return Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject, hand, and reach index
    results = calculate_duration_distance_reach_indices(updated_metrics)

    # Plot heatmap for Spearman correlations for each reach indices (0 to 15) for each subject and hand
    heatmap_spearman_correlation_reach_indices(results, "right")
    heatmap_spearman_correlation_reach_indices(results, "left")
    
    analysis = {}
    # 5. Run analysis based on grouping parameter
    if grouping == "hand":
        # Group correlations by hand
        # Plot histogram overlaying left and right hands
        plot_histogram_spearman_corr_reach_indices(results)
        # Run Wilcoxon tests by hand and compare hands at subject level
        wilcoxon_by_hand = wilcoxon_test_spearman_corr_separate_reach_indices(results)
        compare_hands = compare_left_vs_right_hands_reach_indices(results)
        analysis = {
            "wilcoxon_by_hand": wilcoxon_by_hand,
            "compare_hands": compare_hands
        }
    elif grouping == "hand_by_index":
        # Group correlations by reach index: boxplot, histogram and stats
        plot_correlation_boxplot_reach_indices_by_index(results, "right")
        plot_correlation_boxplot_reach_indices_by_index(results, "left")
        plot_histogram_spearman_corr_with_stats_reach_indices_by_index(results)
        wilcox_by_index = wilcoxon_test_spearman_corr_separate_reach_indices_by_index(results)
        compare_hands_index = compare_left_vs_right_hands_reach_indices_by_index(results)
        analysis = {
            "wilcox_by_index": wilcox_by_index,
            "compare_hands_index": compare_hands_index
        }
    elif grouping == "hand_by_subject":
        # Group correlations by subject: boxplot and histogram by subject
        plot_correlation_boxplot_reach_indices_by_subject(results, "right")
        plot_correlation_boxplot_reach_indices_by_subject(results, "left")
        plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(results)
        wilcox_by_subject = wilcoxon_test_spearman_corr_separate_reach_indices_by_subject(results)
        compare_hands_subject = compare_left_vs_right_hands_reach_indices_by_subject(results)
        analysis = {
            "wilcox_by_subject": wilcox_by_subject,
            "compare_hands_subject": compare_hands_subject
        }
    else:
        raise ValueError("grouping must be one of 'hand', 'hand_by_index', 'hand_by_subject'")
    
    return subject_statistics, results, analysis
