# Funtions 
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from matplotlib.patches import Rectangle


figuresize = (8, 6)
# # reachindexlabels = list(range(1, 17))
# # # Fixed colors assigned to specific reach indices.
# # color_groups = {
# #     0: 'black', 4: 'red', 8: 'red', 12: 'red',
# #     1: 'green', 5: 'green', 9: 'green', 13: 'green',
# #     2: 'blue', 6: 'blue', 10: 'blue', 14: 'blue',
# #     3: 'purple', 7: 'purple', 11: 'purple', 15: 'purple'
# # }

labelfontsize =24
tickfontsize =10
numbrticks =5
legendfontsize =16
# legend_position = {"loc": "upper left", "bbox_to_anchor": (0.01, 0.99)}
marker_size =150
handorder = ['non_dominant', 'dominant']
legendonoroff = False 
tittleononoroff = False

# Set global matplotlib and seaborn defaults
plt.rcParams['figure.figsize'] = figuresize
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = labelfontsize
plt.rcParams['xtick.labelsize'] = tickfontsize
plt.rcParams['ytick.labelsize'] = tickfontsize
plt.rcParams['legend.fontsize'] = legendfontsize
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['axes.grid'] = False

sns.set_style("whitegrid")



# # -------------------------------------------------------------------------------------------------------------------
# # PART 4: Data Analysis and Visualization
# # -------------------------------------------------------------------------------------------------------------------
# # 1.1	Does a subject show a speed–accuracy trade-off trial by trial?
# result_Check_SAT_in_trial_by_trial = utils6.Check_SAT_in_trial_by_trial(updated_metrics, All_dates, sample_subject= ['07/22/HW'], overlay_hands=False)

# # 1.2	Do reach types that are faster on average also tend to be less accurate on average?
# result_Check_SAT_in_trials_mean_median_of_reach_indices = utils6.Check_SAT_in_trials_mean_median_of_reach_indices(updated_metrics, '07/22/HW', 'durations', 'distance', stat_type="median")

# # 1.3	Within one reach location, is there still a speed–accuracy trade-off across repetitions?
# _, _, result_Check_SAT_in_reach_indices_by_hand, _ = utils6.Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics, '07/22/HW', grouping="hand", hyperbolic=False)
# _, _, result_Check_SAT_in_reach_indices_by_hand_by_index, _ = utils6.Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics, '07/22/HW', grouping="hand_by_index", hyperbolic=False)
# _, _, result_Check_SAT_in_reach_indices_by_hand_by_subject, heatmap_medians = utils6.Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics, '07/22/HW', grouping="hand_by_subject", hyperbolic=False)
# subject_statistics, _, _, _ = utils6.Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics, '07/22/HW', grouping="hand", hyperbolic=True)
# # -------------------------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------------------------------
# # Appendix
# utils6.scatter_plot_duration_distance_by_choice(updated_metrics, overlay_hands=False, selected_subjects=['07/22/HW'])
# utils6.scatter_plot_duration_distance_by_choice(updated_metrics, overlay_hands=False, selected_subjects=['07/22/HW'], special_indices=[0, 4, 8, 12])
# utils6.scatter_plot_duration_distance_by_choice(updated_metrics, overlay_hands=False, selected_subjects=['07/22/HW'], special_indices=[0])
# utils6.scatter_plot_duration_distance_by_choice(updated_metrics, overlay_hands=False, selected_subjects=['07/22/HW'], special_indices=[0], show_hyperbolic_fit=False, color_mode="uniform", show_median_overlay=False)
# # -------------------------------------------------------------------------------------------------------------------


def scatter_plot_duration_distance_by_choice(updated_metrics, overlay_hands=True, selected_subjects=None, 
                                               special_indices=None, show_hyperbolic_fit=True, color_mode="uniform",
                                               show_median_overlay=True):
    """
    Plots a scatter plot of all durations vs all distances across selected subjects, hands, and trials,
    and (optionally) fits a hyperbolic curve. Optionally overlays both hands or separates them.
    Data points can be colored with a light-to-dark gradient (from trial one to last) or in one fixed color.
    
    Parameters:
        updated_metrics (dict): Updated metrics data.
        overlay_hands (bool): If True, overlays both hands in the same plot. If False, separates them.
        selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
        special_indices (list or None): List of reach indices to include. If None, all indices are included.
        show_hyperbolic_fit (bool): If True, displays the hyperbolic fit.
        color_mode (str): One of {"gradient", "uniform"}; if "gradient" uses a colormap for trial order.
        show_median_overlay (bool): If True, overlays median values for each reach index if available.
    """
    def extract_data(metrics, hand, indices, color_mode):
        # For each reach index, collect durations, distances and the trial order (to color by trial order)
        durations = [[] for _ in range(16)]
        distances = [[] for _ in range(16)]
        trial_orders = [[] for _ in range(16)]
        trial_counter = 1
        # Loop in sorted order for consistency
        for trial in sorted(metrics[hand]['durations'].keys()):
            trial_durations = metrics[hand]['durations'][trial]
            trial_distances = metrics[hand]['distance'][trial]
            for reach_index, (duration, distance) in enumerate(zip(trial_durations, trial_distances)):
                if (indices is None or reach_index in indices):
                    if not pd.isna(duration) and not pd.isna(distance):
                        durations[reach_index].append(duration)
                        distances[reach_index].append(distance)
                        trial_orders[reach_index].append(trial_counter)
            trial_counter += 1
        return durations, distances, trial_orders

    def _plot_scatter_with_fit(durations, distances, trial_orders, title, special_indices=None, 
                               show_hyperbolic_fit=True, color_mode="uniform", show_median_overlay=True):
        """
        Helper function to plot scatter and (optionally) fit a hyperbolic curve.
        
        Parameters:
            durations (list of lists): Collected durations for each reach index.
            distances (list of lists): Collected distances for each reach index.
            trial_orders (list of lists): Trial order for each point (used if gradient coloring is chosen).
            title (str): Title of the plot.
            special_indices (list or None): List of reach indices to include. If None, all indices are used.
            show_hyperbolic_fit (bool): If True, displays the hyperbolic fit.
            color_mode (str): "gradient" or "uniform" for point colors.
            show_median_overlay (bool): If True, overlays median values for each reach index if available.
        """
        # Define fixed colors for each reach index if using uniform colors
        color_groups = {
            0: 'black', 4: 'red', 8: 'red', 12: 'red',
            1: 'green', 5: 'green', 9: 'green', 13: 'green',
            2: 'blue', 6: 'blue', 10: 'blue', 14: 'blue',
            3: 'purple', 7: 'black', 11: 'purple', 15: 'purple'
        }
        # Flatten lists for Spearman correlation calculation
        flat_durations = []
        flat_distances = []
        flat_trials = []
        for i, d_list in enumerate(durations):
            if special_indices is None or i in special_indices:
                for j, d in enumerate(d_list):
                    flat_durations.append(d)
                    flat_distances.append(distances[i][j])
                    flat_trials.append(trial_orders[i][j])
                    
        # Calculate Spearman correlation
        if len(flat_durations) > 1 and len(flat_distances) > 1:
            spearman_corr, p_value = spearmanr(flat_durations, flat_distances)
        else:
            spearman_corr, p_value = np.nan, np.nan
    
        plt.figure(figsize=(8, 6))
        # If using gradient colors, we will base the gradient on each reach index's fixed color
        # and vary its lightness from light (first trial) to dark (last trial)
        # For uniform mode, we use the fixed color directly.
        # Iterate over each reach index
        # Import mcolors for color conversion in gradient mode.
        import matplotlib.colors as mcolors
        for i in range(16):
            if special_indices is not None and i not in special_indices:
                continue
            for idx, d in enumerate(durations[i]):
                dist = distances[i][idx]
                if color_mode == "gradient":
                    base_color = color_groups.get(i, 'black')
                    trial_val = trial_orders[i][idx]
                    if flat_trials:
                        max_trial = max(flat_trials)
                    else:
                        max_trial = 1
                    # Normalize trial order so that first trial is 0 and last is 1
                    normalized = (trial_val - 1) / (max_trial - 1) if max_trial > 1 else 1

                    # Define a helper blend function
                    def blend(c1, c2, t):
                        return tuple((1-t)*c1_i + t*c2_i for c1_i, c2_i in zip(c1, c2))
                    base_rgb = mcolors.to_rgb(base_color)
                    # Create a light version of the base color by blending it with white (50% lighter)
                    light_variant = blend(base_rgb, (1, 1, 1), 0.5)
                    # Interpolate between the light variant and the base color based on normalized trial order:
                    color = blend(light_variant, base_rgb, normalized)
                    marker_size = 100
                else:
                    color = color_groups.get(i, 'black')
                    marker_size = 100
                # Label only once for certain indices
                if special_indices is None or len(special_indices) != 1:
                    label = f"{i+1}, {i+5}, {i+9}, {i+13}" if (i in [0, 1, 2, 3] and idx==0) else None 
                else:
                    label = f"Reach Index {i+1}" if idx == 0 else None # Label only once per reach index
                plt.scatter(d, dist, color=color, s=marker_size, alpha=0.8, label=label)
            # Overlay median values for each reach index if available and enabled
            if show_median_overlay and durations[i] and distances[i]:
                median_duration = np.median(durations[i])
                median_distance = np.median(distances[i])
                plt.scatter(median_duration, median_distance, facecolors='none', edgecolors='black', s=100, zorder=5)
                plt.text(median_duration, median_distance, f"{i}", fontsize=7, color='black', ha="center", va="center",
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle', linewidth=1.5))
    
        # plt.title(f"Scatter Plot of Durations vs Distances ({title})")
        labels = {
            'distance': "Good → Bad (cm)",
            'durations': "Good / Fast → Bad / Slow (s)"
        }
        plt.xlabel(f"Durations ({labels.get('durations', '')})")
        plt.ylabel(f"Distance ({labels.get('distance', '')})")
        plt.grid(alpha=0.5)
        plt.title(f"{title} - {'Overlayed Hands' if overlay_hands else f'{hand.capitalize()} Hand'}", fontsize=16)
        # Add Spearman correlation and number of data points to the plot
        plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.4f}\nP-value: {p_value:.4f}\nn: {len(flat_durations)}",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    
        # Option to fit a hyperbolic curve if requested
        if show_hyperbolic_fit and len(flat_durations) > 1:
            def hyperbolic_func(x, a, b):
                return a / (x + b)
            try:
                params, _ = curve_fit(hyperbolic_func, flat_durations, flat_distances)
                x_fit = np.linspace(min(flat_durations), max(flat_durations), 500)
                y_fit = hyperbolic_func(x_fit, *params)
                plt.plot(x_fit, y_fit, color='red', linestyle='--', 
                         label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
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
        all_trial_orders = [[] for _ in range(16)]
        for subject in selected_subjects:
            if subject in updated_metrics:
                for hand in ['non_dominant', 'dominant']:
                    d, dist, trials = extract_data(updated_metrics[subject], hand, special_indices, color_mode)
                    for i in range(16):
                        all_durations[i].extend(d[i])
                        all_distances[i].extend(dist[i])
                        all_trial_orders[i].extend(trials[i])
        _plot_scatter_with_fit(all_durations, all_distances, all_trial_orders, 
                               title=f"Overlayed Hands (Subjects: {', '.join(selected_subjects)})", 
                               special_indices=special_indices, show_hyperbolic_fit=show_hyperbolic_fit,
                               color_mode=color_mode, show_median_overlay=show_median_overlay)
    else:
        for hand in ['non_dominant', 'dominant']:
            all_durations = [[] for _ in range(16)]
            all_distances = [[] for _ in range(16)]
            all_trial_orders = [[] for _ in range(16)]
            for subject in selected_subjects:
                if subject in updated_metrics:
                    d, dist, trials = extract_data(updated_metrics[subject], hand, special_indices, color_mode)
                    for i in range(16):
                        all_durations[i].extend(d[i])
                        all_distances[i].extend(dist[i])
                        all_trial_orders[i].extend(trials[i])
            _plot_scatter_with_fit(all_durations, all_distances, all_trial_orders, 
                                   title=f"{hand.capitalize()} Hand (Subjects: {', '.join(selected_subjects)})",
                                   special_indices=special_indices, show_hyperbolic_fit=show_hyperbolic_fit,
                                   color_mode=color_mode, show_median_overlay=show_median_overlay)
# -------------------------------------------------------------------------------------------------------------------
# Do reach types that are faster on average also tend to be less accurate on average?
# -------------------------------------------------------------------------------------------------------------------

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
        hand (str): Hand ('non_dominant' or 'dominant').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        stat_type (str): Type of statistics to use ("mean" or "median").
        use_unique_colors (bool): If True, assigns a unique color to each reach index. Otherwise, groups by color.

    Returns:
        tuple: Spearman correlation and p-value for the overlayed points.
    """
    # Initialize the plot
    plt.figure(figsize=(8, 6))

    x_values = []
    y_values = []

    # Define color groups for reach indices
    if use_unique_colors:
        # color_palette = sns.color_palette("viridis", 16)
        # color_groups = {i: color_palette[i] for i in range(16)}
        color_groups = {i: 'black' for i in range(16)}
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
        # Overlay the points as empty markers with the reach index annotated inside
        if not np.isnan(duration) and not np.isnan(distance):
            x_values.append(duration)
            y_values.append(distance)
            color = color_groups[reach_index]
            label = f"Group {reach_index % 4}, {group_indexes[reach_index % 4]}" if not use_unique_colors and reach_index < 4 else None
            import matplotlib.colors as mcolors
            dark_color = tuple(c * 0.8 for c in mcolors.to_rgb(color))
            plt.scatter(duration, distance, facecolors='none', edgecolors=dark_color, s=270, label=label, zorder=5, alpha=1.0)
            # Annotate the reach index at the center of the empty marker
            plt.text(duration, distance, f"{reach_index+1}", fontsize=12, color=dark_color, ha="center", va="center")

    # Calculate Spearman correlation
    if len(x_values) > 1 and len(y_values) > 1:
        spearman_corr, p_value = spearmanr(x_values, y_values)
    else:
        spearman_corr, p_value = np.nan, np.nan
    
    # Add Spearman correlation and number of data points to the plot
    plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.4f}\nP-value: {p_value:.4f}\nn: 16",
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))


    plt.ylabel('Median distance : Good → Bad (cm)', fontsize=20)
    plt.xlabel('Median duration : Fast → Slow (s)', fontsize=20)
    plt.title(f"Overlay of Reach Statistics ({subject}, {hand.capitalize()}, {stat_type.capitalize()})\nSpearman Corr: {spearman_corr:.2f}, P-value: {p_value:.2f}", fontsize=16)
    # plt.legend(fontsize=20)
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
            for hand in ['non_dominant', 'dominant']:
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

# Plot histograms for Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
def plot_histogram_spearman_corr_with_stats_trials_mean_median_of_reach_indices(results):
    """
    Plots histograms of Spearman correlations for durations vs distances across all subjects,
    overlaying non_dominant and dominant hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
    """
    correlations_non_dominant = []
    correlations_dominant = []

    for subject, hands_data in results.items():
        for hand, metrics in hands_data.items():
            if 'spearman_corr' in metrics and not np.isnan(metrics['spearman_corr']):
                if hand == 'non_dominant':
                    correlations_non_dominant.append(metrics['spearman_corr'])
                elif hand == 'dominant':
                    correlations_dominant.append(metrics['spearman_corr'])

    # Calculate statistics
    median_non_dominant = np.median(correlations_non_dominant)
    iqr_non_dominant = np.percentile(correlations_non_dominant, 75) - np.percentile(correlations_non_dominant, 25)
    q1_non_dominant = np.percentile(correlations_non_dominant, 25)
    q3_non_dominant = np.percentile(correlations_non_dominant, 75)

    median_dominant = np.median(correlations_dominant)
    iqr_dominant = np.percentile(correlations_dominant, 75) - np.percentile(correlations_dominant, 25)
    q1_dominant = np.percentile(correlations_dominant, 25)
    q3_dominant = np.percentile(correlations_dominant, 75)

    # Perform Wilcoxon signed-rank test for each hand separately
    stat_non_dominant, p_value_non_dominant = wilcoxon(correlations_non_dominant)
    stat_dominant, p_value_dominant = wilcoxon(correlations_dominant)

    # Plot histogram for Spearman correlations
    plt.figure(figsize=(8, 6))
    plt.hist(correlations_non_dominant, bins=15, color='orange', alpha=0.7, edgecolor='black', label='non_dominant Hand')
    plt.hist(correlations_dominant, bins=15, color='blue', alpha=0.7, edgecolor='black', label='dominant Hand')
    plt.axvline(median_non_dominant, color='orange', linestyle='--', label=f"Median non_dominant: {median_non_dominant:.2f}")
    plt.axvline(median_dominant, color='blue', linestyle='--', label=f"Median dominant: {median_dominant:.2f}")
    plt.axvspan(q1_non_dominant, q3_non_dominant, color='orange', alpha=0.2, label=f"IQR non_dominant: {iqr_non_dominant:.2f}")
    plt.axvspan(q1_dominant, q3_dominant, color='blue', alpha=0.2, label=f"IQR dominant: {iqr_dominant:.2f}")
    # plt.hist(correlations_non_dominant, bins=15, color='orange', alpha=0.7, edgecolor='black', label='non dominant')
    # plt.hist(correlations_dominant, bins=15, color='blue', alpha=0.7, edgecolor='black', label='dominant')
    # plt.axvline(median_non_dominant, color='orange', linestyle='--', label="non dominant median")
    # plt.axvline(median_dominant, color='blue', linestyle='--', label="dominant median")
    # plt.axvspan(q1_non_dominant, q3_non_dominant, color='orange', alpha=0.2, label="non dominant IQR")
    # plt.axvspan(q1_dominant, q3_dominant, color='blue', alpha=0.2, label="dominant IQR")
    
    plt.title("Histogram of Spearman Correlations by Hand")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

    # # Print statistics
    # print(f"non_dominant Hand: Median = {median_non_dominant:.2f}, IQR = {iqr_non_dominant:.2f}")
    # print(f"dominant Hand: Median = {median_dominant:.2f}, IQR = {iqr_dominant:.2f}")
    # print(f"Wilcoxon Signed-Rank Test (non_dominant Hand): Statistic = {stat_non_dominant:.2f}, P-value = {p_value_non_dominant:.4f}")
    # print(f"Wilcoxon Signed-Rank Test (dominant Hand): Statistic = {stat_dominant:.2f}, P-value = {p_value_dominant:.4f}")

# Perform Wilcoxon signed-rank test on Spearman correlation values for non_dominant and dominant hands separately
def wilcoxon_test_spearman_corr_separate(results):
    """
    Performs the Wilcoxon signed-rank test on Spearman correlation values for non_dominant and dominant hands separately.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Test statistics and p-values for non_dominant and dominant hands.
    """
    correlations_non_dominant = []
    correlations_dominant = []

    for subject, hands_data in results.items():
        if 'non_dominant' in hands_data and 'spearman_corr' in hands_data['non_dominant']:
            correlations_non_dominant.append(hands_data['non_dominant']['spearman_corr'])
        if 'dominant' in hands_data and 'spearman_corr' in hands_data['dominant']:
            correlations_dominant.append(hands_data['dominant']['spearman_corr'])

    # Perform Wilcoxon signed-rank test for non_dominant hand
    stat_non_dominant, p_value_non_dominant = wilcoxon(correlations_non_dominant)

    # Perform Wilcoxon signed-rank test for dominant hand
    stat_dominant, p_value_dominant = wilcoxon(correlations_dominant)

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results:")
    print(f"non_dominant Hand: Statistic = {stat_non_dominant:.2f}, P-value = {p_value_non_dominant:.4f}, Data Points = {len(correlations_non_dominant)}")
    print(f"dominant Hand: Statistic = {stat_dominant:.2f}, P-value = {p_value_dominant:.4f}, Data Points = {len(correlations_dominant)}")

    return {
        "non_dominant": {"statistic": stat_non_dominant, "p_value": p_value_non_dominant},
        "dominant": {"statistic": stat_dominant, "p_value": p_value_dominant}
    }

# Compare non_dominant vs dominant hands at the subject level using Wilcoxon signed-rank test on paired subject Spearman correlations for durations and distances
def compare_non_dominant_vs_dominant_hands_trials_mean_median_of_reach_indices(results):
    """
    Compares the non_dominant and dominant hands at the subject level using the Wilcoxon signed-rank test
    on paired subject Spearman correlations for durations and distances.

    Parameters:
        results (dict): Dictionary containing Spearman correlation results for each subject.

    Returns:
        dict: Results of the Wilcoxon signed-rank test for durations and distances.
    """
    non_dominant_spearman_corrs = []
    dominant_spearman_corrs = []


    for subject, hands_data in results.items():
        if 'non_dominant' in hands_data and 'dominant' in hands_data:
            non_dominant_corr = hands_data['non_dominant']['spearman_corr']
            dominant_corr = hands_data['dominant']['spearman_corr']
            if not (pd.isna(non_dominant_corr) or pd.isna(dominant_corr)):
                non_dominant_spearman_corrs.append(non_dominant_corr)
                dominant_spearman_corrs.append(dominant_corr)

    non_dominant_data_points = len(non_dominant_spearman_corrs)
    dominant_data_points = len(dominant_spearman_corrs)

    # Perform Wilcoxon signed-rank test for Spearman correlations
    stat_corrs, p_value_corrs = wilcoxon(non_dominant_spearman_corrs, dominant_spearman_corrs)

    results = {
        "spearman_correlations": {"statistic": stat_corrs, "p_value": p_value_corrs},
        "data_points": {"non_dominant": non_dominant_data_points, "dominant": dominant_data_points}
    }

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results Compare non_dominant And dominant:")
    print(f"Spearman Correlations: Statistic = {stat_corrs:.2f}, P-value = {p_value_corrs:.4f}")
    print(f"Data Points: non_dominant Hand = {non_dominant_data_points}, dominant Hand = {dominant_data_points}")

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
    # Calculate average and median metrics for all reach indices for each reach_index for all subjects and hands across trials
    mean_stats, median_stats = calculate_trials_mean_median_of_reach_indices(updated_metrics, metric_x, metric_y)

    # Overlay scatter plots for all reach indices
    plot_trials_mean_median_of_reach_indices(median_stats, sample_subject, 'non_dominant', metric_x, metric_y, stat_type=stat_type, use_unique_colors=True)
    plot_trials_mean_median_of_reach_indices(median_stats, sample_subject, 'dominant', metric_x, metric_y, stat_type=stat_type, use_unique_colors=True)

    # Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters
    results = calculate_duration_distance_trials_mean_median_of_reach_indices(median_stats, stat_type=stat_type)

    # Plot histograms for Spearman correlations and p-values
    plot_histogram_spearman_corr_with_stats_trials_mean_median_of_reach_indices(results)

    # Perform Wilcoxon signed-rank test on Spearman correlation values
    wilcoxon_results_by_hand_trials_mean_median_of_reach_indices = wilcoxon_test_spearman_corr_separate(results)

    # Compare non_dominant vs dominant hands at the subject level using Wilcoxon signed-rank test on paired Spearman correlations
    wilcoxon_results_compare_hand_trials_mean_median_of_reach_indices = compare_non_dominant_vs_dominant_hands_trials_mean_median_of_reach_indices(results)

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
        hand (str): Hand ('non_dominant' or 'dominant').
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
                                fontsize=12, color="black", ha="non_dominant", va="bottom")
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
        hand (str): Hand ('non_dominant' or 'dominant').
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
                    # plt.text(xi, yi, f"({xi:.2f}, {yi:.2f})", fontsize=8, color=color_palette[reach_index], ha="dominant", va="bottom")

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
        for hand in ['non_dominant', 'dominant']:
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


def heatmap_spearman_correlation_reach_indices(results, hand="both", simplified=False, return_medians=False, overlay_median=False):
    """
    Plots a heatmap of Spearman correlations for the specified hand(s) and optionally returns the column and row medians.
    Optionally overlays a green square on each row at the cell closest to the row median.
    
    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
        hand (str): Which hand to plot; "non_dominant", "dominant", or "both". Default is "both".
        simplified (bool): If True, plots a compact version with no annotations and no subject labels.
                           When hand == "both", each hand is plotted as a subplot.
        return_medians (bool): If True, returns a dictionary containing column and row medians.
        overlay_median (bool): If True, overlays a green square on each row at the cell closest to the row median.
        
    Returns:
        dict or None: If return_medians is True, returns a dictionary with keys corresponding to each hand 
                      (or the chosen hand) and values as dictionaries with 'column_medians' and 'row_medians'.
    """
    import matplotlib.pyplot as plt
    reach_indices = list(range(16))
    medians = {}
    
    if hand == "both":
        if simplified:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, len(results) * 0.5))
        
        for idx, h in enumerate(["non_dominant", "dominant"]):
            subjects = list(results.keys())
            data = []
            for subject in subjects:
                if h in results[subject]:
                    correlations = [
                        results[subject][h].get(ri, {}).get("spearman_corr", np.nan)
                        for ri in reach_indices
                    ]
                    data.append(correlations)
            df = pd.DataFrame(data, index=subjects, columns=reach_indices)
            ax = axes[idx] if isinstance(axes, (list, np.ndarray)) else axes
            sns.heatmap(
                df,
                annot=not simplified,
                fmt=".2f",
                cmap="coolwarm",
                cbar=True,
                xticklabels=list(range(1, 17)),
                yticklabels=[] if simplified else subjects,
                vmin=-1,
                vmax=1,
                ax=ax
            )
            ax.set_title(f"{h.capitalize()} Hand", fontsize=18)
            ax.set_xlabel("Reach Index", fontsize=18)
            ax.set_xticklabels(range(1, 17), fontsize=10, rotation=0)
            ax.set_ylabel("Subjects", fontsize=18)
            if overlay_median:
                import matplotlib.patches as patches
                for i, subject in enumerate(df.index):
                    # Calculate row median from non-NaN values
                    row_data = df.loc[subject].dropna()
                    if row_data.empty:
                        continue
                    median_val = np.median(row_data.values)
                    # Find the column index with the value closest to the row median
                    col_idx = np.argmin(np.abs(df.loc[subject].values - median_val))
                    # Overlay a green rectangle to highlight the cell
                    ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
            if return_medians:
                medians[h] = {
                    "column_medians": df.median(axis=0).to_dict(),
                    "row_medians": df.median(axis=1).to_dict()
                }
        
        plt.tight_layout()
        plt.show()
    
    else:
        subjects = list(results.keys())
        data = []
        for subject in subjects:
            if hand in results[subject]:
                correlations = [
                    results[subject][hand].get(ri, {}).get("spearman_corr", np.nan)
                    for ri in reach_indices
                ]
                data.append(correlations)
        df = pd.DataFrame(data, index=subjects, columns=reach_indices)
        fig, ax = plt.subplots(figsize=(8, 4) if simplified else (12, len(subjects) * 0.5))
        sns.heatmap(
            df,
            annot=not simplified,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            xticklabels=list(range(1, 17)),
            yticklabels=[] if simplified else subjects,
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title(f"{hand.capitalize()} Hand")
        ax.set_xlabel("Reach Index")
        ax.set_xticklabels(range(1, 17), fontsize=10, rotation=0)
        ax.set_ylabel("Subjects")
        ax.set_yticklabels([] if simplified else ax.get_yticklabels())
        if overlay_median:
            import matplotlib.patches as patches
            for i, subject in enumerate(df.index):
                # Calculate row median from non-NaN values
                row_data = df.loc[subject].dropna()
                if row_data.empty:
                    continue
                median_val = np.median(row_data.values)
                # Find the column index with the value closest to the row median
                col_idx = np.argmin(np.abs(df.loc[subject].values - median_val))
                # Overlay a green rectangle to highlight the cell
                ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
        plt.tight_layout()
        plt.show()
        if return_medians:
            medians[hand] = {
                "column_medians": df.median(axis=0).to_dict(),
                "row_medians": df.median(axis=1).to_dict()
            }
    
    if return_medians:
        return medians

# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
## correlations grouped by hand by subject 
# -------------------------------------------------------------------------------------------------------------------
# Plot histograms for median Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
def plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(results):
    """
    Plots histograms of median Spearman correlations for each subject,
    overlaying non_dominant and dominant hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
    """
    median_correlations_non_dominant = []
    median_correlations_dominant = []

    # Collect median correlation for each subject
    for subject in results.keys():
        for hand in ['non_dominant', 'dominant']:
            if hand in results[subject]:
                correlations = [
                    results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                    for reach_index in range(16)
                ]
                correlations = [corr for corr in correlations if not np.isnan(corr)]
                if correlations:
                    if hand == 'non_dominant':
                        median_correlations_non_dominant.append(np.median(correlations))
                    elif hand == 'dominant':
                        median_correlations_dominant.append(np.median(correlations))


    # Calculate statistics
    median_non_dominant = np.median(median_correlations_non_dominant)
    iqr_non_dominant = np.percentile(median_correlations_non_dominant, 75) - np.percentile(median_correlations_non_dominant, 25)
    q1_non_dominant = np.percentile(median_correlations_non_dominant, 25)
    q3_non_dominant = np.percentile(median_correlations_non_dominant, 75)

    median_dominant = np.median(median_correlations_dominant)
    iqr_dominant = np.percentile(median_correlations_dominant, 75) - np.percentile(median_correlations_dominant, 25)
    q1_dominant = np.percentile(median_correlations_dominant, 25)
    q3_dominant = np.percentile(median_correlations_dominant, 75)

    # Perform Wilcoxon signed-rank test for each hand separately
    stat_non_dominant, p_value_non_dominant = wilcoxon(median_correlations_non_dominant)
    stat_dominant, p_value_dominant = wilcoxon(median_correlations_dominant)

    # Plot histogram for median Spearman correlations
    plt.figure(figsize=(8, 6))
    plt.hist(median_correlations_non_dominant, bins=15, color='orange', alpha=0.7, edgecolor='black', label='non dominant Hand')
    plt.hist(median_correlations_dominant, bins=15, color='blue', alpha=0.7, edgecolor='black', label='dominant Hand')
    plt.axvline(median_non_dominant, color='orange', linestyle='--', label=f"Median non dominant: {median_non_dominant:.2f}")
    plt.axvline(median_dominant, color='blue', linestyle='--', label=f"Median dominant: {median_dominant:.2f}")
    plt.axvspan(q1_non_dominant, q3_non_dominant, color='orange', alpha=0.2, label=f"IQR non dominant: {iqr_non_dominant:.2f}")
    plt.axvspan(q1_dominant, q3_dominant, color='blue', alpha=0.2, label=f"IQR dominant: {iqr_dominant:.2f}")
    plt.title("Histogram of Median Spearman Correlations by Hand")

    # plt.hist(median_correlations_non_dominant, bins=15, color='orange', alpha=0.7, edgecolor='black', label='non dominant')
    # plt.hist(median_correlations_dominant, bins=15, color='blue', alpha=0.7, edgecolor='black', label='dominant')
    # plt.axvline(median_non_dominant, color='orange', linestyle='--', label="non dominant median")
    # plt.axvline(median_dominant, color='blue', linestyle='--', label="dominant median")
    # plt.axvspan(q1_non_dominant, q3_non_dominant, color='orange', alpha=0.2, label="non dominant IQR")
    # plt.axvspan(q1_dominant, q3_dominant, color='blue', alpha=0.2, label="dominant IQR")
    plt.xlabel("Median Spearman Correlation", fontsize=24)
    plt.ylabel("Frequency", fontsize=24)
    plt.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

    # # Print statistics
    # print(f"non_dominant Hand: Median = {median_non_dominant:.2f}, IQR = {iqr_non_dominant:.2f}")
    # print(f"dominant Hand: Median = {median_dominant:.2f}, IQR = {iqr_dominant:.2f}")
    # print(f"Wilcoxon Signed-Rank Test (non_dominant Hand): Statistic = {stat_non_dominant:.2f}, P-value = {p_value_non_dominant:.4f}")
    # print(f"Wilcoxon Signed-Rank Test (dominant Hand): Statistic = {stat_dominant:.2f}, P-value = {p_value_dominant:.4f}")

# Perform Wilcoxon signed-rank test on median Spearman correlation values for non_dominant and dominant hands separately
def wilcoxon_test_spearman_corr_separate_reach_indices_by_subject(results):
    """
    Performs the one-tailed Wilcoxon signed-rank test on median Spearman correlation values for non_dominant and dominant hands separately.
    By default, the alternative hypothesis is that the true median is smaller than zero.

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Test statistics, p-values, and median statistics for non_dominant and dominant hands.
    """
    median_correlations_non_dominant = []
    median_correlations_dominant = []

    # Collect median correlation for each subject
    for subject in results.keys():
        for hand in ['non_dominant', 'dominant']:
            if hand in results[subject]:
                correlations = [
                    results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                    for reach_index in range(16)
                ]
                correlations = [corr for corr in correlations if not np.isnan(corr)]
                if correlations:
                    if hand == 'non_dominant':
                        median_correlations_non_dominant.append(np.median(correlations))
                    elif hand == 'dominant':
                        median_correlations_dominant.append(np.median(correlations))

    # Calculate statistics
    median_non_dominant = np.median(median_correlations_non_dominant)
    iqr_non_dominant = np.percentile(median_correlations_non_dominant, 75) - np.percentile(median_correlations_non_dominant, 25)
    q1_non_dominant = np.percentile(median_correlations_non_dominant, 25)
    q3_non_dominant = np.percentile(median_correlations_non_dominant, 75)

    median_dominant = np.median(median_correlations_dominant)
    iqr_dominant = np.percentile(median_correlations_dominant, 75) - np.percentile(median_correlations_dominant, 25)
    q1_dominant = np.percentile(median_correlations_dominant, 25)
    q3_dominant = np.percentile(median_correlations_dominant, 75)

    # Perform one-tailed Wilcoxon signed-rank test for each hand separately (alternative hypothesis: median < 0)
    stat_non_dominant, p_value_non_dominant = wilcoxon(median_correlations_non_dominant, alternative='less')
    stat_dominant, p_value_dominant = wilcoxon(median_correlations_dominant, alternative='less')

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results (One-tailed):")
    print(f"non dominant Hand: Statistic = {stat_non_dominant:.2f}, P-value = {p_value_non_dominant:.4f}, Data Points = {len(median_correlations_non_dominant)}")
    print(f"dominant Hand: Statistic = {stat_dominant:.2f}, P-value = {p_value_dominant:.4f}, Data Points = {len(median_correlations_dominant)}")

    return {
        "non_dominant": {
            "statistic": stat_non_dominant,
            "p_value": p_value_non_dominant,
            "median": median_non_dominant,
            "iqr": iqr_non_dominant,
            "q1": q1_non_dominant,
            "q3": q3_non_dominant,
        },
        "dominant": {
            "statistic": stat_dominant,
            "p_value": p_value_dominant,
            "median": median_dominant,
            "iqr": iqr_dominant,
            "q1": q1_dominant,
            "q3": q3_dominant,
        },
    }

# Compare non_dominant vs dominant hands using Wilcoxon signed-rank test on median Spearman correlations
def compare_non_dominant_vs_dominant_hands_reach_indices_by_subject(results):
    """
    Compares the non_dominant and dominant hands using a one-tailed Wilcoxon signed-rank test
    on median Spearman correlations for durations and distances, predicting that the dominant hand
    has a weaker negative SAT (i.e. less negative correlation).

    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Results of the one-tailed Wilcoxon signed-rank test, median correlations, and 
              data points for non_dominant and dominant hands.
    """
    median_correlations_non_dominant = []
    median_correlations_dominant = []

    # Collect median correlation for each subject
    for subject in results.keys():
        for hand in ['non_dominant', 'dominant']:
            if hand in results[subject]:
                correlations = [
                    results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                    for reach_index in range(16)
                ]
                correlations = [corr for corr in correlations if not np.isnan(corr)]
                if correlations:
                    if hand == 'non_dominant':
                        median_correlations_non_dominant.append(np.median(correlations))
                    elif hand == 'dominant':
                        median_correlations_dominant.append(np.median(correlations))

    # Perform one-tailed Wilcoxon signed-rank test 
    # (alternative hypothesis: non_dominant - dominant < 0,
    #  i.e. dominant correlations are less negative than non_dominant)
    stat, p_value = wilcoxon(median_correlations_non_dominant,
                             median_correlations_dominant,
                             alternative='less')

    # Print the results of the Wilcoxon signed-rank test
    print(f"Wilcoxon Signed-Rank Test Results Compare non_dominant And dominant:")
    print(f"Spearman Correlations: Statistic = {stat:.2f}, P-value = {p_value:.4f}")
    print(f"Data Points: non_dominant Hand = {len(median_correlations_non_dominant)}, dominant Hand = {len(median_correlations_dominant)}")

    return {
        "median_correlations_non_dominant": median_correlations_non_dominant,
        "median_correlations_dominant": median_correlations_dominant,
        "data_points_non_dominant": len(median_correlations_non_dominant),
        "data_points_dominant": len(median_correlations_dominant),
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

    # 42. Calculate and return Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject, hand, and reach index
    results = calculate_duration_distance_reach_indices(updated_metrics)

    # Plot heatmap for Spearman correlations for each reach indices (1 to 16) for each subject and hand
    medians = heatmap_spearman_correlation_reach_indices(results, hand="both", simplified=True, return_medians=True, overlay_median=True)
    
    analysis = {}
    # 5. Run analysis based on grouping parameter
    if grouping == "hand_by_subject":

        plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(results)
        wilcox_by_subject = wilcoxon_test_spearman_corr_separate_reach_indices_by_subject(results)
        compare_hands_subject = compare_non_dominant_vs_dominant_hands_reach_indices_by_subject(results)
        analysis = {
            "wilcox_by_subject": wilcox_by_subject,
            "compare_hands_subject": compare_hands_subject
        }
    else:
        raise ValueError("grouping must be one of 'hand', 'hand_by_index', 'hand_by_subject'")
    
    return subject_statistics, results, analysis, medians


# --------------------------------------------------------------------------------------------------------------------
## bin
def heatmap_spearman_correlation_reach_indices_signifcant(results, hand="both", simplified=False, return_medians=False, overlay_median=False):
    """
    Plots a heatmap of Spearman correlations for the specified hand(s) showing only significant correlations 
    (p < 0.05). Non-significant correlations are masked (set to NaN) so that only the significant ones are shown.
    Additionally, calculates and prints various statistics about the correlations.
    
    Statistics include:
        - Total number and percentage of significant correlations (p < 0.05), divided into positive and negative.
        - For positive correlations: number, percentage significant, and percentage non-significant.
        - For negative correlations: number, percentage significant, and percentage non-significant.
    
    Parameters:
        results (dict): Results containing Spearman correlations and p-values for each subject and hand.
        hand (str): Which hand to plot; "non_dominant", "dominant", or "both". Default is "both".
        simplified (bool): If True, plots a compact version with no annotations and no subject labels.
                           When hand == "both", each hand is plotted as a subplot.
        return_medians (bool): If True, returns a dictionary containing column and row medians.
        overlay_median (bool): If True, overlays a green square on each row at the cell closest to the row median.
        
    Returns:
        dict or None: If return_medians is True, returns a dictionary with keys corresponding to each hand 
                      (or the chosen hand) and values as dictionaries with 'column_medians' and 'row_medians'.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    reach_indices = list(range(16))
    medians = {}
    sig_threshold = 0.05  # significance threshold

    def custom_annot(df):
        return df.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

    if hand == "both":
        if simplified:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, len(results) * 0.5))
        
        for idx, h in enumerate(["non_dominant", "dominant"]):
            subjects = list(results.keys())
            data_corr = []
            data_p = []
            for subject in subjects:
                if h in results[subject]:
                    correlations = []
                    p_values = []
                    for ri in reach_indices:
                        d = results[subject][h].get(ri, {})
                        correlations.append(d.get("spearman_corr", np.nan))
                        p_values.append(d.get("p_value", np.nan))
                    data_corr.append(correlations)
                    data_p.append(p_values)
            df = pd.DataFrame(data_corr, index=subjects, columns=reach_indices)
            df_p = pd.DataFrame(data_p, index=subjects, columns=reach_indices)
            
            # Keep only significant correlations
            df = df.where(df_p < sig_threshold)
            mask = df.isna()
            
            annot_data = custom_annot(df) if not simplified else False
            
            ax = axes[idx] if isinstance(axes, (list, np.ndarray)) else axes
            sns.heatmap(
                df,
                annot=annot_data,
                fmt="",
                cmap="coolwarm",
                cbar=True,
                mask=mask,
                xticklabels=list(range(1, 17)),
                yticklabels=[] if simplified else subjects,
                vmin=-1,
                vmax=1,
                ax=ax
            )
            ax.set_title(f"{h.capitalize()} Hand", fontsize=18)
            ax.set_xlabel("Reach Index", fontsize=18)
            ax.set_xticklabels(range(1, 17), fontsize=10, rotation=0)
            ax.set_ylabel("Subjects", fontsize=18)
            if overlay_median:
                for i, subject in enumerate(df.index):
                    row_data = df.loc[subject].dropna()
                    if row_data.empty:
                        continue
                    median_val = np.median(row_data.values)
                    col_idx = np.argmin(np.abs(row_data.values - median_val))
                    # ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
            if return_medians:
                medians[h] = {
                    "column_medians": df.median(axis=0).to_dict(),
                    "row_medians": df.median(axis=1).to_dict()
                }
            # Calculate statistics
            total_count = df_p.notna().sum().sum()
            sig_total = (df_p < sig_threshold).sum().sum()
            pos_total = (df > 0).sum().sum()
            neg_total = (df < 0).sum().sum()
            pos_sig = (((df > 0) & (df_p < sig_threshold)).sum()).sum()
            neg_sig = (((df < 0) & (df_p < sig_threshold)).sum()).sum()
            pos_non_sig = ((df > 0) & (df_p >= sig_threshold)).sum().sum()
            neg_non_sig = ((df < 0) & (df_p >= sig_threshold)).sum().sum()
            percent_sig = (sig_total / total_count * 100) if total_count > 0 else np.nan
            percent_pos_overall = (pos_total / total_count * 100) if total_count > 0 else np.nan
            percent_neg_overall = (neg_total / total_count * 100) if total_count > 0 else np.nan
            percent_pos_sig = (pos_sig / pos_total * 100) if pos_total > 0 else np.nan
            percent_pos_non_sig = (pos_non_sig / pos_total * 100) if pos_total > 0 else np.nan
            percent_neg_sig = (neg_sig / neg_total * 100) if neg_total > 0 else np.nan
            percent_neg_non_sig = (neg_non_sig / neg_total * 100) if neg_total > 0 else np.nan
            if sig_total > 0:
                perc_pos_in_sig = pos_sig / sig_total * 100
                perc_neg_in_sig = neg_sig / sig_total * 100
            else:
                perc_pos_in_sig = perc_neg_in_sig = np.nan
            
            print(f"{h.capitalize()} Hand Statistics (Only Significant Correlations):")
            print(f"  Total correlations (all): {total_count}")
            print(f"  Significant correlations (p < {sig_threshold}): {sig_total} ({percent_sig:.2f}%)")
            print(f"    Among significant correlations: {pos_sig} positive ({perc_pos_in_sig:.2f}%), {neg_sig} negative ({perc_neg_in_sig:.2f}%)")
            print(f"  Positive correlations (significant only): {pos_sig} ({percent_pos_sig:.2f}%)")
            print(f"  Negative correlations (significant only): {neg_sig} ({percent_neg_sig:.2f}%)")
        
        plt.tight_layout()
        plt.show()
    
    else:
        subjects = list(results.keys())
        data_corr = []
        data_p = []
        for subject in subjects:
            if hand in results[subject]:
                correlations = []
                p_values = []
                for ri in reach_indices:
                    d = results[subject][hand].get(ri, {})
                    correlations.append(d.get("spearman_corr", np.nan))
                    p_values.append(d.get("p_value", np.nan))
                data_corr.append(correlations)
                data_p.append(p_values)
        df = pd.DataFrame(data_corr, index=subjects, columns=reach_indices)
        df_p = pd.DataFrame(data_p, index=subjects, columns=reach_indices)
        
        # Keep only significant correlations
        df = df.where(df_p < sig_threshold)
        mask = df.isna()
        
        annot_data = custom_annot(df) if not simplified else False
        fig, ax = plt.subplots(figsize=(8, 4) if simplified else (12, len(subjects) * 0.5))
        sns.heatmap(
            df,
            annot=annot_data,
            fmt="",
            cmap="coolwarm",
            cbar=True,
            mask=mask,
            xticklabels=list(range(1, 17)),
            yticklabels=[] if simplified else subjects,
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title(f"{hand.capitalize()} Hand (Only Significant Correlations)")
        ax.set_xlabel("Reach Index")
        ax.set_xticklabels(range(1, 17), fontsize=10, rotation=0)
        ax.set_ylabel("Subjects")
        ax.set_yticklabels([] if simplified else ax.get_yticklabels())
        if overlay_median:
            for i, subject in enumerate(df.index):
                row_data = df.loc[subject].dropna()
                if row_data.empty:
                    continue
                median_val = np.median(row_data.values)
                col_idx = np.argmin(np.abs(row_data.values - median_val))
                # ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
        # Calculate statistics
        total_count = df_p.notna().sum().sum()
        sig_total = (df_p < sig_threshold).sum().sum()
        pos_total = (df > 0).sum().sum()
        neg_total = (df < 0).sum().sum()
        pos_sig = (((df > 0) & (df_p < sig_threshold)).sum()).sum()
        neg_sig = (((df < 0) & (df_p < sig_threshold)).sum()).sum()
        pos_non_sig = ((df > 0) & (df_p >= sig_threshold)).sum().sum()
        neg_non_sig = ((df < 0) & (df_p >= sig_threshold)).sum().sum()
        percent_sig = (sig_total / total_count * 100) if total_count > 0 else np.nan
        percent_pos_overall = (pos_total / total_count * 100) if total_count > 0 else np.nan
        percent_neg_overall = (neg_total / total_count * 100) if total_count > 0 else np.nan
        percent_pos_sig = (pos_sig / pos_total * 100) if pos_total > 0 else np.nan
        percent_pos_non_sig = (pos_non_sig / pos_total * 100) if pos_total > 0 else np.nan
        percent_neg_sig = (neg_sig / neg_total * 100) if neg_total > 0 else np.nan
        percent_neg_non_sig = (neg_non_sig / neg_total * 100) if neg_total > 0 else np.nan
        if sig_total > 0:
            perc_pos_in_sig = pos_sig / sig_total * 100
            perc_neg_in_sig = neg_sig / sig_total * 100
        else:
            perc_pos_in_sig = perc_neg_in_sig = np.nan
        
        print(f"{hand.capitalize()} Hand Statistics (Only Significant Correlations):")
        print(f"  Total correlations (all): {total_count}")
        print(f"  Significant correlations (p < {sig_threshold}): {sig_total} ({percent_sig:.2f}%)")
        print(f"    Among significant correlations: {pos_sig} positive ({perc_pos_in_sig:.2f}%), {neg_sig} negative ({perc_neg_in_sig:.2f}%)")
        print(f"  Positive correlations (significant only): {pos_sig} ({percent_pos_sig:.2f}%)")
        print(f"  Negative correlations (significant only): {neg_sig} ({percent_neg_sig:.2f}%)")
        
        plt.tight_layout()
        plt.show()
        if return_medians:
            medians[hand] = {
                "column_medians": df.median(axis=0).to_dict(),
                "row_medians": df.median(axis=1).to_dict()
            }
    
    if return_medians:
        return medians
