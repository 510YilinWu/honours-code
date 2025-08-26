


# -------------------------------------------------------------------------------------------------------------------
# Compute standard deviation for each subject and hand, for durations or distance
def compute_std_for_subjects(updated_metrics, metric):
    """
    Computes the standard deviation for each subject and hand for the specified metric.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        metric (str): Metric to compute standard deviation for ('durations' or 'distance').

    Returns:
        dict: Dictionary containing standard deviation values for each subject and hand.
    """
    std_results = {}

    for subject, hands_data in updated_metrics.items():
        std_results[subject] = {}
        for hand, metrics in hands_data.items():
            all_values = []

            for trial, values in metrics[metric].items():
                all_values.extend([val for val in values if not pd.isna(val)])

            # Calculate standard deviation
            std_value = np.std(all_values) if all_values else np.nan
            std_results[subject][hand] = std_value

    return std_results

# Example usage
std_durations = compute_std_for_subjects(updated_metrics, 'durations')
std_distance = compute_std_for_subjects(updated_metrics, 'distance')

# Plot histograms for standard deviations of durations and distances in two subplots, overlaying left and right hands
def plot_std_histograms_overlay(std_durations, std_distance):
    """
    Plots histograms of standard deviations for durations and distances in two subplots,
    overlaying left and right hands in different colors.

    Parameters:
        std_durations (dict): Standard deviations for durations for each subject and hand.
        std_distance (dict): Standard deviations for distances for each subject and hand.
    """
    subjects = list(std_durations.keys())
    hands = ['left', 'right']

    # Prepare data for plotting
    durations_left = [std_durations[subject]['left'] for subject in subjects]
    durations_right = [std_durations[subject]['right'] for subject in subjects]
    distances_left = [std_distance[subject]['left'] for subject in subjects]
    distances_right = [std_distance[subject]['right'] for subject in subjects]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot histogram for durations
    axes[0].hist(durations_left, bins=10, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    axes[0].hist(durations_right, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
    axes[0].set_title('Standard Deviations of Durations')
    axes[0].set_xlabel('Standard Deviation')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot histogram for distances
    axes[1].hist(distances_left, bins=10, color='green', alpha=0.7, edgecolor='black', label='Left Hand')
    axes[1].hist(distances_right, bins=10, color='purple', alpha=0.7, edgecolor='black', label='Right Hand')
    axes[1].set_title('Standard Deviations of Distances')
    axes[1].set_xlabel('Standard Deviation')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot histograms
plot_std_histograms_overlay(std_durations, std_distance)
# -------------------------------------------------------------------------------------------------------------------



# Plot histogram for all distances and indicate statistical measures
def plot_all_distance_histogram_with_stats(filtered_metrics):
    """
    Plots a histogram of all distance values across all subjects, hands, and trials,
    and indicates statistical measures such as median, mean, standard deviations, and range.

    Parameters:
        filtered_metrics (dict): Filtered combined metrics data.
    """
    all_distances = []

    for subject, hands_data in filtered_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, distances in metrics['distance'].items():
                all_distances.extend([dist for dist in distances if not pd.isna(dist)])

    # Calculate statistics
    mean_distance = np.mean(all_distances)
    median_distance = np.median(all_distances)
    std_distance = np.std(all_distances)
    range_distance = (min(all_distances), max(all_distances))

    # Count the number of data points larger than 15
    count_larger_than_15 = sum(1 for dist in all_distances if dist > 15)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_distances, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_distance, color='red', linestyle='--', label=f"Mean: {mean_distance:.2f}")
    plt.axvline(median_distance, color='green', linestyle='--', label=f"Median: {median_distance:.2f}")
    plt.axvline(mean_distance + std_distance, color='orange', linestyle='--', label=f"Mean + 1 SD: {mean_distance + std_distance:.2f}")
    plt.axvline(mean_distance - std_distance, color='orange', linestyle='--', label=f"Mean - 1 SD: {mean_distance - std_distance:.2f}")
    plt.axvline(mean_distance + 2 * std_distance, color='purple', linestyle='--', label=f"Mean + 2 SD: {mean_distance + 2 * std_distance:.2f}")
    plt.axvline(mean_distance - 2 * std_distance, color='purple', linestyle='--', label=f"Mean - 2 SD: {mean_distance - 2 * std_distance:.2f}")
    plt.axvline(mean_distance + 3 * std_distance, color='brown', linestyle='--', label=f"Mean + 3 SD: {mean_distance + 3 * std_distance:.2f}")
    plt.axvline(mean_distance - 3 * std_distance, color='brown', linestyle='--', label=f"Mean - 3 SD: {mean_distance - 3 * std_distance:.2f}")
    plt.title("Histogram of All Distances with Statistical Measures")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Total number of distance values: {len(all_distances)}")
    print(f"Mean distance: {mean_distance:.2f}")
    print(f"Median distance: {median_distance:.2f}")
    print(f"Standard deviation: {std_distance:.2f}")
    print(f"Range of distances: {range_distance}")
    print(f"Number of data points larger than 15: {count_larger_than_15}")

    # Find the subject, hand, and trial corresponding to the maximum distance
    max_distance = max(all_distances)
    for subject, hands_data in filtered_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, distances in metrics['distance'].items():
                if max_distance in distances:
                    print(f"Maximum distance found in Subject: {subject}, Hand: {hand}, Trial: {trial}")
                    break

# Call the function to plot the histogram with statistics
plot_all_distance_histogram_with_stats(all_combined_metrics)

# Plot histogram for all durations and indicate statistical measures
def plot_all_duration_histogram_with_stats(filtered_metrics):
    """
    Plots a histogram of all duration values across all subjects, hands, and trials,
    and indicates statistical measures such as median, mean, standard deviations, and range.

    Parameters:
        filtered_metrics (dict): Filtered combined metrics data.
    """
    all_durations = []

    for subject, hands_data in filtered_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, durations in metrics['durations'].items():
                all_durations.extend([dur for dur in durations if not pd.isna(dur)])

    # Calculate statistics
    mean_duration = np.mean(all_durations)
    median_duration = np.median(all_durations)
    std_duration = np.std(all_durations)
    range_duration = (min(all_durations), max(all_durations))

    # Count the number of data points larger than 1.6
    count_larger_than_1_6 = sum(1 for dur in all_durations if dur > 1.6)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_durations, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_duration, color='red', linestyle='--', label=f"Mean: {mean_duration:.2f}")
    plt.axvline(median_duration, color='green', linestyle='--', label=f"Median: {median_duration:.2f}")
    plt.axvline(mean_duration + std_duration, color='orange', linestyle='--', label=f"Mean + 1 SD: {mean_duration + std_duration:.2f}")
    plt.axvline(mean_duration - std_duration, color='orange', linestyle='--', label=f"Mean - 1 SD: {mean_duration - std_duration:.2f}")
    plt.axvline(mean_duration + 2 * std_duration, color='purple', linestyle='--', label=f"Mean + 2 SD: {mean_duration + 2 * std_duration:.2f}")
    plt.axvline(mean_duration - 2 * std_duration, color='purple', linestyle='--', label=f"Mean - 2 SD: {mean_duration - 2 * std_duration:.2f}")
    plt.axvline(mean_duration + 3 * std_duration, color='brown', linestyle='--', label=f"Mean + 3 SD: {mean_duration + 3 * std_duration:.2f}")
    plt.axvline(mean_duration - 3 * std_duration, color='brown', linestyle='--', label=f"Mean - 3 SD: {mean_duration - 3 * std_duration:.2f}")
    plt.axvline(mean_duration + 4 * std_duration, color='brown', linestyle='--', label=f"Mean + 4 SD: {mean_duration + 4 * std_duration:.2f}")
    plt.axvline(mean_duration - 4 * std_duration, color='brown', linestyle='--', label=f"Mean - 4 SD: {mean_duration - 4 * std_duration:.2f}")
    plt.title("Histogram of All Durations with Statistical Measures")
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Total number of duration values: {len(all_durations)}")
    print(f"Mean duration: {mean_duration:.2f}")
    print(f"Median duration: {median_duration:.2f}")
    print(f"Standard deviation: {std_duration:.2f}")
    print(f"Range of durations: {range_duration}")
    print(f"Number of data points larger than 1.6: {count_larger_than_1_6}")

    # Find the maximum duration and its corresponding subject, hand, trial, and index
    max_duration = float('-inf')
    max_info = None

    for subject, hands_data in filtered_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, durations in metrics['durations'].items():
                for i, duration in enumerate(durations):
                    if duration > max_duration:
                        max_duration = duration
                        max_info = (subject, hand, trial, i)

    if max_info:
        subject, hand, trial, index = max_info
        print(f"Maximum duration: {max_duration} found in Subject: {subject}, Hand: {hand}, Trial: {trial}, Index: {index}")

# Call the function to plot the histogram with statistics
plot_all_duration_histogram_with_stats(filtered_metrics)


# Plot heatmap for each subject as a subplot with individual color bars and highlighted outliers
def plot_heatmaps_for_all_subjects_with_outliers(all_combined_metrics, metric='durations', hand='left', threshold=None, percentile_range=(5, 95)):
    """
    Plots heatmaps of a selected metric for each subject and hand as subplots with individual color bars
    and highlights outliers based on a threshold or percentile range.

    Parameters:
        all_combined_metrics (dict): Combined metrics for all subjects.
        metric (str): Metric to plot ('durations', 'distance', 'ldlj', 'sparc').
        hand (str): Hand to plot ('left' or 'right').
        threshold (float, optional): Threshold for identifying outliers. If None, percentile_range is used.
        percentile_range (tuple, optional): Percentile range for color scaling (default is (5, 95)).
    """
    subjects = list(all_combined_metrics.keys())
    num_subjects = len(subjects)
    fig, axes = plt.subplots(num_subjects, 1, figsize=(8, 1 * num_subjects), sharey=True)
    axes = axes.flatten()

    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        # Extract the selected metric data for the specified hand
        metric_data = all_combined_metrics[subject][hand][metric]

        # Convert metric data to a DataFrame for heatmap plotting
        metric_df = pd.DataFrame.from_dict(metric_data, orient='index').transpose()

        # Calculate color scaling based on threshold or percentile range
        if threshold is None:
            vmin = metric_df.quantile(percentile_range[0] / 100).min()
            vmax = metric_df.quantile(percentile_range[1] / 100).max()
        else:
            vmin, vmax = metric_df.min().min(), metric_df.max().max()

        # Plot heatmap with individual color bar
        sns.heatmap(metric_df, cmap="coolwarm", annot=False, cbar=True, vmin=vmin, vmax=vmax, ax=ax, xticklabels=False)
        # ax.set_title(f"Subject: {subject} ({hand.capitalize()}, {metric.capitalize()})")
        # ax.set_xlabel("Trial")
        # ax.set_ylabel("Reach Location (0 to 15)")

        # Identify and label outliers if a threshold is provided
        if threshold is not None:
            for i in range(metric_df.shape[0]):
                for j in range(metric_df.shape[1]):
                    value = metric_df.iloc[i, j]
                    if value > threshold:
                        ax.text(j + 0.5, i + 0.5, f"{value:.1f}", color="black", ha="center", va="center", fontsize=8,
                                bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.3"))

    # Hide unused subplots
    for i in range(len(subjects), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(h_pad=0.01)  # Reduce the gap between subplots
    plt.show()

# Example usage
plot_heatmaps_for_all_subjects_with_outliers(all_combined_metrics, metric='distance', hand='left', percentile_range=(2.5, 97.5))
plot_heatmaps_for_all_subjects_with_outliers(all_combined_metrics, metric='durations', hand='left', percentile_range=(2.5, 97.5))
plot_heatmaps_for_all_subjects_with_outliers(all_combined_metrics, metric='sparc', hand='left', percentile_range=(2.5, 97.5))
plot_heatmaps_for_all_subjects_with_outliers(all_combined_metrics, metric='ldlj', hand='left', percentile_range=(2.5, 97.5))




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
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

# Calculate and print statistics for each subject
subject_statistics = calculate_statistics_for_subjects(updated_metrics)

# Scatter plot for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation
def scatter_plot_all_reach_indices(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
    """
    Plots scatter plots for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        subject_statistics (dict): Statistics containing max durations and max distances for the subject.
    """
    # Create subplots in a 4x4 layout
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes = axes.flatten()

    # Get max duration and max distance for the subject
    max_duration = subject_statistics[subject]['max_durations']
    max_distance = subject_statistics[subject]['max_distance']

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
        print(len(x_values), len(y_values), len(trial_colors))
        print(x_values)
        print("----")
        print(y_values)

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

        # Perform hyperbolic regression if applicable
        if metric_x == 'durations' and metric_y == 'distance':
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
                                fontsize=12, color="black", ha="right", va="bottom")
            except Exception as e:
                print(f"Hyperbolic regression failed for Reach Index {reach_index}: {e}")
        else:
            print(f"Error: Hyperbolic regression is only applicable for 'durations' vs 'distance'.")

        # Add Spearman correlation to the plot
        ax.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Add diagonal line from (0, 0) to max duration and max distance
        if not np.isnan(max_duration) and not np.isnan(max_distance):
            ax.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")

    # Hide unused subplots
    for i in range(16, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
scatter_plot_all_reach_indices(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', subject_statistics)

for subject, stats in subject_statistics.items():
    scatter_plot_all_reach_indices(updated_metrics, subject, 'right', 'durations', 'distance', subject_statistics)

# Overlay 16 hyperbolic regressions, intersections, and diagonal line in one figure
def overlay_hyperbolic_regressions(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
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
    plt.figure(figsize=(10, 8))

    # Generate a color palette from light blue to black
    color_palette = sns.color_palette("light:black", 16)

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
                    plt.scatter(xi, yi, color=color_palette[reach_index], s=50, zorder=5)
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
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

# Example usage
results = overlay_hyperbolic_regressions(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', subject_statistics)


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# Extract durations for the specified subject and hand
durations_data = all_combined_metrics['06/19/CZ']['right']['durations']

# Convert durations data to a DataFrame for heatmap plotting
durations_df = pd.DataFrame.from_dict(durations_data, orient='index')

# Calculate the global min and max for consistent color scaling
vmin = durations_df.min().min()
vmax = durations_df.max().max()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(durations_df, cmap="coolwarm", annot=False, cbar=True, vmin=vmin, vmax=vmax)
plt.title("Heatmap of Durations for Subject one Hand")
plt.xlabel("Reach Index")
plt.ylabel("Trial")
plt.tight_layout()

# Identify and label outliers
threshold = 1.6  # Define a threshold for outliers (e.g., values > 1.6)
for i in range(durations_df.shape[0]):
    for j in range(durations_df.shape[1]):
        value = durations_df.iloc[i, j]
        if value > threshold:
            plt.text(j + 0.5, i + 0.5, f"{value:.1f}", color="black", ha="center", va="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.3"))

plt.show()


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

# result corr section
# Calculate and plot Spearman correlation for SPARC, LDLJ with duration and distance
def calculate_and_plot_spearman_corr(updated_metrics, metric_x, metric_y):
    """
    Calculates and plots Spearman correlation between two metrics for each subject, hand, and reach type.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.

    Returns:
        dict: Spearman correlations for each subject, hand, and reach type.
    """
    spearman_results = {}
    for subject, hands_data in updated_metrics.items():
        spearman_results[subject] = {}
        for hand, metrics in hands_data.items():
            spearman_results[subject][hand] = {}
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()
            for reach_index in range(16):
                x_values, y_values = [], []
                for trial in metrics[metric_x].keys():
                    trial_x = np.array(metrics[metric_x][trial])
                    trial_y = np.array(metrics[metric_y][trial])
                    if reach_index < len(trial_x) and reach_index < len(trial_y):
                        x_values.append(trial_x[reach_index])
                        y_values.append(trial_y[reach_index])
                x_values, y_values = np.array(x_values), np.array(y_values)
                valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values, y_values = x_values[valid_indices], y_values[valid_indices]
                spearman_corr = spearmanr(x_values, y_values)[0] if len(x_values) > 1 else np.nan
                spearman_results[subject][hand][reach_index] = spearman_corr
                ax = axes[reach_index]
                ax.scatter(x_values, y_values, alpha=0.7, color='blue')
                ax.set_title(f"Reach {reach_index}\nSpearman: {spearman_corr:.2f}")
                ax.set_xlabel(metric_x.capitalize())
                ax.set_ylabel(metric_y.capitalize())
                ax.grid(alpha=0.5)
            plt.suptitle(f"Spearman Correlations - {subject} ({hand.capitalize()})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
    return spearman_results

# Plot histogram of Spearman correlations for all subjects, overlaying right and left hands in different colors
def plot_spearman_results_histogram_overlay(spearman_results):
    """
    Plots histograms of Spearman correlations for all subjects, overlaying right and left hands in different colors.
    Also calculates and displays the mean and median for each hand.

    Parameters:
        spearman_results (dict): Spearman correlation results for each subject, hand, and reach type.
    """
    right_hand_correlations = []
    left_hand_correlations = []

    # Collect correlations for all subjects
    for subject, hands_data in spearman_results.items():
        for hand, reach_data in hands_data.items():
            correlations = [corr for corr in reach_data.values() if not np.isnan(corr)]
            if hand == "right":
                right_hand_correlations.extend(correlations)
            elif hand == "left":
                left_hand_correlations.extend(correlations)

    # Calculate statistics
    mean_right = np.mean(right_hand_correlations) if right_hand_correlations else np.nan
    median_right = np.median(right_hand_correlations) if right_hand_correlations else np.nan
    mean_left = np.mean(left_hand_correlations) if left_hand_correlations else np.nan
    median_left = np.median(left_hand_correlations) if left_hand_correlations else np.nan

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(right_hand_correlations, bins=15, alpha=0.7, label="Right Hand", color="blue", edgecolor="black")
    plt.hist(left_hand_correlations, bins=15, alpha=0.7, label="Left Hand", color="orange", edgecolor="black")

    # Add mean and median lines
    plt.axvline(mean_right, color="blue", linestyle="--", label=f"Mean Right: {mean_right:.2f}")
    plt.axvline(median_right, color="blue", linestyle="-.", label=f"Median Right: {median_right:.2f}")
    plt.axvline(mean_left, color="orange", linestyle="--", label=f"Mean Left: {mean_left:.2f}")
    plt.axvline(median_left, color="orange", linestyle="-.", label=f"Median Left: {median_left:.2f}")

    plt.title("Histogram of Spearman Correlations for All Subjects")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Calculate average Spearman correlation per subject per hand
def average_spearman_corr_per_subject_hand(spearman_corr_results):
    """
    Calculates the average Spearman correlation for each subject and hand.

    Parameters:
        spearman_corr_results (dict): Spearman correlation results for each subject, hand, and reach type.

    Returns:
        dict: Average Spearman correlations for each subject and hand.
    """
    averages = {}
    for subject, hands_data in spearman_corr_results.items():
        averages[subject] = {}
        for hand, reach_data in hands_data.items():
            valid_correlations = [corr for corr in reach_data.values() if not np.isnan(corr)]
            averages[subject][hand] = np.mean(valid_correlations) if valid_correlations else np.nan
    return averages

# Plot histogram of average Spearman correlations for left and right hands
def plot_histogram_of_average_correlations_all_hands(averages, title_suffix=""):
    """
    Plots histograms of average Spearman correlations for left and right hands across all subjects.

    Parameters:
        averages (dict): Averages of Spearman correlations for each subject and hand.
        title_suffix (str): Suffix to add to the plot title (e.g., "ldlj_distance").
    """
    avg_left = [averages[subject]["left"] for subject in averages.keys() if not np.isnan(averages[subject]["left"])]
    avg_right = [averages[subject]["right"] for subject in averages.keys() if not np.isnan(averages[subject]["right"])]
    median_left, median_right = np.median(avg_left) if avg_left else np.nan, np.median(avg_right) if avg_right else np.nan
    plt.figure(figsize=(6, 6))
    plt.hist(avg_left, bins=10, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(avg_right, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
    plt.axvline(median_left, color='orange', linestyle='--', label=f'Median Left: {median_left:.2f}')
    plt.axvline(median_right, color='blue', linestyle='--', label=f'Median Right: {median_right:.2f}')
    plt.xlabel('Average Spearman Correlation')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Average Spearman Correlations for Left and Right Hands ({title_suffix})')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
sparc_duration_results = calculate_and_plot_spearman_corr(updated_metrics, 'sparc', 'durations')
sparc_distance_results = calculate_and_plot_spearman_corr(updated_metrics, 'sparc', 'distance')
ldlj_duration_results = calculate_and_plot_spearman_corr(updated_metrics, 'ldlj', 'durations')
ldlj_distance_results = calculate_and_plot_spearman_corr(updated_metrics, 'ldlj', 'distance')

plot_spearman_results_histogram_overlay(sparc_duration_results)
plot_spearman_results_histogram_overlay(sparc_distance_results)
plot_spearman_results_histogram_overlay(ldlj_duration_results)
plot_spearman_results_histogram_overlay(ldlj_distance_results)

average_spearman_corr_sparc_duration = average_spearman_corr_per_subject_hand(sparc_duration_results)
average_spearman_corr_sparc_distance = average_spearman_corr_per_subject_hand(sparc_distance_results)
average_spearman_corr_ldlj_duration = average_spearman_corr_per_subject_hand(ldlj_duration_results)
average_spearman_corr_ldlj_distance = average_spearman_corr_per_subject_hand(ldlj_distance_results)

plot_histogram_of_average_correlations_all_hands(average_spearman_corr_sparc_duration, title_suffix="sparc_duration")

plot_histogram_of_average_correlations_all_hands(average_spearman_corr_sparc_distance, title_suffix="sparc_distance")

plot_histogram_of_average_correlations_all_hands(average_spearman_corr_ldlj_duration, title_suffix="ldlj_duration")

plot_histogram_of_average_correlations_all_hands(average_spearman_corr_ldlj_distance, title_suffix="ldlj_distance")



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

# def convert_distance_to_accuracy(updated_metrics, max_distance=15):
#     """
#     Converts distance values to accuracy percentages for all subjects, hands, and trials.

#     Parameters:
#         updated_metrics (dict): Filtered combined metrics data.
#         max_distance (float): Maximum distance value for calculating accuracy.

#     Returns:
#         dict: Updated metrics with accuracy values added.
#     """
#     for subject, hands_data in updated_metrics.items():
#         for hand, metrics in hands_data.items():
#             if 'accuracy' not in metrics:
#                 metrics['accuracy'] = {}
#             for trial, distances in metrics['distance'].items():
#                 accuracies = []
#                 for dist in distances:
#                     if pd.isna(dist):
#                         accuracies.append(np.nan)
#                     else:
#                         # accuracy = max(0, 100 - (dist / max_distance) * 100)
#                         accuracy = 1/dist
#                         accuracies.append(accuracy)
#                 metrics['accuracy'][trial] = accuracies
#     return updated_metrics

# # Convert distances to accuracy
# updated_metrics = convert_distance_to_accuracy(updated_metrics)


# def convert_duration_to_speed(updated_metrics):
#     """
#     Converts duration values to speed (1/duration) for all subjects, hands, and trials.

#     Parameters:
#         updated_metrics (dict): Filtered combined metrics data.

#     Returns:
#         dict: Updated metrics with speed values added.
#     """
#     for subject, hands_data in updated_metrics.items():
#         for hand, metrics in hands_data.items():
#             if 'speed' not in metrics:
#                 metrics['speed'] = {}
#             for trial, durations in metrics['durations'].items():
#                 speeds = []
#                 for dur in durations:
#                     if pd.isna(dur) or dur == 0:
#                         speeds.append(np.nan)
#                     else:
#                         speed = 1 / dur
#                         speeds.append(speed)
#                 metrics['speed'][trial] = speeds
#     return updated_metrics

# # Convert durations to speed
# updated_metrics = convert_duration_to_speed(updated_metrics)
# -------------------------------------------------------------------------------------------------------------------
# Plot histogram for all accuracies and indicate statistical measures
def plot_all_accuracy_histogram_with_stats(updated_metrics):
    """
    Plots a histogram of all accuracy values across all subjects, hands, and trials,
    and indicates statistical measures such as median, mean, standard deviations, and range.

    Parameters:
        updated_metrics (dict): Updated metrics data.
    """
    all_accuracies = []

    for subject, hands_data in updated_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, accuracies in metrics['accuracy'].items():
                all_accuracies.extend([acc for acc in accuracies if not pd.isna(acc)])

    # Calculate statistics
    mean_accuracy = np.mean(all_accuracies)
    median_accuracy = np.median(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    range_accuracy = (min(all_accuracies), max(all_accuracies))

    # Count the number of data points larger than 6
    count_larger_than_6 = sum(1 for acc in all_accuracies if acc > 6)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_accuracies, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_accuracy, color='red', linestyle='--', label=f"Mean: {mean_accuracy:.2f}")
    plt.axvline(median_accuracy, color='green', linestyle='--', label=f"Median: {median_accuracy:.2f}")
    plt.axvline(mean_accuracy + std_accuracy, color='orange', linestyle='--', label=f"Mean + 1 SD: {mean_accuracy + std_accuracy:.2f}")
    plt.axvline(mean_accuracy - std_accuracy, color='orange', linestyle='--', label=f"Mean - 1 SD: {mean_accuracy - std_accuracy:.2f}")
    plt.title("Histogram of All Accuracies with Statistical Measures")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Total number of accuracy values: {len(all_accuracies)}")
    print(f"Mean accuracy: {mean_accuracy:.2f}")
    print(f"Median accuracy: {median_accuracy:.2f}")
    print(f"Standard deviation: {std_accuracy:.2f}")
    print(f"Range of accuracies: {range_accuracy}")
    print(f"Number of data points larger than 6: {count_larger_than_6}")

    # Find the subject, hand, trial, and index corresponding to accuracies greater than 6
    for subject, hands_data in updated_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, accuracies in metrics['accuracy'].items():
                for index, accuracy in enumerate(accuracies):
                    if accuracy > 6:
                        distance = 1 / accuracy
                        print(f"Accuracy > 6 found in Subject: {subject}, Hand: {hand}, Trial: {trial}, Index: {index}, Accuracy: {accuracy:.2f}, Distance: {distance:.2f}")

# Call the function to plot the histogram with statistics
plot_all_accuracy_histogram_with_stats(updated_metrics)

# Plot histogram for all speeds and indicate statistical measures
def plot_all_speed_histogram_with_stats(updated_metrics):
    """
    Plots a histogram of all speed values across all subjects, hands, and trials,
    and indicates statistical measures such as median, mean, standard deviations, and range.

    Parameters:
        updated_metrics (dict): Updated metrics data.
    """
    all_speeds = []

    for subject, hands_data in updated_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, speeds in metrics['speed'].items():
                all_speeds.extend([spd for spd in speeds if not pd.isna(spd)])

    # Calculate statistics
    mean_speed = np.mean(all_speeds)
    median_speed = np.median(all_speeds)
    std_speed = np.std(all_speeds)
    range_speed = (min(all_speeds), max(all_speeds))

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_speeds, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_speed, color='red', linestyle='--', label=f"Mean: {mean_speed:.2f}")
    plt.axvline(median_speed, color='green', linestyle='--', label=f"Median: {median_speed:.2f}")
    plt.axvline(mean_speed + std_speed, color='orange', linestyle='--', label=f"Mean + 1 SD: {mean_speed + std_speed:.2f}")
    plt.axvline(mean_speed - std_speed, color='orange', linestyle='--', label=f"Mean - 1 SD: {mean_speed - std_speed:.2f}")
    plt.title("Histogram of All Speeds with Statistical Measures")
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Total number of speed values: {len(all_speeds)}")
    print(f"Mean speed: {mean_speed:.2f}")
    print(f"Median speed: {median_speed:.2f}")
    print(f"Standard deviation: {std_speed:.2f}")
    print(f"Range of speeds: {range_speed}")

# Call the function to plot the histogram with statistics
plot_all_speed_histogram_with_stats(updated_metrics)

# Scatter plot for all speed and all accuracy with hyperbolic fit
def scatter_plot_all_speed_accuracy(updated_metrics):
    """
    Plots a scatter plot of all speed vs all accuracy across all subjects, hands, and trials,
    and fits a hyperbolic curve.

    Parameters:
        updated_metrics (dict): Updated metrics data.
    """
    all_speeds = []
    all_accuracies = []

    for subject, hands_data in updated_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, speeds in metrics['speed'].items():
                all_speeds.extend([speed for speed in speeds if not pd.isna(speed)])
            for trial, accuracies in metrics['accuracy'].items():
                all_accuracies.extend([accuracy for accuracy in accuracies if not pd.isna(accuracy)])

    # Calculate Spearman correlation
    if len(all_speeds) > 1 and len(all_accuracies) > 1:
        spearman_corr, p_value = spearmanr(all_speeds, all_accuracies)
    else:
        spearman_corr, p_value = np.nan, np.nan

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(all_speeds, all_accuracies, alpha=0.7, color='blue')
    plt.title("Scatter Plot of All Speed vs All Accuracy")

    labels = {
        'distance': "Good → Bad (cm)",
        'accuracy': "Bad → Good (%)",
        'durations': "Good / Fast → Bad / Slow (s)",
        'speed': "Bad / Slow → Good / Fast (1/s)"
    }
    plt.xlabel(f"Speed ({labels.get('speed', '')})")
    plt.ylabel(f"Accuracy ({labels.get('accuracy', '')})")
    plt.grid(alpha=0.5)

    # Add Spearman correlation to the plot
    plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Fit a hyperbolic curve
    def hyperbolic_func(x, a, b):
        return a / (x + b)

    try:
        params, _ = curve_fit(hyperbolic_func, all_speeds, all_accuracies)
        x_fit = np.linspace(min(all_speeds), max(all_speeds), 500)
        y_fit = hyperbolic_func(x_fit, *params)
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
    except Exception as e:
        print(f"Hyperbolic fit failed: {e}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the scatter plot
scatter_plot_all_speed_accuracy(updated_metrics)
# -------------------------------------------------------------------------------------------------------------------

# Scatter plot for all durations and all distances with hyperbolic fit
def scatter_plot_all_duration_distance(updated_metrics):
    """
    Plots a scatter plot of all durations vs all distances across all subjects, hands, and trials,
    and fits a hyperbolic curve.

    Parameters:
        updated_metrics (dict): Updated metrics data.
    """
    all_durations = []
    all_distances = []

    for subject, hands_data in updated_metrics.items():
        for hand, metrics in hands_data.items():
            for trial, durations in metrics['durations'].items():
                all_durations.extend([duration for duration in durations if not pd.isna(duration)])
            for trial, distances in metrics['distance'].items():
                all_distances.extend([distance for distance in distances if not pd.isna(distance)])

    # Calculate Spearman correlation
    if len(all_durations) > 1 and len(all_distances) > 1:
        spearman_corr, p_value = spearmanr(all_durations, all_distances)
    else:
        spearman_corr, p_value = np.nan, np.nan

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(all_durations, all_distances, alpha=0.7, color='blue')
    plt.title("Scatter Plot of All Durations vs All Distances")

    labels = {
        'distance': "Good → Bad (cm)",
        'accuracy': "Bad → Good (%)",
        'durations': "Good / Fast → Bad / Slow (s)",
        'speed': "Bad / Slow → Good / Fast (1/s)"
    }
    plt.xlabel(f"Durations ({labels.get('durations', '')})")
    plt.ylabel(f"Distance ({labels.get('distance', '')})")
    plt.grid(alpha=0.5)

    # Add Spearman correlation to the plot
    plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Fit a hyperbolic curve
    def hyperbolic_func(x, a, b):
        return a / (x + b)

    try:
        params, _ = curve_fit(hyperbolic_func, all_durations, all_distances)
        x_fit = np.linspace(min(all_durations), max(all_durations), 500)
        y_fit = hyperbolic_func(x_fit, *params)
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
    except Exception as e:
        print(f"Hyperbolic fit failed: {e}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the scatter plot
scatter_plot_all_duration_distance(updated_metrics)

# -------------------------------------------------------------------------------------------------------------------

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

# Calculate and print statistics for each subject
subject_statistics = calculate_statistics_for_subjects(updated_metrics)

print("Subject Statistics:")
for subject, stats in subject_statistics.items():
    print(f"Subject: {subject}")
    print(f"  Max Durations: {stats['max_durations']:.2f}")
    print(f"  Max Distance: {stats['max_distance']:.2f}")

# Scatter plot for median durations vs median distances for all subjects with optional hyperbolic regression
def scatter_plot_statistics_values(statistics_values):
    """
    Plots a scatter plot of median durations vs median distances for all subjects, calculates Spearman correlation,
    and optionally fits a hyperbolic regression.

    Parameters:
        statistics_values (dict): Dictionary containing median durations and distances for each subject.
    """
    subjects = list(statistics_values.keys())
    median_durations = [statistics_values[subject]['median_durations'] for subject in subjects]
    median_distances = [statistics_values[subject]['median_distance'] for subject in subjects]

    # Calculate Spearman correlation
    if len(median_durations) > 1 and len(median_distances) > 1:
        spearman_corr, p_value = spearmanr(median_durations, median_distances)
    else:
        spearman_corr, p_value = np.nan, np.nan

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(median_durations, median_distances, color='blue', alpha=0.7, label="Subjects")

    # Add labels for each subject
    for i, subject in enumerate(subjects):
        plt.text(median_durations[i], median_distances[i], subject, fontsize=9, ha='right')

    plt.title("Scatter Plot of Median Durations vs Median Distances")
    labels = {
        'distance': "Good → Bad (cm)",
        'accuracy': "Bad → Good (%)",
        'durations': "Good / Fast → Bad / Slow (s)",
        'speed': "Bad / Slow → Good / Fast (1/s)"
    }
    plt.xlabel(f"Median Durations ({labels.get('durations', '')})")
    plt.ylabel(f"Median Distances ({labels.get('distance', '')})")
    plt.grid(alpha=0.5)

    # Add Spearman correlation to the plot
    plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Perform hyperbolic regression
    def hyperbolic_func(x, a, b):
        return a / (x + b)

    try:
        params, _ = curve_fit(hyperbolic_func, median_durations, median_distances)
        x_fit = np.linspace(min(median_durations), max(median_durations), 500)
        y_fit = hyperbolic_func(x_fit, *params)
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
    except Exception as e:
        print(f"Hyperbolic regression failed: {e}")

    plt.legend()
    plt.tight_layout()
    plt.show()

scatter_plot_statistics_values(subject_statistics)
# -----------------------------------------------------------------
# Calculate median, mean, and standard deviation for durations and distances for each subject by hand
def calculate_statistics_by_hand(updated_metrics):
    """
    Calculates median, mean, and standard deviation for durations and distances for each subject, separated by hand.

    Parameters:
        updated_metrics (dict): Updated metrics data.

    Returns:
        dict: Dictionary containing statistics for each subject and hand.
    """
    statistics_by_hand = {}

    for subject, hands_data in updated_metrics.items():
        statistics_by_hand[subject] = {}
        for hand, metrics in hands_data.items():
            all_durations = []
            all_distances = []

            for trial, durations in metrics['durations'].items():
                all_durations.extend([dur for dur in durations if not pd.isna(dur)])
            for trial, distances in metrics['distance'].items():
                all_distances.extend([dist for dist in distances if not pd.isna(dist)])

            statistics_by_hand[subject][hand] = {
                'median_duration': np.median(all_durations) if all_durations else np.nan,
                'mean_duration': np.mean(all_durations) if all_durations else np.nan,
                'std_duration': np.std(all_durations) if all_durations else np.nan,
                'median_distance': np.median(all_distances) if all_distances else np.nan,
                'mean_distance': np.mean(all_distances) if all_distances else np.nan,
                'std_distance': np.std(all_distances) if all_distances else np.nan,
            }

    return statistics_by_hand

# Calculate and print statistics for each subject and hand
statistics_by_hand = calculate_statistics_by_hand(updated_metrics)


table = PrettyTable()
table.field_names = ["Subject", "Hand", "Median Duration", "Mean Duration", "Std Duration", 
                     "Median Distance", "Mean Distance", "Std Distance"]

for subject, hands_data in statistics_by_hand.items():
    for hand, stats in hands_data.items():
        table.add_row([
            subject,
            hand,
            f"{stats['median_duration']:.2f}",
            f"{stats['mean_duration']:.2f}",
            f"{stats['std_duration']:.2f}",
            f"{stats['median_distance']:.2f}",
            f"{stats['mean_distance']:.2f}",
            f"{stats['std_distance']:.2f}"
        ])

# Print the table
print(table)

# Scatter plot for median durations vs median distances for all subjects and hands with optional hyperbolic regression
def scatter_plot_statistics_by_hand(statistics_by_hand):
    """
    Plots a scatter plot of median durations vs median distances for all subjects and hands, calculates Spearman correlation,
    and optionally fits a hyperbolic regression. Each subject is assigned a unique color, and hands are differentiated by marker style.

    Parameters:
        statistics_by_hand (dict): Dictionary containing median durations and distances for each subject and hand.
    """
    subjects_hands = []
    median_durations = []
    median_distances = []
    colors = []
    markers = {'left': 'o', 'right': 's'}  # Different markers for left and right hands
    alphas = {'left': 0.5, 'right': 0.9}  # Lighter for left hand, darker for right hand
    color = {'left': 'blue', 'right': 'orange'}  # Assign colors for left and right hands

    # Assign a unique color to each subject
    # subject_colors = {subject: color for subject, color in zip(statistics_by_hand.keys(), sns.color_palette("tab10", len(statistics_by_hand)))}

    for subject, hands_data in statistics_by_hand.items():
        for hand, stats in hands_data.items():
            subjects_hands.append(f"{subject} ({hand})")
            median_durations.append(stats['median_duration'])
            median_distances.append(stats['median_distance'])
            # colors.append(subject_colors[subject])

    # Calculate Spearman correlation
    if len(median_durations) > 1 and len(median_distances) > 1:
        spearman_corr, p_value = spearmanr(median_durations, median_distances)
    else:
        spearman_corr, p_value = np.nan, np.nan

    # Scatter plot
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(subjects_hands):
        hand = label.split('(')[-1].strip(')')
        # plt.scatter(median_durations[i], median_distances[i], color=colors[i], alpha=alphas[hand], marker=markers[hand], label=label if hand == 'left' else None)
        plt.scatter(median_durations[i], median_distances[i], alpha=alphas[hand], marker=markers[hand],  color=color[hand], label=label if hand == 'left' else None)


    plt.title("Scatter Plot of Median Durations vs Median Distances (By Hand)")
    labels = {
        'distance': "Good → Bad (cm)",
        'accuracy': "Bad → Good (%)",
        'durations': "Good / Fast → Bad / Slow (s)",
        'speed': "Bad / Slow → Good / Fast (1/s)"
    }
    plt.xlabel(f"Median Durations ({labels.get('durations', '')})")
    plt.ylabel(f"Median Distances ({labels.get('distance', '')})")
    plt.grid(alpha=0.5)

    # Add Spearman correlation to the plot
    plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Perform hyperbolic regression
    def hyperbolic_func(x, a, b):
        return a / (x + b)

    try:
        params, _ = curve_fit(hyperbolic_func, median_durations, median_distances)
        x_fit = np.linspace(min(median_durations), max(median_durations), 500)
        y_fit = hyperbolic_func(x_fit, *params)
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
    except Exception as e:
        print(f"Hyperbolic regression failed: {e}")

    plt.tight_layout()
    plt.show()

scatter_plot_statistics_by_hand(statistics_by_hand)


# Scatter plot for median durations vs median distances for all subjects and hands with optional hyperbolic regression
def scatter_plot_statistics_by_hand(statistics_by_hand):
    """
    Plots two subplots of median durations vs median distances for left and right hands.
    Each subject is assigned a unique color, and hands are differentiated by subplots.
    Includes Spearman correlation and hyperbolic fit for each subplot.

    Parameters:
        statistics_by_hand (dict): Dictionary containing median durations and distances for each subject and hand.
    """
    subjects = list(statistics_by_hand.keys())
    subject_colors = {subject: color for subject, color in zip(subjects, sns.color_palette("tab10", len(subjects)))}

    # Prepare data for left and right hands
    left_durations, left_distances, left_labels, left_colors = [], [], [], []
    right_durations, right_distances, right_labels, right_colors = [], [], [], []

    for subject, hands_data in statistics_by_hand.items():
        for hand, stats in hands_data.items():
            if hand == 'left':
                left_durations.append(stats['median_duration'])
                left_distances.append(stats['median_distance'])
                left_labels.append(subject)
                left_colors.append(subject_colors[subject])
            elif hand == 'right':
                right_durations.append(stats['median_duration'])
                right_distances.append(stats['median_distance'])
                right_labels.append(subject)
                right_colors.append(subject_colors[subject])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axes[0].set_title("Left Hand")
    axes[1].set_title("Right Hand")

    # Helper function to add Spearman correlation and hyperbolic fit
    def add_statistics(ax, durations, distances, title):
        # Calculate Spearman correlation
        if len(durations) > 1 and len(distances) > 1:
            spearman_corr, p_value = spearmanr(durations, distances)
        else:
            spearman_corr, p_value = np.nan, np.nan

        # Scatter plot
        for i, subject in enumerate(subjects):
            subject_durations = [durations[j] for j in range(len(durations)) if left_labels[j] == subject]
            subject_distances = [distances[j] for j in range(len(distances)) if left_labels[j] == subject]
            ax.scatter(subject_durations, subject_distances, color=subject_colors[subject], label=subject, alpha=0.7)
        ax.set_xlabel("Median Durations (s)")
        ax.set_ylabel("Median Distances (cm)")
        ax.grid(alpha=0.5)
        ax.set_title(title)


        # Perform hyperbolic regression
        def hyperbolic_func(x, a, b):
            return a / (x + b)

        try:
            params, _ = curve_fit(hyperbolic_func, durations, distances)
            x_fit = np.linspace(min(durations), max(durations), 500)
            y_fit = hyperbolic_func(x_fit, *params)
            ax.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")

            # ax.legend()
        except Exception as e:
            print(f"Hyperbolic regression failed: {e}")

        # Add Spearman correlation to the plot
        ax.text(0.95, 0.05, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}\nHyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Plot left hand data
    add_statistics(axes[0], left_durations, left_distances, "Left Hand")

    # Plot right hand data
    add_statistics(axes[1], right_durations, right_distances, "Right Hand")

    # # Add legend for subjects
    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=subject_colors[subject], markersize=10)
    #            for subject in subjects]
    # labels = subjects
    # fig.legend(handles, labels, loc='upper center', ncol=len(subjects), bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.show()

scatter_plot_statistics_by_hand(statistics_by_hand)

# -------------------------------------------------------------------------------------------------------------------

# Scatter plot for selected metrics with Spearman correlation and optional hyperbolic regression
def scatter_plot_with_options(updated_metrics, subject, hand, metric_x, metric_y, reach_indices, subject_statistics):
    """
    Parameters:
        updated_metrics (dict): Updated metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        reach_indices (int or list): Reach index or list of indices to include.
        subject_statistics (dict): Statistics values for the subject.
    """
    if isinstance(reach_indices, int):
        reach_indices = [reach_indices]  # Convert single index to list

    x_values = []
    y_values = []
    trial_colors = []

    trials = updated_metrics[subject][hand][metric_x].keys()
    color_palette = sns.color_palette("Reds", len(trials))  # Generate a color palette from light to dark

    for i, trial in enumerate(trials):
        trial_x = np.array(updated_metrics[subject][hand][metric_x][trial])
        trial_y = np.array(updated_metrics[subject][hand][metric_y][trial])

        # Collect data for the specified reach indices
        for reach_index in reach_indices:
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

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c=trial_colors, alpha=0.7, label=f"Reach {reach_indices}")
    plt.title(f"{metric_y.capitalize()} vs {metric_x.capitalize()} Scatter Plot")

    labels = {
        'distance': "Good → Bad (cm)",
        'accuracy': "Bad → Good (%)",
        'durations': "Good / Fast → Bad / Slow (s)",
        'speed': "Bad / Slow → Good / Fast (1/s)"
    }
    plt.xlabel(f"{metric_x.capitalize()} ({labels.get(metric_x, '')})")
    plt.ylabel(f"{metric_y.capitalize()} ({labels.get(metric_y, '')})")

    # Add Spearman correlation and p-value to the plot
    plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Perform hyperbolic regression if applicable
    if metric_x == 'durations' and metric_y == 'distance':
        def hyperbolic_func(x, a, b):
            return a / (x + b)

        try:
            params, _ = curve_fit(hyperbolic_func, x_values, y_values)
            x_fit = np.linspace(min(x_values), max(x_values), 500)
            y_fit = hyperbolic_func(x_fit, *params)
            plt.plot(x_fit, y_fit, color='blue', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
        except Exception as e:
            print(f"Hyperbolic regression failed: {e}")

    # Add diagonal line from (0, 0) to max duration and max distance
    max_duration = subject_statistics[subject]['max_durations']
    max_distance = subject_statistics[subject]['max_distance']
    if not np.isnan(max_duration) and not np.isnan(max_distance):
        plt.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")

    plt.legend()
    plt.tight_layout()
    plt.show()

scatter_plot_with_options(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', 4, subject_statistics)

# -------------------------------------------------------------------------------------------------------------------

# Scatter plot for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation
def scatter_plot_all_reach_indices(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
    """
    Plots scatter plots for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        subject_statistics (dict): Statistics containing max durations and max distances for the subject.
    """
    # Create subplots in a 4x4 layout
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes = axes.flatten()

    # Get max duration and max distance for the subject
    max_duration = subject_statistics[subject]['max_durations']
    max_distance = subject_statistics[subject]['max_distance']

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

        # Perform hyperbolic regression if applicable
        if metric_x == 'durations' and metric_y == 'distance':
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
                                fontsize=12, color="black", ha="right", va="bottom")
            except Exception as e:
                print(f"Hyperbolic regression failed for Reach Index {reach_index}: {e}")
        else:
            print(f"Error: Hyperbolic regression is only applicable for 'durations' vs 'distance'.")

        # Add Spearman correlation to the plot
        ax.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Add diagonal line from (0, 0) to max duration and max distance
        if not np.isnan(max_duration) and not np.isnan(max_distance):
            ax.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")

    # Hide unused subplots
    for i in range(16, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
scatter_plot_all_reach_indices(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', subject_statistics)

# Overlay 16 hyperbolic regressions, intersections, and diagonal line in one figure
def overlay_hyperbolic_regressions(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
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
    plt.figure(figsize=(10, 8))

    # Generate a color palette from light blue to black
    color_palette = sns.color_palette("light:black", 16)

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
                    plt.scatter(xi, yi, color=color_palette[reach_index], s=50, zorder=5)
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
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

# Example usage
results = overlay_hyperbolic_regressions(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', subject_statistics)
# -------------------------------------------------------------------------------------------------------------------
def overlay_hyperbolic_regressions_all_subjects(updated_metrics, subjects, hand, metric_x, metric_y, subject_statistics):
    """
    Overlays 16 hyperbolic regressions, intersections, and diagonal line in one figure for all subjects.
    Calculates and returns Spearman correlation, p-values, and intersections for each reach index.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subjects (list): List of subject identifiers.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        subject_statistics (dict): Statistics containing max durations and max distances for each subject.

    Returns:
        dict: Results containing Spearman correlation, p-values, and intersections for each subject and reach index.
    """
    all_results = {}

    for subject in subjects:
        print(f"Processing subject: {subject}")
        # Initialize results storage
        results = []

        # Get max duration and max distance for the subject
        max_duration = subject_statistics[subject]['max_durations']
        max_distance = subject_statistics[subject]['max_distance']

        # Create a figure
        plt.figure(figsize=(10, 8))

        # Generate a color palette from light blue to black
        color_palette = sns.color_palette("light:black", 16)

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
                        plt.scatter(xi, yi, color=color_palette[reach_index], s=50, zorder=5)

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
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Store results for the subject
        all_results[subject] = results

    return all_results

# Example usage for all subjects and both hands
subjects = list(updated_metrics.keys())
results_right = overlay_hyperbolic_regressions_all_subjects(updated_metrics, subjects, 'right', 'durations', 'distance', subject_statistics)
results_left = overlay_hyperbolic_regressions_all_subjects(updated_metrics, subjects, 'left', 'durations', 'distance', subject_statistics)


def overlay_hyperbolic_regressions_all_subjects(updated_metrics, subjects, hand, metric_x, metric_y, subject_statistics):
    """
    Overlays 16 hyperbolic regressions, intersections, and diagonal line in subplots for each subject and hand.
    Calculates and returns Spearman correlation, p-values, and intersections for each reach index.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subjects (list): List of subject identifiers.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        subject_statistics (dict): Statistics containing max durations and max distances for each subject.

    Returns:
        dict: Results containing Spearman correlation, p-values, and intersections for each subject and reach index.
    """
    all_results = {}

    # Create subplots for each subject
    num_subjects = len(subjects)
    rows, cols = (num_subjects // 4) + (num_subjects % 4 > 0), 4
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes = axes.flatten()

    for idx, subject in enumerate(subjects):
        print(f"Processing subject: {subject}")
        # Initialize results storage
        results = []

        # Get max duration and max distance for the subject
        max_duration = subject_statistics[subject]['max_durations']
        max_distance = subject_statistics[subject]['max_distance']

        ax = axes[idx]
        ax.set_title(f"Subject: {subject} ({hand.capitalize()})")

        # Generate a color palette from light blue to black
        color_palette = sns.color_palette("light:black", 16)

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
                ax.plot(x_fit, y_fit, label=f"Reach {reach_index}", color=color_palette[reach_index], alpha=0.7)

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
                        ax.scatter(xi, yi, color=color_palette[reach_index], s=50, zorder=5)

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
            ax.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")

        ax.set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
        ax.set_ylabel(f"{metric_y.capitalize()} (Bad → Good)")
        # ax.legend()

        # Store results for the subject
        all_results[subject] = results

    # Hide unused subplots
    for i in range(len(subjects), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return all_results

# Example usage for all subjects and both hands
subjects = list(updated_metrics.keys())
results_right = overlay_hyperbolic_regressions_all_subjects(updated_metrics, subjects, 'right', 'durations', 'distance', subject_statistics)
results_left = overlay_hyperbolic_regressions_all_subjects(updated_metrics, subjects, 'left', 'durations', 'distance', subject_statistics)
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
def calculate_distances_to_intersections(results_right, results_left):
    """
    Calculates the distance from (0, 0) to the intersection points for each reach index
    for both right and left hands, and returns the results as a dictionary.

    Parameters:
        results_right (dict): Results containing intersections for the right hand.
        results_left (dict): Results containing intersections for the left hand.

    Returns:
        dict: Dictionary containing distances to intersections for each subject, hand, and reach index.
    """
    distances = {}

    for subject in results_right.keys():
        distances[subject] = {"right": {}, "left": {}}

        # Process right hand
        for reach_result in results_right[subject]:
            reach_index = reach_result["reach_index"]
            intersections = reach_result.get("intersections", [])
            if intersections:
                # Calculate distance from (0, 0) to the first intersection
                xi, yi = intersections[0]
                distance = np.sqrt(xi**2 + yi**2)
                distances[subject]["right"][reach_index] = distance
            else:
                distances[subject]["right"][reach_index] = None

        # Process left hand
        for reach_result in results_left[subject]:
            reach_index = reach_result["reach_index"]
            intersections = reach_result.get("intersections", [])
            if intersections:
                # Calculate distance from (0, 0) to the first intersection
                xi, yi = intersections[0]
                distance = np.sqrt(xi**2 + yi**2)
                distances[subject]["left"][reach_index] = distance
            else:
                distances[subject]["left"][reach_index] = None

    return distances

# Example usage
distances_to_intersections = calculate_distances_to_intersections(results_right, results_left)
print(distances_to_intersections)

def calculate_average_ldlj_sparc_all_subjects(updated_metrics):
    """
    Calculates the average LDLJ and SPARC for all subjects, hands, and each reach type.

    Parameters:
        updated_metrics (dict): Updated metrics data.

    Returns:
        tuple: Two dictionaries containing average LDLJ and average SPARC for each subject, hand, and reach type.
    """
    all_average_ldlj = {}
    all_average_sparc = {}

    for subject, hands_data in updated_metrics.items():
        all_average_ldlj[subject] = {}
        all_average_sparc[subject] = {}
        for hand, metrics in hands_data.items():
            ldlj = {}
            sparc = {}
            for reach_index in range(16):
                reach_ldlj = []
                reach_sparc = []

                trials = metrics['ldlj'].keys()

                for trial in trials:
                    trial_ldlj = np.array(metrics['ldlj'][trial])
                    trial_sparc = np.array(metrics['sparc'][trial])

                    if reach_index < len(trial_ldlj) and reach_index < len(trial_sparc):
                        reach_ldlj.append(trial_ldlj[reach_index])
                        reach_sparc.append(trial_sparc[reach_index])

                # Remove NaN values
                reach_ldlj = np.array(reach_ldlj)
                reach_sparc = np.array(reach_sparc)
                valid_indices = ~np.isnan(reach_ldlj) & ~np.isnan(reach_sparc)
                reach_ldlj = reach_ldlj[valid_indices]
                reach_sparc = reach_sparc[valid_indices]

                # Calculate averages
                avg_ldlj = np.mean(reach_ldlj) if len(reach_ldlj) > 0 else np.nan
                avg_sparc = np.mean(reach_sparc) if len(reach_sparc) > 0 else np.nan

                ldlj[reach_index] = avg_ldlj
                sparc[reach_index] = avg_sparc

            all_average_ldlj[subject][hand] = ldlj
            all_average_sparc[subject][hand] = sparc

    return all_average_ldlj, all_average_sparc

def plot_average_ldlj_sparc_all_subjects(all_average_ldlj, all_average_sparc):
    """
    Plots scatter plots of average LDLJ vs average SPARC for each reach type for all subjects and hands.

    Parameters:
        all_average_ldlj (dict): Dictionary containing average LDLJ for each subject, hand, and reach type.
        all_average_sparc (dict): Dictionary containing average SPARC for each subject, hand, and reach type.
    """
    for subject, hands_data in all_average_ldlj.items():
        for hand, ldlj in hands_data.items():
            reach_indices = list(ldlj.keys())
            avg_ldlj = [ldlj[reach] for reach in reach_indices]
            avg_sparc = [all_average_sparc[subject][hand][reach] for reach in reach_indices]

            plt.figure(figsize=(8, 6))
            plt.scatter(avg_ldlj, avg_sparc, color='blue', alpha=0.7)
            for i, reach_index in enumerate(reach_indices):
                plt.text(avg_ldlj[i], avg_sparc[i], str(reach_index), fontsize=9, ha='right')

            plt.title(f"Scatter Plot of Average LDLJ vs Average SPARC\nSubject: {subject}, Hand: {hand.capitalize()}")
            plt.xlabel("Average LDLJ")
            plt.ylabel("Average SPARC")
            plt.grid(alpha=0.5)
            plt.tight_layout()
            plt.show()

# Example usage
all_average_ldlj, all_average_sparc = calculate_average_ldlj_sparc_all_subjects(updated_metrics)
plot_average_ldlj_sparc_all_subjects(all_average_ldlj, all_average_sparc)
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# Calculate correlation between all_average_ldlj and distances_to_intersections for each subject
def calculate_correlation_ldlj_intersections(all_average_ldlj, distances_to_intersections):
    """
    Calculates the correlation between average LDLJ and distances to intersections for each subject.

    Parameters:
        all_average_ldlj (dict): Dictionary containing average LDLJ for each subject, hand, and reach type.
        distances_to_intersections (dict): Dictionary containing distances to intersections for each subject, hand, and reach type.

    Returns:
        dict: Spearman correlation results for each subject and hand.
    """
    correlation_results = {}

    for subject in all_average_ldlj.keys():
        correlation_results[subject] = {}
        for hand in all_average_ldlj[subject].keys():
            ldlj_values = []
            intersection_distances = []

            for reach_index in range(16):
                avg_ldlj = all_average_ldlj[subject][hand].get(reach_index, np.nan)
                intersection_distance = distances_to_intersections[subject][hand].get(reach_index, np.nan)

                if not np.isnan(avg_ldlj) and not np.isnan(intersection_distance):
                    ldlj_values.append(avg_ldlj)
                    intersection_distances.append(intersection_distance)

            # Calculate Spearman correlation
            if len(ldlj_values) > 1 and len(intersection_distances) > 1:
                spearman_corr, p_value = spearmanr(ldlj_values, intersection_distances)
            else:
                spearman_corr, p_value = np.nan, np.nan

            correlation_results[subject][hand] = {
                "spearman_corr": spearman_corr,
                "p_value": p_value,
                "ldlj_values": ldlj_values,
                "intersection_distances": intersection_distances
            }

    return correlation_results

def plot_correlation_ldlj_intersections(correlation_results):
    """
    Plots scatter plots of average LDLJ vs distances to intersections for each subject and hand.

    Parameters:
        correlation_results (dict): Spearman correlation results for each subject and hand.
    """
    for subject, hands_data in correlation_results.items():
        for hand, result in hands_data.items():
            ldlj_values = result["ldlj_values"]
            intersection_distances = result["intersection_distances"]
            spearman_corr = result["spearman_corr"]
            p_value = result["p_value"]

            if len(ldlj_values) > 1 and len(intersection_distances) > 1:
                plt.figure(figsize=(8, 6))
                plt.scatter(ldlj_values, intersection_distances, color='blue', alpha=0.7)
                plt.title(f"Scatter Plot of Average LDLJ vs Distances to Intersections\n"
                          f"Subject: {subject}, Hand: {hand.capitalize()}\n"
                          f"Spearman Corr: {spearman_corr:.2f}, P-value: {p_value:.2f}")
                plt.xlabel("Average LDLJ")
                plt.ylabel("Distances to Intersections")
                plt.grid(alpha=0.5)
                plt.tight_layout()
                plt.show()

# Example usage
correlation_ldlj_intersections = calculate_correlation_ldlj_intersections(all_average_ldlj, distances_to_intersections)

# Plot the scatter plots
plot_correlation_ldlj_intersections(correlation_ldlj_intersections)

# Plot histogram of Spearman correlations for left and right hands
def plot_histogram_of_correlations_ldlj_intersections(correlation_ldlj_intersections):
    """
    Plots histograms of Spearman correlations for left and right hands across all subjects
    based on the correlation between LDLJ and distances to intersections.

    Parameters:
        correlation_ldlj_intersections (dict): Results containing Spearman correlations for each subject and hand.
    """
    correlations_left = []
    correlations_right = []

    for subject, hands_data in correlation_ldlj_intersections.items():
        if 'left' in hands_data and hands_data['left']['spearman_corr'] is not None:
            correlations_left.append(hands_data['left']['spearman_corr'])
        if 'right' in hands_data and hands_data['right']['spearman_corr'] is not None:
            correlations_right.append(hands_data['right']['spearman_corr'])

    # Calculate medians
    median_left = np.median(correlations_left) if correlations_left else np.nan
    median_right = np.median(correlations_right) if correlations_right else np.nan

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')

    # Add median lines
    plt.axvline(median_left, color='orange', linestyle='--', label=f'Median Left: {median_left:.2f}')
    plt.axvline(median_right, color='blue', linestyle='--', label=f'Median Right: {median_right:.2f}')

    # Add labels, title, and legend
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Frequency')
    plt.title('Histogram of Spearman Correlations for LDLJ vs Intersections (Left and Right Hands)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the histogram
plot_histogram_of_correlations_ldlj_intersections(correlation_ldlj_intersections)


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# Calculate correlation between all_average_sparc and distances_to_intersections for each subject
def calculate_correlation_sparc_intersections(all_average_sparc, distances_to_intersections):
    """
    Calculates the correlation between average SPARC and distances to intersections for each subject.

    Parameters:
        all_average_sparc (dict): Dictionary containing average SPARC for each subject, hand, and reach type.
        distances_to_intersections (dict): Dictionary containing distances to intersections for each subject, hand, and reach type.

    Returns:
        dict: Spearman correlation results for each subject and hand.
    """
    correlation_results = {}

    for subject in all_average_sparc.keys():
        correlation_results[subject] = {}
        for hand in all_average_sparc[subject].keys():
            sparc_values = []
            intersection_distances = []

            for reach_index in range(16):
                avg_sparc = all_average_sparc[subject][hand].get(reach_index, np.nan)
                intersection_distance = distances_to_intersections[subject][hand].get(reach_index, np.nan)

                if not np.isnan(avg_sparc) and not np.isnan(intersection_distance):
                    sparc_values.append(avg_sparc)
                    intersection_distances.append(intersection_distance)

            # Calculate Spearman correlation
            if len(sparc_values) > 1 and len(intersection_distances) > 1:
                spearman_corr, p_value = spearmanr(sparc_values, intersection_distances)
            else:
                spearman_corr, p_value = np.nan, np.nan

            correlation_results[subject][hand] = {
                "spearman_corr": spearman_corr,
                "p_value": p_value,
                "sparc_values": sparc_values,
                "intersection_distances": intersection_distances
            }

    return correlation_results

def plot_correlation_sparc_intersections(correlation_results):
    """
    Plots scatter plots of average SPARC vs distances to intersections for each subject and hand.

    Parameters:
        correlation_results (dict): Spearman correlation results for each subject and hand.
    """
    for subject, hands_data in correlation_results.items():
        for hand, result in hands_data.items():
            sparc_values = result["sparc_values"]
            intersection_distances = result["intersection_distances"]
            spearman_corr = result["spearman_corr"]
            p_value = result["p_value"]

            if len(sparc_values) > 1 and len(intersection_distances) > 1:
                plt.figure(figsize=(8, 6))
                plt.scatter(sparc_values, intersection_distances, color='blue', alpha=0.7)
                plt.title(f"Scatter Plot of Average SPARC vs Distances to Intersections\n"
                          f"Subject: {subject}, Hand: {hand.capitalize()}\n"
                          f"Spearman Corr: {spearman_corr:.2f}, P-value: {p_value:.2f}")
                plt.xlabel("Average SPARC")
                plt.ylabel("Distances to Intersections")
                plt.grid(alpha=0.5)
                plt.tight_layout()
                plt.show()

# Example usage
correlation_sparc_intersections = calculate_correlation_sparc_intersections(all_average_sparc, distances_to_intersections)

# Plot the scatter plots
plot_correlation_sparc_intersections(correlation_sparc_intersections)

# Plot histogram of Spearman correlations for left and right hands
def plot_histogram_of_correlations_sparc_intersections(correlation_sparc_intersections):
    """
    Plots histograms of Spearman correlations for left and right hands across all subjects
    based on the correlation between SPARC and distances to intersections.

    Parameters:
        correlation_sparc_intersections (dict): Results containing Spearman correlations for each subject and hand.
    """
    correlations_left = []
    correlations_right = []

    for subject, hands_data in correlation_sparc_intersections.items():
        if 'left' in hands_data and hands_data['left']['spearman_corr'] is not None:
            correlations_left.append(hands_data['left']['spearman_corr'])
        if 'right' in hands_data and hands_data['right']['spearman_corr'] is not None:
            correlations_right.append(hands_data['right']['spearman_corr'])

    # Calculate medians
    median_left = np.median(correlations_left) if correlations_left else np.nan
    median_right = np.median(correlations_right) if correlations_right else np.nan

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')

    # Add median lines
    plt.axvline(median_left, color='orange', linestyle='--', label=f'Median Left: {median_left:.2f}')
    plt.axvline(median_right, color='blue', linestyle='--', label=f'Median Right: {median_right:.2f}')

    # Add labels, title, and legend
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Frequency')
    plt.title('Histogram of Spearman Correlations for SPARC vs Intersections (Left and Right Hands)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the histogram
plot_histogram_of_correlations_sparc_intersections(correlation_sparc_intersections)

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# Plot histogram of each participant's Spearman correlations as subplots
def plot_histogram_of_correlations_per_participant_with_averages(results_right, results_left):
    """
    Plots histograms of Spearman correlations for each participant's right and left hand as subplots,
    calculates the average correlation across the 16 reach types for each subject and hand, and overlays the averages.

    Parameters:
        results_right (dict): Results containing Spearman correlations for the right hand.
        results_left (dict): Results containing Spearman correlations for the left hand.

    Returns:
        dict: Averages of Spearman correlations for each subject and hand.
    """
    num_subjects = len(results_right.keys())
    rows, cols = (num_subjects // 4) + (num_subjects % 4 > 0), 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    averages = {}

    for idx, subject in enumerate(results_right.keys()):
        correlations_right = []
        correlations_left = []

        # Extract Spearman correlations for right hand
        for reach_result in results_right[subject]:
            if not np.isnan(reach_result["spearman_corr"]):
                correlations_right.append(reach_result["spearman_corr"])

        # Extract Spearman correlations for left hand
        for reach_result in results_left[subject]:
            if not np.isnan(reach_result["spearman_corr"]):
                correlations_left.append(reach_result["spearman_corr"])

        # Calculate averages
        avg_right = np.mean(correlations_right) if correlations_right else np.nan
        avg_left = np.mean(correlations_left) if correlations_left else np.nan
        averages[subject] = {"right": avg_right, "left": avg_left}

        # Plot histograms for the current participant
        ax = axes[idx]
        ax.hist(correlations_right, bins=20, color='blue', alpha=0.7, edgecolor='black', label="Right Hand")
        ax.hist(correlations_left, bins=20, color='orange', alpha=0.7, edgecolor='black', label="Left Hand")
        ax.axvline(avg_right, color='blue', linestyle='--', label=f"Avg Right: {avg_right:.2f}")
        ax.axvline(avg_left, color='orange', linestyle='--', label=f"Avg Left: {avg_left:.2f}")
        ax.set_title(f"Spearman Correlations - {subject}")
        ax.set_xlabel("Spearman Correlation")
        ax.set_ylabel("Frequency")
        ax.legend()

    # Hide unused subplots
    for i in range(len(results_right.keys()), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return averages

# Call the function to plot histograms for each participant and get averages
averages = plot_histogram_of_correlations_per_participant_with_averages(results_right, results_left)

# Plot the Average Spearman Correlations for all participants
def plot_average_spearman_correlations(averages):
    """
    Plots the average Spearman correlations for all participants, separated by right and left hands.

    Parameters:
        averages (dict): Averages of Spearman correlations for each subject and hand.
    """
    subjects = list(averages.keys())
    avg_right = [averages[subject]["right"] for subject in subjects]
    avg_left = [averages[subject]["left"] for subject in subjects]

    x = np.arange(len(subjects))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_right = ax.bar(x - width / 2, avg_right, width, label='Right Hand', color='blue', alpha=0.7)
    bars_left = ax.bar(x + width / 2, avg_left, width, label='Left Hand', color='orange', alpha=0.7)

    # Add labels, title, and legend
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Average Spearman Correlation')
    ax.set_title('Average Spearman Correlations by Subject and Hand')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.legend()

    # Add value annotations on bars
    for bar in bars_right:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for bar in bars_left:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

# Call the function to plot the average Spearman correlations
plot_average_spearman_correlations(averages)

# Plot the Average Spearman Correlations for all participants as a histogram
def plot_average_spearman_correlations_histogram(averages):
    """
    Plots the average and median Spearman correlations for all participants as a histogram,
    separated by right and left hands.

    Parameters:
        averages (dict): Averages of Spearman correlations for each subject and hand.
    """
    subjects = list(averages.keys())
    avg_right = [averages[subject]["right"] for subject in subjects]
    avg_left = [averages[subject]["left"] for subject in subjects]

    # Calculate medians
    median_right = np.median(avg_right) if avg_right else np.nan
    median_left = np.median(avg_left) if avg_left else np.nan

    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(avg_right, bins=10, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
    plt.hist(avg_left, bins=10, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')

    # Add median lines
    plt.axvline(median_right, color='blue', linestyle='--', label=f'Median Right: {median_right:.2f}')
    plt.axvline(median_left, color='orange', linestyle='--', label=f'Median Left: {median_left:.2f}')

    # Add labels, title, and legend
    plt.xlabel('Average Spearman Correlation')
    plt.ylabel('Frequency')
    plt.title('Histogram of Average Spearman Correlations by Hand')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the histogram of average Spearman correlations
plot_average_spearman_correlations_histogram(averages)

# -------------------------------------------------------------------------------------------------------------------
# Plot histogram of each participant's Spearman correlations as subplots
def plot_histogram_of_correlations_per_participant_with_averages(results_right, results_left):
    """
    Plots histograms of Spearman correlations for each participant's right and left hand as subplots,
    calculates the average correlation across the 16 reach types for each subject and hand, and overlays the averages.

    Parameters:
        results_right (dict): Results containing Spearman correlations for the right hand.
        results_left (dict): Results containing Spearman correlations for the left hand.

    Returns:
        dict: Averages of Spearman correlations for each subject and hand.
    """
    num_subjects = len(results_right.keys())
    rows, cols = (num_subjects // 4) + (num_subjects % 4 > 0), 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    averages = {}

    for idx, subject in enumerate(results_right.keys()):
        correlations_right = []
        correlations_left = []

        # Extract Spearman correlations for right hand
        for reach_result in results_right[subject]:
            if not np.isnan(reach_result["spearman_corr"]) and reach_result["spearman_corr"] < 0:
                correlations_right.append(reach_result["spearman_corr"])

        # Extract Spearman correlations for left hand
        for reach_result in results_left[subject]:
            if not np.isnan(reach_result["spearman_corr"]) and reach_result["spearman_corr"] < 0:
                correlations_left.append(reach_result["spearman_corr"])

        # Calculate averages
        avg_right = np.mean(correlations_right) if correlations_right else np.nan
        avg_left = np.mean(correlations_left) if correlations_left else np.nan
        averages[subject] = {"right": avg_right, "left": avg_left}

        # Plot histograms for the current participant
        ax = axes[idx]
        ax.hist(correlations_right, bins=20, color='blue', alpha=0.7, edgecolor='black', label="Right Hand")
        ax.hist(correlations_left, bins=20, color='orange', alpha=0.7, edgecolor='black', label="Left Hand")
        ax.axvline(avg_right, color='blue', linestyle='--', label=f"Avg Right: {avg_right:.2f}")
        ax.axvline(avg_left, color='orange', linestyle='--', label=f"Avg Left: {avg_left:.2f}")
        ax.set_title(f"Spearman Correlations - {subject}")
        ax.set_xlabel("Spearman Correlation")
        ax.set_ylabel("Frequency")
        ax.legend()

    # Hide unused subplots
    for i in range(len(results_right.keys()), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return averages

# Call the function to plot histograms for each participant and get averages
averages = plot_histogram_of_correlations_per_participant_with_averages(results_right, results_left)

# Plot the Average Spearman Correlations for all participants
def plot_average_spearman_correlations(averages):
    """
    Plots the average Spearman correlations for all participants, separated by right and left hands.

    Parameters:
        averages (dict): Averages of Spearman correlations for each subject and hand.
    """
    subjects = list(averages.keys())
    avg_right = [averages[subject]["right"] for subject in subjects]
    avg_left = [averages[subject]["left"] for subject in subjects]

    x = np.arange(len(subjects))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_right = ax.bar(x - width / 2, avg_right, width, label='Right Hand', color='blue', alpha=0.7)
    bars_left = ax.bar(x + width / 2, avg_left, width, label='Left Hand', color='orange', alpha=0.7)

    # Add labels, title, and legend
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Average Spearman Correlation')
    ax.set_title('Average Spearman Correlations by Subject and Hand')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.legend()

    # Add value annotations on bars
    for bar in bars_right:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for bar in bars_left:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

# Call the function to plot the average Spearman correlations
plot_average_spearman_correlations(averages)

# Plot the Average Spearman Correlations for all participants as a histogram
def plot_average_spearman_correlations_histogram(averages):
    """
    Plots the average and median Spearman correlations for all participants as a histogram,
    separated by right and left hands.

    Parameters:
        averages (dict): Averages of Spearman correlations for each subject and hand.
    """
    subjects = list(averages.keys())
    avg_right = [averages[subject]["right"] for subject in subjects]
    avg_left = [averages[subject]["left"] for subject in subjects]

    # Calculate medians
    median_right = np.median(avg_right) if avg_right else np.nan
    median_left = np.median(avg_left) if avg_left else np.nan

    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(avg_right, bins=18, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
    plt.hist(avg_left, bins=18, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')

    # Add median lines
    plt.axvline(median_right, color='blue', linestyle='--', label=f'Median Right: {median_right:.2f}')
    plt.axvline(median_left, color='orange', linestyle='--', label=f'Median Left: {median_left:.2f}')

    # Add labels, title, and legend
    plt.xlabel('Average Spearman Correlation')
    plt.ylabel('Frequency')
    plt.yticks(np.arange(0, 5, 1))
    plt.title('Histogram of Average Spearman Correlations by Hand')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the histogram of average Spearman correlations
plot_average_spearman_correlations_histogram(averages)


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# Scatter plot for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation
def scatter_plot_all_reach_indices(updated_metrics, subject, hand, metric_x, metric_y):
    """
    Plots scatter plots for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation.
    Calculates average and median 'durations', and average and median 'distance' for each reach_index, and plots these values on the figure.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
    """
    # Create subplots in a 4x4 layout
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes = axes.flatten()

    # Initialize storage for statistics
    reach_statistics = []

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

        # Calculate statistics
        avg_duration = np.mean(x_values) if len(x_values) > 0 else np.nan
        median_duration = np.median(x_values) if len(x_values) > 0 else np.nan
        avg_distance = np.mean(y_values) if len(y_values) > 0 else np.nan
        median_distance = np.median(y_values) if len(y_values) > 0 else np.nan
        reach_statistics.append({
            "reach_index": reach_index,
            "avg_duration": avg_duration,
            "median_duration": median_duration,
            "avg_distance": avg_distance,
            "median_distance": median_distance
        })

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

        # Annotate statistics on the plot
        ax.text(0.05, 0.95, f"Avg Dur: {avg_duration:.2f}\nMed Dur: {median_duration:.2f}\n"
                            f"Avg Dist: {avg_distance:.2f}\nMed Dist: {median_distance:.2f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Overlay the points
        if len(x_values) > 0 and len(y_values) > 0:
            ax.scatter(avg_duration, avg_distance, color='black', s=50, label="Average Point", zorder=5)
            ax.scatter(median_duration, median_distance, color='blue', s=50, label="Median Point", zorder=5)

    # Hide unused subplots
    for i in range(16, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return reach_statistics

# Example usage
reach_statistics = scatter_plot_all_reach_indices(updated_metrics, '07/22/HW', 'right', 'durations', 'distance')


def calculate_reach_statistics_all_subjects(updated_metrics, metric_x, metric_y):
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

# Example usage
mean_stats, median_stats = calculate_reach_statistics_all_subjects(updated_metrics, 'durations', 'distance')

def plot_reach_statistics_overlay(stats, subject, hand, metric_x, metric_y, stat_type="avg"):
    """
    Overlays scatter plots for all reach indices in a single plot using either mean or median statistics.
    Groups reach indices by 0, 4, 8, 12; 1, 5, 9, 13; 2, 6, 10, 14; 3, 7, 11, 15, and uses the same color for each group.
    Calculates and returns the Spearman correlation for the overlayed points.

    Parameters:
        stats (dict): Statistics (mean or median) for all subjects and hands.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        stat_type (str): Type of statistics to use ("mean" or "median").

    Returns:
        tuple: Spearman correlation and p-value for the overlayed points.
    """
    # Initialize the plot
    plt.figure(figsize=(10, 8))

    x_values = []
    y_values = []

    # Define color groups for reach indices
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
            label = f"Group {reach_index % 4}, {group_indexes[reach_index % 4]}" if reach_index < 4 else None
            plt.scatter(duration, distance, color=color, s=50, label=label, zorder=5)

            # Annotate the reach index
            plt.text(duration, distance, f"{reach_index}", fontsize=9, color=color, ha="right" if stat_type == "mean" else "left")

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

# Example usage
spearman_corr_avg, p_value_avg = plot_reach_statistics_overlay(mean_stats, '07/22/HW', 'right', 'durations', 'distance', stat_type="avg")
print(f"Spearman Correlation (Mean): {spearman_corr_avg}, P-value: {p_value_avg}")

def plot_reach_statistics_overlay_all_subjects(stats, metric_x, metric_y, stat_type="avg"):
    """
    Overlays scatter plots for all reach indices in a single plot using either mean or median statistics
    for all subjects and hands. Groups reach indices by 0, 4, 8, 12; 1, 5, 9, 13; 2, 6, 10, 14; 3, 7, 11, 15,
    and uses the same color for each group. Calculates and returns the Spearman correlation for the overlayed points.

    Parameters:
        stats (dict): Statistics (mean or median) for all subjects and hands.
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        stat_type (str): Type of statistics to use ("mean" or "median").

    Returns:
        dict: Spearman correlation and p-value for each subject and hand.
    """
    results = {}

    for subject, hands_data in stats.items():
        results[subject] = {}
        for hand, reach_stats in hands_data.items():
            # Initialize the plot
            plt.figure(figsize=(10, 8))

            x_values = []
            y_values = []

            # Define color groups for reach indices
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
                duration = reach_stats[reach_index][f"{stat_type}_duration"]
                distance = reach_stats[reach_index][f"{stat_type}_distance"]

                # Overlay the points
                if not np.isnan(duration) and not np.isnan(distance):
                    x_values.append(duration)
                    y_values.append(distance)
                    color = color_groups[reach_index]
                    label = f"Group {reach_index % 4}, {group_indexes[reach_index % 4]}" if reach_index < 4 else None
                    plt.scatter(duration, distance, color=color, s=50, label=label, zorder=5)

                    # Annotate the reach index
                    plt.text(duration, distance, f"{reach_index}", fontsize=9, color=color, ha="right" if stat_type == "mean" else "left")

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

            # Store results
            results[subject][hand] = {"spearman_corr": spearman_corr, "p_value": p_value}

    return results

# Example usage
results_all = plot_reach_statistics_overlay_all_subjects(mean_stats, 'durations', 'distance', stat_type="avg")

# Plot histogram of median Spearman correlations for left and right hands using results_all
def plot_histogram_of_median_correlations_all_hands(results_all):
    """
    Plots histograms of median Spearman correlations for left and right hands across all subjects.

    Parameters:
        results_all (dict): Results containing Spearman correlations for each subject and hand.
    """
    correlations_left = []
    correlations_right = []

    for subject, hands_data in results_all.items():
        if 'left' in hands_data and hands_data['left']['spearman_corr'] is not None:
            correlations_left.append(hands_data['left']['spearman_corr'])
        if 'right' in hands_data and hands_data['right']['spearman_corr'] is not None:
            correlations_right.append(hands_data['right']['spearman_corr'])

    # Calculate medians
    median_left = np.median(correlations_left) if correlations_left else np.nan
    median_right = np.median(correlations_right) if correlations_right else np.nan

    # Plot histogram
    plt.figure(figsize=(6, 6))
    plt.hist(correlations_left, bins=18, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
    plt.hist(correlations_right, bins=18, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')

    # Add median lines
    plt.axvline(median_left, color='orange', linestyle='--', label=f'Median Left: {median_left:.2f}')
    plt.axvline(median_right, color='blue', linestyle='--', label=f'Median Right: {median_right:.2f}')

    # Add labels, title, and legend
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Frequency')
    plt.title('Histogram of Median Spearman Correlations for Left and Right Hands')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot the histogram
plot_histogram_of_median_correlations_all_hands(results_all)




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

# Scatter plot for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation
def scatter_plot_all_reach_indices(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
    """
    Plots scatter plots for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        subject_statistics (dict): Statistics containing max durations and max distances for the subject.
    """
    # Create subplots in a 4x4 layout
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes = axes.flatten()

    # Get max duration and max distance for the subject
    max_duration = subject_statistics[subject]['max_durations']
    max_distance = subject_statistics[subject]['max_distance']

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

        # Perform hyperbolic regression if applicable
        if metric_x == 'durations' and metric_y == 'distance':
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
                                fontsize=12, color="black", ha="right", va="bottom")
            except Exception as e:
                print(f"Hyperbolic regression failed for Reach Index {reach_index}: {e}")
        else:
            print(f"Error: Hyperbolic regression is only applicable for 'durations' vs 'distance'.")

        # Add Spearman correlation to the plot
        ax.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Add diagonal line from (0, 0) to max duration and max distance
        if not np.isnan(max_duration) and not np.isnan(max_distance):
            ax.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")

    # Hide unused subplots
    for i in range(16, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
scatter_plot_all_reach_indices(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', subject_statistics)

# Overlay 16 hyperbolic regressions, intersections, and diagonal line in one figure
def overlay_hyperbolic_regressions(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
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
    plt.figure(figsize=(10, 8))

    # Generate a color palette from light blue to black
    color_palette = sns.color_palette("light:black", 16)

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
                    plt.scatter(xi, yi, color=color_palette[reach_index], s=50, zorder=5)
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
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

# Example usage
results = overlay_hyperbolic_regressions(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', subject_statistics)




# -------------------------------------------------------------------------------------------------------------------


# # Scatter plot for all durations and all distances with hyperbolic fit per subject per hand (or overlay hands)
# ## subject plot
# def scatter_plot_all_duration_distance(updated_metrics, overlay_hands=True, selected_subjects=None):
#     """
#     Plots a scatter plot of all durations vs all distances across selected subjects, hands, and trials,
#     and fits a hyperbolic curve. Optionally overlays both hands or separates them.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.
#         overlay_hands (bool): If True, overlays both hands in the same plot. If False, separates them.
#         selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
#     """
#     def extract_data(metrics, hand):
#         durations = []
#         distances = []
#         for trial, trial_durations in metrics[hand]['durations'].items():
#             durations.extend([duration for duration in trial_durations if not pd.isna(duration)])
#         for trial, trial_distances in metrics[hand]['distance'].items():
#             distances.extend([distance for distance in trial_distances if not pd.isna(distance)])
#         return durations, distances

#     if selected_subjects is None:
#         selected_subjects = updated_metrics.keys()

#     if overlay_hands:
#         all_durations, all_distances = [], []
#         for subject in selected_subjects:
#             if subject in updated_metrics:
#                 for hand in ['left', 'right']:
#                     durations, distances = extract_data(updated_metrics[subject], hand)
#                     all_durations.extend(durations)
#                     all_distances.extend(distances)

#         # Plot overlayed hands
#         _plot_scatter_with_fit(all_durations, all_distances, title=f"Overlayed Hands (Subjects: {', '.join(selected_subjects)})")
#     else:
#         for hand in ['left', 'right']:
#             all_durations, all_distances = [], []
#             for subject in selected_subjects:
#                 if subject in updated_metrics:
#                     durations, distances = extract_data(updated_metrics[subject], hand)
#                     all_durations.extend(durations)
#                     all_distances.extend(distances)

#             # Plot separate hands
#             _plot_scatter_with_fit(all_durations, all_distances, title=f"{hand.capitalize()} Hand (Subjects: {', '.join(selected_subjects)})")

# def _plot_scatter_with_fit(durations, distances, title):
#     """
#     Helper function to plot scatter and fit a hyperbolic curve.

#     Parameters:
#         durations (list): List of durations.
#         distances (list): List of distances.
#         title (str): Title of the plot.
#     """
#     # Calculate Spearman correlation
#     if len(durations) > 1 and len(distances) > 1:
#         spearman_corr, p_value = spearmanr(durations, distances)
#     else:
#         spearman_corr, p_value = np.nan, np.nan

#     # Scatter plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(durations, distances, alpha=0.7, color='blue')
#     plt.title(f"Scatter Plot of Durations vs Distances ({title})")

#     labels = {
#         'distance': "Good → Bad (cm)",
#         'durations': "Good / Fast → Bad / Slow (s)"
#     }
#     plt.xlabel(f"Durations ({labels.get('durations', '')})")
#     plt.ylabel(f"Distance ({labels.get('distance', '')})")
#     plt.grid(alpha=0.5)

#     # Add Spearman correlation and number of data points to the plot
#     plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.4f}\nP-value: {p_value:.4f}\nData Points: {len(durations)}",
#              transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

#     # Fit a hyperbolic curve
#     def hyperbolic_func(x, a, b):
#         return a / (x + b)

#     try:
#         params, _ = curve_fit(hyperbolic_func, durations, distances)
#         x_fit = np.linspace(min(durations), max(durations), 500)
#         y_fit = hyperbolic_func(x_fit, *params)
#         plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
#     except Exception as e:
#         print(f"Hyperbolic fit failed: {e}")

#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=All_dates)
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'])
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['07/22/HW'])

# ## summary plot for all subjects

# # Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject and hand
# def calculate_duration_distance_metrics(updated_metrics, selected_subjects=None):
#     """
#     Calculates Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b)
#     for durations vs distances for each subject and hand.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.
#         selected_subjects (list or None): List of subjects to include. If None, includes all subjects.

#     Returns:
#         dict: Dictionary containing results for each subject and hand.
#     """
#     def extract_data(metrics, hand):
#         durations = []
#         distances = []
#         for trial, trial_durations in metrics[hand]['durations'].items():
#             durations.extend([duration for duration in trial_durations if not pd.isna(duration)])
#         for trial, trial_distances in metrics[hand]['distance'].items():
#             distances.extend([distance for distance in trial_distances if not pd.isna(distance)])
#         return durations, distances

#     if selected_subjects is None:
#         selected_subjects = updated_metrics.keys()

#     results = {}

#     for subject in selected_subjects:
#         if subject in updated_metrics:
#             results[subject] = {}
#             for hand in ['left', 'right']:
#                 durations, distances = extract_data(updated_metrics[subject], hand)

#                 # Calculate Spearman correlation
#                 if len(durations) > 1 and len(distances) > 1:
#                     spearman_corr, p_value = spearmanr(durations, distances)
#                 else:
#                     spearman_corr, p_value = np.nan, np.nan

#                 # Fit a hyperbolic curve
#                 def hyperbolic_func(x, a, b):
#                     return a / (x + b)

#                 try:
#                     params, _ = curve_fit(hyperbolic_func, durations, distances)
#                     a, b = params
#                 except Exception:
#                     a, b = np.nan, np.nan

#                 # Store results
#                 results[subject][hand] = {
#                     "spearman_corr": spearman_corr,
#                     "p_value": p_value,
#                     "data_points": len(durations),
#                     "hyperbolic_fit_a": a,
#                     "hyperbolic_fit_b": b
#                 }

#     return results

# results_duration_distance = calculate_duration_distance_metrics(updated_metrics)

# # Plot column chart of Spearman correlations and p-values for left and right hands
# def plot_column_spearman_corr_pvalues(results_all):
#     """
#     Plots column charts of Spearman correlations and p-values for left and right hands across all subjects.

#     Parameters:
#         results_all (dict): Results containing Spearman correlations and p-values for each subject and hand.
#     """
#     subjects = list(results_all.keys())
#     correlations_left = [results_all[subject]['left']['spearman_corr'] if 'left' in results_all[subject] else np.nan for subject in subjects]
#     correlations_right = [results_all[subject]['right']['spearman_corr'] if 'right' in results_all[subject] else np.nan for subject in subjects]
#     pvalues_left = [results_all[subject]['left']['p_value'] if 'left' in results_all[subject] else np.nan for subject in subjects]
#     pvalues_right = [results_all[subject]['right']['p_value'] if 'right' in results_all[subject] else np.nan for subject in subjects]

#     x = np.arange(len(subjects))  # the label locations
#     width = 0.35  # the width of the bars

#     # Create figure and subplots
#     fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     # Plot Spearman correlations
#     bars_left_corr = axes[0].bar(x - width / 2, correlations_left, width, label='Left Hand', color='orange', alpha=0.7)
#     bars_right_corr = axes[0].bar(x + width / 2, correlations_right, width, label='Right Hand', color='blue', alpha=0.7)
#     axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
#     axes[0].set_ylabel('Spearman Correlation')
#     axes[0].set_title('Spearman Correlations by Subject and Hand')
#     axes[0].legend()

#     # Annotate Spearman correlation values
#     for bar in bars_left_corr:
#         height = bar.get_height()
#         if not np.isnan(height):
#             axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
#     for bar in bars_right_corr:
#         height = bar.get_height()
#         if not np.isnan(height):
#             axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

#     # Plot P-values
#     bars_left_pval = axes[1].bar(x - width / 2, pvalues_left, width, label='Left Hand', color='orange', alpha=0.7)
#     bars_right_pval = axes[1].bar(x + width / 2, pvalues_right, width, label='Right Hand', color='blue', alpha=0.7)
#     axes[1].axhline(0.05, color='red', linewidth=0.8, linestyle='--', label='Significance Threshold (0.05)')
#     axes[1].set_ylabel('P-value')
#     axes[1].set_title('P-values by Subject and Hand')
#     axes[1].legend()

#     # Annotate P-value values only when > 0.05
#     for bar in bars_left_pval:
#         height = bar.get_height()
#         if not np.isnan(height) and height > 0.05:
#             axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
#     for bar in bars_right_pval:
#         height = bar.get_height()
#         if not np.isnan(height) and height > 0.05:
#             axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

#     # Set x-axis labels for both subplots
#     axes[1].set_xticks(x)
#     axes[1].set_xticklabels(subjects, rotation=45, ha='right')
#     axes[1].set_xlabel('Subjects')

#     plt.tight_layout()
#     plt.show()

# plot_column_spearman_corr_pvalues(results_duration_distance)

# # Plot histograms for Spearman correlations and p-values for durations vs distances, overlaying hands
# def plot_histogram_spearman_corr_pvalues_by_hand(results_duration_distance):
#     """
#     Plots histograms of Spearman correlations and p-values for durations vs distances across all subjects,
#     overlaying left and right hands in different colors.

#     Parameters:
#         results_duration_distance (dict): Results containing Spearman correlations and p-values for each subject and hand.
#     """
#     correlations_left = []
#     correlations_right = []
#     p_values_left = []
#     p_values_right = []

#     for subject, hands_data in results_duration_distance.items():
#         for hand, metrics in hands_data.items():
#             if 'spearman_corr' in metrics and not np.isnan(metrics['spearman_corr']):
#                 if hand == 'left':
#                     correlations_left.append(metrics['spearman_corr'])
#                 elif hand == 'right':
#                     correlations_right.append(metrics['spearman_corr'])
#             if 'p_value' in metrics and not np.isnan(metrics['p_value']):
#                 if hand == 'left':
#                     p_values_left.append(metrics['p_value'])
#                 elif hand == 'right':
#                     p_values_right.append(metrics['p_value'])

#     # Plot histogram for Spearman correlations
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
#     plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
#     plt.axvline(np.median(correlations_left), color='orange', linestyle='--', label=f"Median Left: {np.median(correlations_left):.2f}")
#     plt.axvline(np.median(correlations_right), color='blue', linestyle='--', label=f"Median Right: {np.median(correlations_right):.2f}")
#     plt.title("Histogram of Spearman Correlations by Hand")
#     plt.xlabel("Spearman Correlation")
#     plt.ylabel("Frequency")
#     plt.legend()

#     # Plot histogram for p-values
#     plt.subplot(1, 2, 2)
#     plt.hist(p_values_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
#     plt.hist(p_values_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
#     plt.axvline(np.median(p_values_left), color='orange', linestyle='--', label=f"Median Left: {np.median(p_values_left):.2f}")
#     plt.axvline(np.median(p_values_right), color='blue', linestyle='--', label=f"Median Right: {np.median(p_values_right):.2f}")
#     plt.axvline(0.05, color='red', linestyle='--', label="Significance Threshold (0.05)")
#     plt.title("Histogram of P-values by Hand")
#     plt.xlabel("P-value")
#     plt.ylabel("Frequency")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# plot_histogram_spearman_corr_pvalues_by_hand(results_duration_distance)

# # Perform Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately
# def wilcoxon_test_spearman_corr_separate(results_duration_distance):
#     """
#     Performs the Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately.

#     Parameters:
#         results_duration_distance (dict): Results containing Spearman correlations for each subject and hand.

#     Returns:
#         dict: Test statistics and p-values for left and right hands.
#     """
#     correlations_left = []
#     correlations_right = []

#     for subject, hands_data in results_duration_distance.items():
#         if 'left' in hands_data and 'spearman_corr' in hands_data['left']:
#             correlations_left.append(hands_data['left']['spearman_corr'])
#         if 'right' in hands_data and 'spearman_corr' in hands_data['right']:
#             correlations_right.append(hands_data['right']['spearman_corr'])

#     # Perform Wilcoxon signed-rank test for left hand
#     stat_left, p_value_left = wilcoxon(correlations_left)

#     # Perform Wilcoxon signed-rank test for right hand
#     stat_right, p_value_right = wilcoxon(correlations_right)

#     return {
#         "left": {"statistic": stat_left, "p_value": p_value_left},
#         "right": {"statistic": stat_right, "p_value": p_value_right}
#     }

# wilcoxon_test_spearman_corr_separate(results_duration_distance)

# # Plot histograms for Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
# def plot_histogram_spearman_corr_with_stats(results_duration_distance):
#     """
#     Plots histograms of Spearman correlations for durations vs distances across all subjects,
#     overlaying left and right hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

#     Parameters:
#         results_duration_distance (dict): Results containing Spearman correlations for each subject and hand.
#     """
#     correlations_left = []
#     correlations_right = []

#     for subject, hands_data in results_duration_distance.items():
#         for hand, metrics in hands_data.items():
#             if 'spearman_corr' in metrics and not np.isnan(metrics['spearman_corr']):
#                 if hand == 'left':
#                     correlations_left.append(metrics['spearman_corr'])
#                 elif hand == 'right':
#                     correlations_right.append(metrics['spearman_corr'])

#     # Calculate statistics
#     median_left = np.median(correlations_left)
#     iqr_left = np.percentile(correlations_left, 75) - np.percentile(correlations_left, 25)
#     q1_left = np.percentile(correlations_left, 25)
#     q3_left = np.percentile(correlations_left, 75)

#     median_right = np.median(correlations_right)
#     iqr_right = np.percentile(correlations_right, 75) - np.percentile(correlations_right, 25)
#     q1_right = np.percentile(correlations_right, 25)
#     q3_right = np.percentile(correlations_right, 75)

#     # Perform Wilcoxon signed-rank test for each hand separately
#     stat_left, p_value_left = wilcoxon(correlations_left)
#     stat_right, p_value_right = wilcoxon(correlations_right)

#     # Plot histogram for Spearman correlations
#     plt.figure(figsize=(8, 6))
#     plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
#     plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
#     plt.axvline(median_left, color='orange', linestyle='--', label=f"Median Left: {median_left:.2f}")
#     plt.axvline(median_right, color='blue', linestyle='--', label=f"Median Right: {median_right:.2f}")
#     plt.axvspan(q1_left, q3_left, color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
#     plt.axvspan(q1_right, q3_right, color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")
#     plt.title("Histogram of Spearman Correlations by Hand")
#     plt.xlabel("Spearman Correlation")
#     plt.ylabel("Frequency")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # Print statistics
#     print(f"Left Hand: Median = {median_left:.2f}, IQR = {iqr_left:.2f}")
#     print(f"Right Hand: Median = {median_right:.2f}, IQR = {iqr_right:.2f}")
#     print(f"Wilcoxon Signed-Rank Test (Left Hand): Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}")
#     print(f"Wilcoxon Signed-Rank Test (Right Hand): Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}")
# plot_histogram_spearman_corr_with_stats(results_duration_distance)

# # Compare left vs right hands at the subject level using Wilcoxon signed-rank test on paired subject Spearman correlations for durations and distances
# def compare_left_vs_right_hands(results_duration_distance):
#     """
#     Compares the left and right hands at the subject level using the Wilcoxon signed-rank test
#     on paired subject Spearman correlations for durations and distances.

#     Parameters:
#         results_duration_distance (dict): Dictionary containing Spearman correlation results for each subject.

#     Returns:
#         dict: Results of the Wilcoxon signed-rank test for durations and distances.
#     """
#     left_spearman_corrs = []
#     right_spearman_corrs = []


#     for subject, hands_data in results_duration_distance.items():
#         if 'left' in hands_data and 'right' in hands_data:
#             left_corr = hands_data['left']['spearman_corr']
#             right_corr = hands_data['right']['spearman_corr']
#             if not (pd.isna(left_corr) or pd.isna(right_corr)):
#                 left_spearman_corrs.append(left_corr)
#                 right_spearman_corrs.append(right_corr)

#     left_data_points = len(left_spearman_corrs)
#     right_data_points = len(right_spearman_corrs)

#     # Perform Wilcoxon signed-rank test for Spearman correlations
#     stat_corrs, p_value_corrs = wilcoxon(left_spearman_corrs, right_spearman_corrs)

#     results = {
#         "spearman_correlations": {"statistic": stat_corrs, "p_value": p_value_corrs},
#         "data_points": {"left": left_data_points, "right": right_data_points}
#     }

#     # Print the results of the Wilcoxon signed-rank test
#     print(f"Wilcoxon Signed-Rank Test Results:")
#     print(f"Spearman Correlations: Statistic = {stat_corrs:.2f}, P-value = {p_value_corrs:.4f}")
#     print(f"Data Points: Left Hand = {left_data_points}, Right Hand = {right_data_points}")

#     return results

# # Call the function to compare left vs right hands
# compare_left_vs_right_hands(results_duration_distance)

# -------------------------------------------------------------------------------------------------------------------

# ## subject calculations for reach indices
# # Histogram for all reach indices as subplots in a 4x4 layout with average, median statistics, and normal distribution evaluation
# def histogram_all_reach_indices_with_normality(updated_metrics, subject, hand, metric):
#     """
#     Plots histograms for all reach indices as subplots in a 4x4 layout with average, median statistics,
#     and overlays average and median on the plot.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.
#         subject (str): Subject identifier.
#         hand (str): Hand ('left' or 'right').
#         metric (str): Metric to plot ('durations' or 'distance').
#     """

#     # Create subplots in a 4x4 layout
#     rows, cols = 4, 4
#     fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
#     axes = axes.flatten()

#     for reach_index in range(16):
#         values = []

#         trials = updated_metrics[subject][hand][metric].keys()

#         for trial in trials:
#             trial_values = np.array(updated_metrics[subject][hand][metric][trial])

#             # Collect data for the specified reach index
#             if reach_index < len(trial_values):
#                 values.append(trial_values[reach_index])

#         # Remove NaN values
#         values = np.array(values)
#         values = values[~np.isnan(values)]

#         # Calculate statistics
#         avg_value = np.mean(values) if len(values) > 0 else np.nan
#         median_value = np.median(values) if len(values) > 0 else np.nan

#         # Plot histogram for the current reach index
#         ax = axes[reach_index]
#         ax.hist(values, bins=10, color='skyblue', edgecolor='black', alpha=0.7, density=True)
#         ax.set_title(f"Reach Index {reach_index}")
#         ax.set_xlabel(f"{metric.capitalize()}")
#         ax.set_ylabel("Density")

#         # Overlay average and median on the plot
#         if not np.isnan(avg_value):
#             ax.axvline(avg_value, color='red', linestyle='--', label=f"Avg: {avg_value:.2f}")
#         if not np.isnan(median_value):
#             ax.axvline(median_value, color='green', linestyle='-', label=f"Med: {median_value:.2f}")

#         # Annotate statistics on the plot
#         ax.text(0.95, 0.95, f"Avg: {avg_value:.2f}\nMed: {median_value:.2f}",
#                 transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
#                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

#         # Add legend
#         ax.legend()

#     # Hide unused subplots
#     for i in range(16, len(axes)):
#         axes[i].axis('off')

#     # Add a big title for the figure
#     fig.suptitle(f"Histograms for {subject} - {hand.capitalize()} Hand - {metric.capitalize()}", fontsize=16, y=1.02)

#     plt.tight_layout()
#     plt.show()

# histogram_all_reach_indices_with_normality(updated_metrics, '07/22/HW', 'right', 'distance')
# histogram_all_reach_indices_with_normality(updated_metrics, '06/19/CZ', 'right', 'distance')

# ## Group calculations for reach indices
# def calculate_reach_statistics_all_subjects(updated_metrics, metric_x, metric_y):
#     """
#     Calculates average and median 'durations', and average and median 'distance' for each reach_index
#     for all subjects and hands, and returns two dictionaries: one for mean and one for median.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.
#         metric_x (str): Metric for x-axis.
#         metric_y (str): Metric for y-axis.

#     Returns:
#         tuple: (mean_statistics, median_statistics) dictionaries containing statistics for all subjects and hands.
#     """
#     mean_statistics = {}
#     median_statistics = {}

#     for subject, hands_data in updated_metrics.items():
#         mean_statistics[subject] = {}
#         median_statistics[subject] = {}
#         for hand, metrics in hands_data.items():
#             mean_statistics[subject][hand] = []
#             median_statistics[subject][hand] = []

#             for reach_index in range(16):
#                 x_values = []
#                 y_values = []

#                 trials = metrics[metric_x].keys()

#                 for trial in trials:
#                     trial_x = np.array(metrics[metric_x][trial])
#                     trial_y = np.array(metrics[metric_y][trial])

#                     # Collect data for the specified reach index
#                     if reach_index < len(trial_x) and reach_index < len(trial_y):
#                         x_values.append(trial_x[reach_index])
#                         y_values.append(trial_y[reach_index])
                
#                 # Remove NaN values
#                 x_values = np.array(x_values)
#                 y_values = np.array(y_values)
#                 valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
#                 x_values = x_values[valid_indices]
#                 y_values = y_values[valid_indices]

#                 # Calculate statistics
#                 avg_duration = np.mean(x_values) if len(x_values) > 0 else np.nan
#                 median_duration = np.median(x_values) if len(x_values) > 0 else np.nan
#                 avg_distance = np.mean(y_values) if len(y_values) > 0 else np.nan
#                 median_distance = np.median(y_values) if len(y_values) > 0 else np.nan

#                 mean_statistics[subject][hand].append({
#                     "reach_index": reach_index,
#                     "avg_duration": avg_duration,
#                     "avg_distance": avg_distance
#                 })

#                 median_statistics[subject][hand].append({
#                     "reach_index": reach_index,
#                     "median_duration": median_duration,
#                     "median_distance": median_distance
#                 })

#     return mean_statistics, median_statistics

# mean_stats, median_stats = calculate_reach_statistics_all_subjects(updated_metrics, 'durations', 'distance')

# ## subject calculations for reach indices
# # Overlay scatter plots for all reach indices in a single plot using either mean or median statistics
# def plot_reach_statistics_overlay(stats, subject, hand, metric_x, metric_y, stat_type="avg"):
#     """
#     Overlays scatter plots for all reach indices in a single plot using either mean or median statistics.
#     Groups reach indices by 0, 4, 8, 12; 1, 5, 9, 13; 2, 6, 10, 14; 3, 7, 11, 15, and uses the same color for each group.
#     Calculates and returns the Spearman correlation for the overlayed points.

#     Parameters:
#         stats (dict): Statistics (mean or median) for all subjects and hands.
#         subject (str): Subject identifier.
#         hand (str): Hand ('left' or 'right').
#         metric_x (str): Metric for x-axis.
#         metric_y (str): Metric for y-axis.
#         stat_type (str): Type of statistics to use ("mean" or "median").

#     Returns:
#         tuple: Spearman correlation and p-value for the overlayed points.
#     """
#     # Initialize the plot
#     plt.figure(figsize=(6, 6))

#     x_values = []
#     y_values = []

#     # Define color groups for reach indices
#     color_groups = {
#         0: 'red', 4: 'red', 8: 'red', 12: 'red',
#         1: 'blue', 5: 'blue', 9: 'blue', 13: 'blue',
#         2: 'green', 6: 'green', 10: 'green', 14: 'green',
#         3: 'purple', 7: 'purple', 11: 'purple', 15: 'purple'
#     }

#     group_indexes = {
#         0: [0, 4, 8, 12],
#         1: [1, 5, 9, 13],
#         2: [2, 6, 10, 14],
#         3: [3, 7, 11, 15]
#     }

#     for reach_index in range(16):
#         # Get statistics for the current reach index
#         duration = stats[subject][hand][reach_index][f"{stat_type}_duration"]
#         distance = stats[subject][hand][reach_index][f"{stat_type}_distance"]

#         # Overlay the points
#         if not np.isnan(duration) and not np.isnan(distance):
#             x_values.append(duration)
#             y_values.append(distance)
#             color = color_groups[reach_index]
#             label = f"Group {reach_index % 4}, {group_indexes[reach_index % 4]}" if reach_index < 4 else None
#             plt.scatter(duration, distance, color=color, s=50, label=label, zorder=5)

#             # Annotate the reach index
#             plt.text(duration, distance, f"{reach_index}", fontsize=12, color=color, ha="right" if stat_type == "mean" else "left")

#     # Calculate Spearman correlation
#     if len(x_values) > 1 and len(y_values) > 1:
#         spearman_corr, p_value = spearmanr(x_values, y_values)
#     else:
#         spearman_corr, p_value = np.nan, np.nan

#     # Add labels and legend
#     labels = {
#         'distance': "Good → Bad (cm)",
#         'accuracy': "Bad → Good (%)",
#         'durations': "Good / Fast → Bad / Slow (s)",
#         'speed': "Bad / Slow → Good / Fast (1/s)"
#     }
#     plt.xlabel(f"{metric_x.capitalize()} ({labels.get(metric_x, '')})")
#     plt.ylabel(f"{metric_y.capitalize()} ({labels.get(metric_y, '')})")
#     plt.title(f"Overlay of Reach Statistics ({subject}, {hand.capitalize()}, {stat_type.capitalize()})\nSpearman Corr: {spearman_corr:.2f}, P-value: {p_value:.2f}")
#     plt.legend()
#     plt.grid(alpha=0.5)
#     plt.tight_layout()
#     plt.show()

#     return spearman_corr, p_value

# spearman_corr_avg, p_value_avg = plot_reach_statistics_overlay(median_stats, '06/24/PZ', 'right', 'durations', 'distance', stat_type="median")
# spearman_corr_avg, p_value_avg = plot_reach_statistics_overlay(median_stats, '06/24/PZ', 'left', 'durations', 'distance', stat_type="median")

# ## summary plot for all subjects
# def plot_reach_statistics_overlay_all_subjects(stats, metric_x, metric_y, stat_type="avg"):
#     """
#     Overlays scatter plots for all reach indices in subplots, one for each subject, using either mean or median statistics.
#     Groups reach indices by 0, 4, 8, 12; 1, 5, 9, 13; 2, 6, 10, 14; 3, 7, 11, 15,
#     and uses the same color for each group. Creates separate figures for left and right hands.

#     Parameters:
#         stats (dict): Statistics (mean or median) for all subjects and hands.
#         metric_x (str): Metric for x-axis.
#         metric_y (str): Metric for y-axis.
#         stat_type (str): Type of statistics to use ("mean" or "median").

#     Returns:
#         dict: Spearman correlation, p-value, and hyperbolic fit parameters for each subject and hand.
#     """
#     results = {"left": {}, "right": {}}

#     for hand in ["left", "right"]:
#         num_subjects = len(stats)
#         rows, cols = (num_subjects // 4) + (num_subjects % 4 > 0), 4
#         fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
#         axes = axes.flatten()

#         for idx, (subject, hands_data) in enumerate(stats.items()):
#             ax = axes[idx]
#             results[hand][subject] = {}
#             if hand in hands_data:
#                 reach_stats = hands_data[hand]
#                 x_values = []
#                 y_values = []

#                 # Define color groups for reach indices
#                 color_groups = {
#                     0: 'red', 4: 'red', 8: 'red', 12: 'red',
#                     1: 'blue', 5: 'blue', 9: 'blue', 13: 'blue',
#                     2: 'green', 6: 'green', 10: 'green', 14: 'green',
#                     3: 'purple', 7: 'purple', 11: 'purple', 15: 'purple'
#                 }

#                 group_indexes = {
#                     0: [0, 4, 8, 12],
#                     1: [1, 5, 9, 13],
#                     2: [2, 6, 10, 14],
#                     3: [3, 7, 11, 15]
#                 }

#                 for reach_index in range(16):
#                     # Get statistics for the current reach index
#                     duration = reach_stats[reach_index][f"{stat_type}_duration"]
#                     distance = reach_stats[reach_index][f"{stat_type}_distance"]

#                     # Overlay the points
#                     if not np.isnan(duration) and not np.isnan(distance):
#                         x_values.append(duration)
#                         y_values.append(distance)
#                         color = color_groups[reach_index]
#                         ax.scatter(duration, distance, color=color, s=50, zorder=5)

#                         # Annotate the reach index
#                         ax.text(duration, distance, f"{reach_index}", fontsize=9, color=color, ha="right" if stat_type == "mean" else "left")

#                 # Calculate Spearman correlation
#                 if len(x_values) > 1 and len(y_values) > 1:
#                     spearman_corr, p_value = spearmanr(x_values, y_values)
#                 else:
#                     spearman_corr, p_value = np.nan, np.nan

#                 # Perform hyperbolic regression
#                 def hyperbolic_func(x, a, b):
#                     return a / (x + b)

#                 try:
#                     params, _ = curve_fit(hyperbolic_func, x_values, y_values)
#                     a, b = params
#                     x_fit = np.linspace(min(x_values), max(x_values), 500)
#                     y_fit = hyperbolic_func(x_fit, *params)
#                     ax.plot(x_fit, y_fit, color='black', linestyle='--', label=f"Hyperbolic Fit: a={a:.2f}, b={b:.2f}")
#                 except Exception as e:
#                     a, b = np.nan, np.nan
#                     print(f"Hyperbolic fit failed for {subject} ({hand.capitalize()}): {e}")

#                 # Add labels and title
#                 labels = {
#                     'distance': "Good → Bad (cm)",
#                     'accuracy': "Bad → Good (%)",
#                     'durations': "Good / Fast → Bad / Slow (s)",
#                     'speed': "Bad / Slow → Good / Fast (1/s)"
#                 }
#                 ax.set_xlabel(f"{metric_x.capitalize()} ({labels.get(metric_x, '')})")
#                 ax.set_ylabel(f"{metric_y.capitalize()} ({labels.get(metric_y, '')})")
#                 ax.set_title(f"{subject} - {hand.capitalize()}\nSpearman Corr: {spearman_corr:.2f}, P-value: {p_value:.2f}")
#                 ax.grid(alpha=0.5)

#                 # Store results
#                 results[hand][subject] = {
#                     "spearman_corr": spearman_corr,
#                     "p_value": p_value,
#                     "data_points": len(x_values),
#                     "hyperbolic_fit_a": a,
#                     "hyperbolic_fit_b": b
#                 }

#         # Hide unused subplots
#         for i in range(len(stats), len(axes)):
#             axes[i].axis('off')

#         plt.tight_layout()
#         plt.suptitle(f"Reach Statistics Overlay ({hand.capitalize()} Hand)", fontsize=16, y=1.02)
#         plt.show()

#     # return results

# plot_reach_statistics_overlay_all_subjects(median_stats, 'durations', 'distance', stat_type="median")

# ## summary calculations for reach indices
# # Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject and hand
# def calculate_reach_statistics_metrics(stats, selected_subjects=None, stat_type="avg"):
#     """
#     Calculates Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b)
#     for reach statistics (e.g., durations vs distances) for each subject and hand.

#     Parameters:
#         stats (dict): Statistics (mean or median) for all subjects and hands.
#         selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
#         stat_type (str): Type of statistics to use ("mean" or "median").

#     Returns:
#         dict: Dictionary containing results for each subject and hand.
#     """
#     if selected_subjects is None:
#         selected_subjects = stats.keys()

#     results = {}

#     for subject in selected_subjects:
#         if subject in stats:
#             results[subject] = {}
#             for hand in ['left', 'right']:
#                 x_values = []
#                 y_values = []

#                 for reach_index in range(16):
#                     # Get statistics for the current reach index
#                     duration = stats[subject][hand][reach_index].get(f"{stat_type}_duration", np.nan)
#                     distance = stats[subject][hand][reach_index].get(f"{stat_type}_distance", np.nan)

#                     if not np.isnan(duration) and not np.isnan(distance):
#                         x_values.append(duration)
#                         y_values.append(distance)

#                 # Calculate Spearman correlation
#                 if len(x_values) > 1 and len(y_values) > 1:
#                     spearman_corr, p_value = spearmanr(x_values, y_values)
#                 else:
#                     spearman_corr, p_value = np.nan, np.nan

#                 # Fit a hyperbolic curve
#                 def hyperbolic_func(x, a, b):
#                     return a / (x + b)

#                 try:
#                     params, _ = curve_fit(hyperbolic_func, x_values, y_values)
#                     a, b = params
#                 except Exception:
#                     a, b = np.nan, np.nan

#                 # Store results
#                 results[subject][hand] = {
#                     "spearman_corr": spearman_corr,
#                     "p_value": p_value,
#                     "data_points": len(x_values),
#                     "hyperbolic_fit_a": a,
#                     "hyperbolic_fit_b": b
#                 }

#     return results

# results_reach_statistics = calculate_reach_statistics_metrics(median_stats, stat_type="median")
# # results_reach_statistics = calculate_reach_statistics_metrics(mean_stats, stat_type="avg")


# # Plot column chart of Spearman correlations and p-values for left and right hands
# def plot_column_spearman_corr_pvalues(results_all):
#     """
#     Plots column charts of Spearman correlations and p-values for left and right hands across all subjects.

#     Parameters:
#         results_all (dict): Results containing Spearman correlations and p-values for each subject and hand.
#     """
#     subjects = list(results_all.keys())
#     correlations_left = [results_all[subject]['left']['spearman_corr'] if 'left' in results_all[subject] else np.nan for subject in subjects]
#     correlations_right = [results_all[subject]['right']['spearman_corr'] if 'right' in results_all[subject] else np.nan for subject in subjects]
#     pvalues_left = [results_all[subject]['left']['p_value'] if 'left' in results_all[subject] else np.nan for subject in subjects]
#     pvalues_right = [results_all[subject]['right']['p_value'] if 'right' in results_all[subject] else np.nan for subject in subjects]

#     x = np.arange(len(subjects))  # the label locations
#     width = 0.35  # the width of the bars

#     # Create figure and subplots
#     fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     # Plot Spearman correlations
#     bars_left_corr = axes[0].bar(x - width / 2, correlations_left, width, label='Left Hand', color='orange', alpha=0.7)
#     bars_right_corr = axes[0].bar(x + width / 2, correlations_right, width, label='Right Hand', color='blue', alpha=0.7)
#     axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
#     axes[0].set_ylabel('Spearman Correlation')
#     axes[0].set_title('Spearman Correlations by Subject and Hand')
#     axes[0].legend()

#     # Annotate Spearman correlation values
#     for bar in bars_left_corr:
#         height = bar.get_height()
#         if not np.isnan(height):
#             axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
#     for bar in bars_right_corr:
#         height = bar.get_height()
#         if not np.isnan(height):
#             axes[0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

#     # Plot P-values
#     bars_left_pval = axes[1].bar(x - width / 2, pvalues_left, width, label='Left Hand', color='orange', alpha=0.7)
#     bars_right_pval = axes[1].bar(x + width / 2, pvalues_right, width, label='Right Hand', color='blue', alpha=0.7)
#     axes[1].axhline(0.05, color='red', linewidth=0.8, linestyle='--', label='Significance Threshold (0.05)')
#     axes[1].set_ylabel('P-value')
#     axes[1].set_title('P-values by Subject and Hand')
#     axes[1].legend()

#     # Annotate P-value values only when > 0.05
#     for bar in bars_left_pval:
#         height = bar.get_height()
#         if not np.isnan(height) and height > 0.05:
#             axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
#     for bar in bars_right_pval:
#         height = bar.get_height()
#         if not np.isnan(height) and height > 0.05:
#             axes[1].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

#     # Set x-axis labels for both subplots
#     axes[1].set_xticks(x)
#     axes[1].set_xticklabels(subjects, rotation=45, ha='right')
#     axes[1].set_xlabel('Subjects')

#     plt.tight_layout()
#     plt.show()

# plot_column_spearman_corr_pvalues(results_reach_statistics)

# # Plot histograms for Spearman correlations and p-values for durations vs distances, overlaying hands
# def plot_histogram_spearman_corr_pvalues_by_hand(results_duration_distance):
#     """
#     Plots histograms of Spearman correlations and p-values for durations vs distances across all subjects,
#     overlaying left and right hands in different colors.

#     Parameters:
#         results_duration_distance (dict): Results containing Spearman correlations and p-values for each subject and hand.
#     """
#     correlations_left = []
#     correlations_right = []
#     p_values_left = []
#     p_values_right = []

#     for subject, hands_data in results_duration_distance.items():
#         for hand, metrics in hands_data.items():
#             if 'spearman_corr' in metrics and not np.isnan(metrics['spearman_corr']):
#                 if hand == 'left':
#                     correlations_left.append(metrics['spearman_corr'])
#                 elif hand == 'right':
#                     correlations_right.append(metrics['spearman_corr'])
#             if 'p_value' in metrics and not np.isnan(metrics['p_value']):
#                 if hand == 'left':
#                     p_values_left.append(metrics['p_value'])
#                 elif hand == 'right':
#                     p_values_right.append(metrics['p_value'])

#     # Plot histogram for Spearman correlations
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
#     plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
#     plt.axvline(np.median(correlations_left), color='orange', linestyle='--', label=f"Median Left: {np.median(correlations_left):.2f}")
#     plt.axvline(np.median(correlations_right), color='blue', linestyle='--', label=f"Median Right: {np.median(correlations_right):.2f}")
#     plt.title("Histogram of Spearman Correlations by Hand")
#     plt.xlabel("Spearman Correlation")
#     plt.ylabel("Frequency")
#     plt.legend()

#     # Plot histogram for p-values
#     plt.subplot(1, 2, 2)
#     plt.hist(p_values_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
#     plt.hist(p_values_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
#     plt.axvline(np.median(p_values_left), color='orange', linestyle='--', label=f"Median Left: {np.median(p_values_left):.2f}")
#     plt.axvline(np.median(p_values_right), color='blue', linestyle='--', label=f"Median Right: {np.median(p_values_right):.2f}")
#     plt.axvline(0.05, color='red', linestyle='--', label="Significance Threshold (0.05)")
#     plt.title("Histogram of P-values by Hand")
#     plt.xlabel("P-value")
#     plt.ylabel("Frequency")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# plot_histogram_spearman_corr_pvalues_by_hand(results_reach_statistics)

# # Perform Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately
# def wilcoxon_test_spearman_corr_separate(results_duration_distance):
#     """
#     Performs the Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately.

#     Parameters:
#         results_duration_distance (dict): Results containing Spearman correlations for each subject and hand.

#     Returns:
#         dict: Test statistics and p-values for left and right hands.
#     """
#     correlations_left = []
#     correlations_right = []

#     for subject, hands_data in results_duration_distance.items():
#         if 'left' in hands_data and 'spearman_corr' in hands_data['left']:
#             correlations_left.append(hands_data['left']['spearman_corr'])
#         if 'right' in hands_data and 'spearman_corr' in hands_data['right']:
#             correlations_right.append(hands_data['right']['spearman_corr'])

#     # Perform Wilcoxon signed-rank test for left hand
#     stat_left, p_value_left = wilcoxon(correlations_left)

#     # Perform Wilcoxon signed-rank test for right hand
#     stat_right, p_value_right = wilcoxon(correlations_right)

#     return {
#         "left": {"statistic": stat_left, "p_value": p_value_left},
#         "right": {"statistic": stat_right, "p_value": p_value_right}
#     }

# # Call the function to perform the Wilcoxon test
# wilcoxon_test_spearman_corr_separate(results_reach_statistics)

# # Plot histograms for Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
# def plot_histogram_spearman_corr_with_stats(results_duration_distance):
#     """
#     Plots histograms of Spearman correlations for durations vs distances across all subjects,
#     overlaying left and right hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

#     Parameters:
#         results_duration_distance (dict): Results containing Spearman correlations for each subject and hand.
#     """
#     correlations_left = []
#     correlations_right = []

#     for subject, hands_data in results_duration_distance.items():
#         for hand, metrics in hands_data.items():
#             if 'spearman_corr' in metrics and not np.isnan(metrics['spearman_corr']):
#                 if hand == 'left':
#                     correlations_left.append(metrics['spearman_corr'])
#                 elif hand == 'right':
#                     correlations_right.append(metrics['spearman_corr'])

#     # Calculate statistics
#     median_left = np.median(correlations_left)
#     iqr_left = np.percentile(correlations_left, 75) - np.percentile(correlations_left, 25)
#     q1_left = np.percentile(correlations_left, 25)
#     q3_left = np.percentile(correlations_left, 75)

#     median_right = np.median(correlations_right)
#     iqr_right = np.percentile(correlations_right, 75) - np.percentile(correlations_right, 25)
#     q1_right = np.percentile(correlations_right, 25)
#     q3_right = np.percentile(correlations_right, 75)

#     # Perform Wilcoxon signed-rank test for each hand separately
#     stat_left, p_value_left = wilcoxon(correlations_left)
#     stat_right, p_value_right = wilcoxon(correlations_right)

#     # Plot histogram for Spearman correlations
#     plt.figure(figsize=(8, 6))
#     plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
#     plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
#     plt.axvline(median_left, color='orange', linestyle='--', label=f"Median Left: {median_left:.2f}")
#     plt.axvline(median_right, color='blue', linestyle='--', label=f"Median Right: {median_right:.2f}")
#     plt.axvspan(q1_left, q3_left, color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
#     plt.axvspan(q1_right, q3_right, color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")
#     plt.title("Histogram of Spearman Correlations by Hand")
#     plt.xlabel("Spearman Correlation")
#     plt.ylabel("Frequency")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # Print statistics
#     print(f"Left Hand: Median = {median_left:.2f}, IQR = {iqr_left:.2f}")
#     print(f"Right Hand: Median = {median_right:.2f}, IQR = {iqr_right:.2f}")
#     print(f"Wilcoxon Signed-Rank Test (Left Hand): Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}")
#     print(f"Wilcoxon Signed-Rank Test (Right Hand): Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}")

# plot_histogram_spearman_corr_with_stats(results_reach_statistics)

# # Compare left vs right hands at the subject level using Wilcoxon signed-rank test on Spearman correlation values
# compare_left_vs_right_hands(results_reach_statistics)



# -------------------------------------------------------------------------------------------------------------------

# ## subject plot
# def scatter_plot_all_duration_distance(updated_metrics, overlay_hands=True, selected_subjects=None, special_indices=None):
#     """
#     Plots a scatter plot of all durations vs all distances across selected subjects, hands, and trials,
#     and fits a hyperbolic curve. Optionally overlays both hands or separates them.
#     Allows highlighting specific reach indices.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.
#         overlay_hands (bool): If True, overlays both hands in the same plot. If False, separates them.
#         selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
#         special_indices (list or None): List of reach indices to include. If None, all indices are included.
#     """
#     def extract_data(metrics, hand, indices):
#         durations = [[] for _ in range(16)]
#         distances = [[] for _ in range(16)]
#         for trial, trial_durations in metrics[hand]['durations'].items():
#             for reach_index, duration in enumerate(trial_durations):
#                 if not pd.isna(duration) and (indices is None or reach_index in indices):
#                     durations[reach_index].append(duration)
#         for trial, trial_distances in metrics[hand]['distance'].items():
#             for reach_index, distance in enumerate(trial_distances):
#                 if not pd.isna(distance) and (indices is None or reach_index in indices):
#                     distances[reach_index].append(distance)
#         return durations, distances

#     if selected_subjects is None:
#         selected_subjects = updated_metrics.keys()

#     if overlay_hands:
#         all_durations = [[] for _ in range(16)]
#         all_distances = [[] for _ in range(16)]
#         for subject in selected_subjects:
#             if subject in updated_metrics:
#                 for hand in ['left', 'right']:
#                     durations, distances = extract_data(updated_metrics[subject], hand, special_indices)
#                     for i in range(16):
#                         all_durations[i].extend(durations[i])
#                         all_distances[i].extend(distances[i])

#         # Plot overlayed hands
#         _plot_scatter_with_fit(all_durations, all_distances, title=f"Overlayed Hands (Subjects: {', '.join(selected_subjects)})", special_indices=special_indices)
#     else:
#         for hand in ['left', 'right']:
#             all_durations = [[] for _ in range(16)]
#             all_distances = [[] for _ in range(16)]
#             for subject in selected_subjects:
#                 if subject in updated_metrics:
#                     durations, distances = extract_data(updated_metrics[subject], hand, special_indices)
#                     for i in range(16):
#                         all_durations[i].extend(durations[i])
#                         all_distances[i].extend(distances[i])

#             # Plot separate hands
#             _plot_scatter_with_fit(all_durations, all_distances, title=f"{hand.capitalize()} Hand (Subjects: {', '.join(selected_subjects)})", special_indices=special_indices)

# def _plot_scatter_with_fit(durations, distances, title, special_indices=None):
#     """
#     Helper function to plot scatter and fit a hyperbolic curve.

#     Parameters:
#         durations (list of lists): List of durations for each reach index.
#         distances (list of lists): List of distances for each reach index.
#         title (str): Title of the plot.
#         special_indices (list or None): List of reach indices to include. If None, all indices are included.
#     """
#     # Define color groups for reach indices
#     color_groups = {
#         0: 'red', 4: 'red', 8: 'red', 12: 'red',
#         1: 'blue', 5: 'blue', 9: 'blue', 13: 'blue',
#         2: 'green', 6: 'green', 10: 'green', 14: 'green',
#         3: 'purple', 7: 'purple', 11: 'purple', 15: 'purple'
#     }

#     # Flatten durations and distances for correlation calculation
#     flat_durations = [d for i, sublist in enumerate(durations) for d in sublist if special_indices is None or i in special_indices]
#     flat_distances = [d for i, sublist in enumerate(distances) for d in sublist if special_indices is None or i in special_indices]

#     # Calculate Spearman correlation
#     if len(flat_durations) > 1 and len(flat_distances) > 1:
#         spearman_corr, p_value = spearmanr(flat_durations, flat_distances)
#     else:
#         spearman_corr, p_value = np.nan, np.nan

#     # Scatter plot
#     plt.figure(figsize=(8, 6))
#     for i in range(16):
#         if special_indices is None or i in special_indices:
#             alpha = 1.0 if special_indices and i in special_indices else 0.7
#             size = 100 if special_indices and i in special_indices else 50
#             if i in [0, 1, 2, 3]:  # Label only for these indices
#                 plt.scatter(durations[i], distances[i], alpha=alpha, color=color_groups[i], s=size, label=f"{i}, {i+4}, {i+8}, {i+12}")
#             else:
#                 plt.scatter(durations[i], distances[i], alpha=alpha, color=color_groups[i], s=size)

#             # Overlay median values
#             if durations[i] and distances[i]:
#                 median_duration = np.median(durations[i])
#                 median_distance = np.median(distances[i])
#                 plt.scatter(median_duration, median_distance, color='none', edgecolor='black', s=100, zorder=5)
#                 plt.text(median_duration, median_distance, f"{i}", fontsize=7, color='black', ha="center", va="center",
#                          bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle', linewidth=1.5))


#     plt.title(f"Scatter Plot of Durations vs Distances ({title})")

#     labels = {
#         'distance': "Good → Bad (cm)",
#         'durations': "Good / Fast → Bad / Slow (s)"
#     }
#     plt.xlabel(f"Durations ({labels.get('durations', '')})")
#     plt.ylabel(f"Distance ({labels.get('distance', '')})")
#     plt.grid(alpha=0.5)

#     # Add Spearman correlation and number of data points to the plot
#     plt.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.4f}\nP-value: {p_value:.4f}\nData Points: {len(flat_durations)}",
#              transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

#     # Fit a hyperbolic curve
#     def hyperbolic_func(x, a, b):
#         return a / (x + b)

#     try:
#         params, _ = curve_fit(hyperbolic_func, flat_durations, flat_distances)
#         x_fit = np.linspace(min(flat_durations), max(flat_durations), 500)
#         y_fit = hyperbolic_func(x_fit, *params)
#         plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")
#     except Exception as e:
#         print(f"Hyperbolic fit failed: {e}")

#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/24/PZ'])
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['07/22/HW'])

# # Example usage
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[0, 4, 8, 12])
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[1, 5, 9, 13])
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[2, 6, 10, 14])
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[3, 7, 11, 15])

# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[0, 1, 2, 3])
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[4, 5, 6, 7])
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[8, 9, 10, 11])
# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[12, 13, 14, 15])

# scatter_plot_all_duration_distance(updated_metrics, overlay_hands=False, selected_subjects=['06/19/CZ'], special_indices=[0])

# -------------------------------------------------------------------------------------------------------------------

# # Calculate median, mean, standard deviation, max, and min for durations and distances for each subject
# def calculate_statistics_for_subjects(updated_metrics):
#     """
#     Calculates median, mean, standard deviation, max, and min for durations and distances for each subject.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.

#     Returns:
#         dict: Dictionary containing statistics for each subject.
#     """
#     statistics = {}

#     for subject, hands_data in updated_metrics.items():
#         all_durations = []
#         all_distances = []

#         for hand, metrics in hands_data.items():
#             for trial, durations in metrics['durations'].items():
#                 all_durations.extend([dur for dur in durations if not pd.isna(dur)])
#             for trial, distances in metrics['distance'].items():
#                 all_distances.extend([dist for dist in distances if not pd.isna(dist)])

#         # Calculate statistics
#         statistics[subject] = {
#             'median_durations': np.median(all_durations) if all_durations else np.nan,
#             'mean_durations': np.mean(all_durations) if all_durations else np.nan,
#             'std_durations': np.std(all_durations) if all_durations else np.nan,
#             'max_durations': np.max(all_durations) if all_durations else np.nan,
#             'min_durations': np.min(all_durations) if all_durations else np.nan,
#             'median_distance': np.median(all_distances) if all_distances else np.nan,
#             'mean_distance': np.mean(all_distances) if all_distances else np.nan,
#             'std_distance': np.std(all_distances) if all_distances else np.nan,
#             'max_distance': np.max(all_distances) if all_distances else np.nan,
#             'min_distance': np.min(all_distances) if all_distances else np.nan,
#         }

#     return statistics

# # Calculate and print statistics for each subject
# subject_statistics = calculate_statistics_for_subjects(updated_metrics)

# # Scatter plot for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation
# def scatter_plot_all_reach_indices(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
#     """
#     Plots scatter plots for all reach indices as subplots in a 4x4 layout with hyperbolic regression and Spearman correlation.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.
#         subject (str): Subject identifier.
#         hand (str): Hand ('left' or 'right').
#         metric_x (str): Metric for x-axis.
#         metric_y (str): Metric for y-axis.
#         subject_statistics (dict): Statistics containing max durations and max distances for the subject.
#     """
#     # Create subplots in a 4x4 layout
#     rows, cols = 4, 4
#     fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
#     axes = axes.flatten()

#     # Get max duration and max distance for the subject
#     max_duration = subject_statistics[subject]['max_durations']
#     min_duration = subject_statistics[subject]['min_durations']
#     max_distance = subject_statistics[subject]['max_distance']
#     min_distance = subject_statistics[subject]['min_distance']


#     for reach_index in range(16):
#         x_values = []
#         y_values = []
#         trial_colors = []

#         trials = updated_metrics[subject][hand][metric_x].keys()
#         color_palette = sns.color_palette("Reds", len(trials))  # Generate a color palette from light to dark

#         for i, trial in enumerate(trials):
#             trial_x = np.array(updated_metrics[subject][hand][metric_x][trial])
#             trial_y = np.array(updated_metrics[subject][hand][metric_y][trial])

#             # Collect data for the specified reach index
#             if reach_index < len(trial_x) and reach_index < len(trial_y):
#                 x_values.append(trial_x[reach_index])
#                 y_values.append(trial_y[reach_index])
#                 trial_colors.append(color_palette[i])  # Assign color based on trial index
#         # Remove NaN values
#         x_values = np.array(x_values)
#         y_values = np.array(y_values)
#         trial_colors = np.array(trial_colors)
#         valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
#         x_values = x_values[valid_indices]
#         y_values = y_values[valid_indices]
#         trial_colors = trial_colors[valid_indices]

#         # Calculate Spearman correlation
#         if len(x_values) > 1 and len(y_values) > 1:
#             spearman_corr, p_value = spearmanr(x_values, y_values)
#         else:
#             spearman_corr, p_value = np.nan, np.nan

#         # Scatter plot for the current reach index
#         ax = axes[reach_index]
#         ax.scatter(x_values, y_values, c=trial_colors, alpha=0.7)
#         ax.set_title(f"Reach Index {reach_index}")
#         labels = {
#             'distance': "Good → Bad (cm)",
#             'accuracy': "Bad → Good (%)",
#             'durations': "Good / Fast → Bad / Slow (s)",
#             'speed': "Bad / Slow → Good / Fast (1/s)"
#         }
#         ax.set_xlabel(f"{metric_x.capitalize()} ({labels.get(metric_x, '')})")
#         if reach_index % cols == 0:
#             ax.set_ylabel(f"{metric_y.capitalize()} ({labels.get(metric_y, '')})")

#         ax.set_xlim(min_duration * 0.9, max_duration * 1.1)
#         ax.set_ylim(min_distance * 0.9, max_distance * 1.1)

#         # Perform hyperbolic regression if applicable
#         if metric_x == 'durations' and metric_y == 'distance':
#             def hyperbolic_func(x, a, b):
#                return a / (x + b)

#             try:
#                 params, _ = curve_fit(hyperbolic_func, x_values, y_values)
#                 a, b = params
#                 x_fit = np.linspace(min(x_values), max(x_values), 500)
#                 y_fit = hyperbolic_func(x_fit, *params)
#                 ax.plot(x_fit, y_fit, color='blue', linestyle='--', label=f"Hyperbolic Fit: a={params[0]:.2f}, b={params[1]:.2f}")

#                 # ---- Find intersection with diagonal line ----
#                 A = max_distance / max_duration
#                 B = A * b
#                 C = -a

#                 discriminant = B**2 - 4*A*C
#                 if discriminant >= 0:
#                     x_roots = [(-B + np.sqrt(discriminant)) / (2*A),
#                             (-B - np.sqrt(discriminant)) / (2*A)]
#                     intersections = [(x, A*x) for x in x_roots if x > 0]
                    
#                     # Plot intersections
#                     for (xi, yi) in intersections:
#                         ax.scatter(xi, yi, color="black", s=50, zorder=5, label="Intersection")
#                         ax.text(xi, yi, f"({xi:.2f}, {yi:.2f})",
#                                 fontsize=12, color="black", ha="right", va="bottom")
#             except Exception as e:
#                 print(f"Hyperbolic regression failed for Reach Index {reach_index}: {e}")
#         else:
#             print(f"Error: Hyperbolic regression is only applicable for 'durations' vs 'distance'.")

#         # Add Spearman correlation to the plot
#         ax.text(0.05, 0.95, f"Spearman Corr: {spearman_corr:.2f}\nP-value: {p_value:.2f}",
#                 transform=ax.transAxes, fontsize=10, verticalalignment='top',
#                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

#         # Add diagonal line from (0, 0) to max duration and max distance
#         if not np.isnan(max_duration) and not np.isnan(max_distance):
#             ax.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")

#     # Hide unused subplots
#     for i in range(16, len(axes)):
#         axes[i].axis('off')

#     # Add a title for the entire figure
#     fig.suptitle(f"{subject} - {hand.capitalize()}", fontsize=16, y=1)


#     plt.tight_layout()
#     plt.show()

# scatter_plot_all_reach_indices(updated_metrics, '07/22/HW', 'left', 'durations', 'distance', subject_statistics)
# scatter_plot_all_reach_indices(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', subject_statistics)

# # Overlay 16 hyperbolic regressions, intersections, and diagonal line in one figure
# def overlay_hyperbolic_regressions(updated_metrics, subject, hand, metric_x, metric_y, subject_statistics):
#     """
#     Overlays 16 hyperbolic regressions, intersections, and diagonal line in one figure.
#     Calculates and returns Spearman correlation, p-values, and intersections.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.
#         subject (str): Subject identifier.
#         hand (str): Hand ('left' or 'right').
#         metric_x (str): Metric for x-axis.
#         metric_y (str): Metric for y-axis.
#         subject_statistics (dict): Statistics containing max durations and max distances for the subject.

#     Returns:
#         list: Results containing Spearman correlation, p-values, and intersections for each reach index.
#     """
#     # Initialize results storage
#     results = []

#     # Get max duration and max distance for the subject
#     max_duration = subject_statistics[subject]['max_durations']
#     max_distance = subject_statistics[subject]['max_distance']

#     # Create a figure
#     plt.figure(figsize=(10, 8))

#     # Generate a color palette from light blue to black
#     color_palette = sns.color_palette("light:black", 16)

#     for reach_index in range(16):
#         x_values = []
#         y_values = []

#         trials = updated_metrics[subject][hand][metric_x].keys()

#         for trial in trials:
#             trial_x = np.array(updated_metrics[subject][hand][metric_x][trial])
#             trial_y = np.array(updated_metrics[subject][hand][metric_y][trial])

#             # Collect data for the specified reach index
#             if reach_index < len(trial_x) and reach_index < len(trial_y):
#                 x_values.append(trial_x[reach_index])
#                 y_values.append(trial_y[reach_index])

#         # Remove NaN values
#         x_values = np.array(x_values)
#         y_values = np.array(y_values)
#         valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
#         x_values = x_values[valid_indices]
#         y_values = y_values[valid_indices]

#         # Calculate Spearman correlation
#         if len(x_values) > 1 and len(y_values) > 1:
#             spearman_corr, p_value = spearmanr(x_values, y_values)
#         else:
#             spearman_corr, p_value = np.nan, np.nan

#         # Perform hyperbolic regression
#         def hyperbolic_func(x, a, b):
#             return a / (x + b)

#         try:
#             params, _ = curve_fit(hyperbolic_func, x_values, y_values)
#             a, b = params
#             x_fit = np.linspace(min(x_values), max(x_values), 500)
#             y_fit = hyperbolic_func(x_fit, *params)
#             plt.plot(x_fit, y_fit, label=f"Reach {reach_index}", color=color_palette[reach_index], alpha=0.7)

#             # ---- Find intersection with diagonal line ----
#             A = max_distance / max_duration
#             B = A * b
#             C = -a

#             discriminant = B**2 - 4 * A * C
#             intersections = []
#             if discriminant >= 0:
#                 x_roots = [(-B + np.sqrt(discriminant)) / (2 * A),
#                            (-B - np.sqrt(discriminant)) / (2 * A)]
#                 intersections = [(x, A * x) for x in x_roots if x > 0]

#                 # Plot intersections
#                 for idx, (xi, yi) in enumerate(intersections):
#                     plt.scatter(xi, yi, color=color_palette[reach_index], s=50, zorder=5)
#                     # plt.text(xi, yi, f"({xi:.2f}, {yi:.2f})", fontsize=8, color=color_palette[reach_index], ha="right", va="bottom")

#             # Save results for this reach index
#             results.append({
#                 "reach_index": reach_index,
#                 "spearman_corr": spearman_corr,
#                 "p_value": p_value,
#                 "intersections": intersections
#             })

#         except Exception as e:
#             print(f"Hyperbolic regression failed for Reach Index {reach_index}: {e}")
#             results.append({
#                 "reach_index": reach_index,
#                 "spearman_corr": spearman_corr,
#                 "p_value": p_value,
#                 "intersections": None
#             })

#     # Add diagonal line from (0, 0) to max duration and max distance
#     if not np.isnan(max_duration) and not np.isnan(max_distance):
#         plt.plot([0, max_duration], [0, max_distance], color='green', linestyle='--', label="Diagonal (Max)")

#     plt.title(f"Overlay of Hyperbolic Regressions and Intersections ({subject}, {hand.capitalize()})")
#     plt.xlabel(f"{metric_x.capitalize()} (Good → Bad)")
#     plt.ylabel(f"{metric_y.capitalize()} (Bad → Good)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     return results

# results = overlay_hyperbolic_regressions(updated_metrics, '07/22/HW', 'right', 'durations', 'distance', subject_statistics)


# # Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject and hand
# def calculate_spearman_and_hyperbolic_fit(updated_metrics, subject, hand, metric_x, metric_y):
#     """
#     Calculates Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b)
#     for durations vs distances for each reach index for a specific subject and hand.

#     Parameters:
#         updated_metrics (dict): Updated metrics data.
#         subject (str): Subject identifier.
#         hand (str): Hand ('left' or 'right').
#         metric_x (str): Metric for x-axis.
#         metric_y (str): Metric for y-axis.

#     Returns:
#         dict: Dictionary containing results for each reach index.
#     """
#     results = {}

#     for reach_index in range(16):
#         x_values = []
#         y_values = []

#         trials = updated_metrics[subject][hand][metric_x].keys()

#         for trial in trials:
#             trial_x = np.array(updated_metrics[subject][hand][metric_x][trial])
#             trial_y = np.array(updated_metrics[subject][hand][metric_y][trial])

#             if reach_index < len(trial_x) and reach_index < len(trial_y):
#                 x_values.append(trial_x[reach_index])
#                 y_values.append(trial_y[reach_index])

#         # Remove NaN values
#         x_values = np.array(x_values)
#         y_values = np.array(y_values)
#         valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
#         x_values = x_values[valid_indices]
#         y_values = y_values[valid_indices]

#         # Calculate Spearman correlation
#         if len(x_values) > 1 and len(y_values) > 1:
#             spearman_corr, p_value = spearmanr(x_values, y_values)
#         else:
#             spearman_corr, p_value = np.nan, np.nan

#         # Fit a hyperbolic curve
#         def hyperbolic_func(x, a, b):
#             return a / (x + b)

#         try:
#             params, _ = curve_fit(hyperbolic_func, x_values, y_values)
#             a, b = params
#         except Exception:
#             a, b = np.nan, np.nan

#         # Store results
#         results[reach_index] = {
#             "spearman_corr": spearman_corr,
#             "p_value": p_value,
#             "data_points": len(x_values),
#             "hyperbolic_fit_a": a,
#             "hyperbolic_fit_b": b
#         }

#     return results

# spearman_hyperbolic_results = {}
# for subject in updated_metrics.keys():
#     spearman_hyperbolic_results[subject] = {}
#     for hand in ['left', 'right']:
#         spearman_hyperbolic_results[subject][hand] = calculate_spearman_and_hyperbolic_fit(updated_metrics, subject, hand, 'durations', 'distance') 

# -----------------------------------

# ## all correlations heatmap and histogram
# # Heatmap of Spearman correlations for right and left hands with median and IQR calculations
# def plot_correlation_heatmap_with_stats(results_all, hand):
#     """
#     Plots a heatmap of Spearman correlations for the specified hand (right or left).
#     X-axis represents reach indices (0 to 15), and Y-axis represents subjects.
#     Calculates and displays column (reach index) and row (subject) medians and IQRs.

#     Parameters:
#         results_all (dict): Results containing Spearman correlations for each subject and hand.
#         hand (str): Hand to plot ('right' or 'left').
#     """
#     subjects = list(results_all.keys())
#     reach_indices = range(16)

#     # Prepare data for the heatmap
#     correlation_data = []
#     for subject in subjects:
#         if hand in results_all[subject]:
#             correlations = [
#                 results_all[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
#                 for reach_index in reach_indices
#             ]
#             correlation_data.append(correlations)

#     # Convert to DataFrame for heatmap
#     correlation_df = pd.DataFrame(correlation_data, index=subjects, columns=reach_indices)

#     # Calculate column (reach index) medians and IQRs
#     column_medians = correlation_df.median()
#     column_iqr = correlation_df.quantile(0.75) - correlation_df.quantile(0.25)

#     # Calculate row (subject) medians and IQRs
#     row_medians = correlation_df.median(axis=1)
#     row_iqr = correlation_df.quantile(0.75, axis=1) - correlation_df.quantile(0.25, axis=1)

#     # Plot heatmap
#     plt.figure(figsize=(12, len(subjects) * 0.5))
#     sns.heatmap(
#         correlation_df,
#         annot=True,
#         fmt=".2f",
#         cmap="coolwarm",
#         cbar=True,
#         xticklabels=reach_indices,
#         yticklabels=subjects,
#         vmin=-1,
#         vmax=1,
#     )
#     plt.title(f"Spearman Correlation Heatmap ({hand.capitalize()} Hand)")
#     plt.xlabel("Reach Index")
#     plt.ylabel("Subjects")
#     plt.tight_layout()
#     plt.show()

#     # Print column medians and IQRs
#     print(f"Column (Reach Index) Medians ({hand.capitalize()} Hand):")
#     print(column_medians)
#     print(f"\nColumn (Reach Index) IQRs ({hand.capitalize()} Hand):")
#     print(column_iqr)

#     # Print row medians and IQRs
#     print(f"\nRow (Subject) Medians ({hand.capitalize()} Hand):")
#     print(row_medians)
#     print(f"\nRow (Subject) IQRs ({hand.capitalize()} Hand):")
#     print(row_iqr)

# # Plot heatmaps for right and left hands with stats
# plot_correlation_heatmap_with_stats(spearman_hyperbolic_results, "right")
# plot_correlation_heatmap_with_stats(spearman_hyperbolic_results, "left")


# # Plot histogram of Spearman correlations for left and right hands, overlaying them
# def plot_correlation_histogram_overlay(results_all):
#     """
#     Plots a histogram of Spearman correlations for left and right hands, overlaying them in different colors.
#     X-axis represents Spearman correlations, and Y-axis represents frequency.
#     Displays the overall median for each hand.

#     Parameters:
#         results_all (dict): Results containing Spearman correlations for each subject and hand.
#     """
#     correlations_left = []
#     correlations_right = []

#     # Collect data for left and right hands
#     for subject in results_all.keys():
#         if 'left' in results_all[subject]:
#             for reach_index in range(16):
#                 corr = results_all[subject]['left'].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr):
#                     correlations_left.append(corr)
#         if 'right' in results_all[subject]:
#             for reach_index in range(16):
#                 corr = results_all[subject]['right'].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr):
#                     correlations_right.append(corr)
    
#     # print(f"Total Left Hand Correlations: {len(correlations_left)}")
#     # print(f"Total Right Hand Correlations: {len(correlations_right)}")

#     # Calculate overall medians
#     overall_median_left = np.median(correlations_left) if correlations_left else np.nan
#     overall_median_right = np.median(correlations_right) if correlations_right else np.nan

#     # Plot histogram
#     plt.figure(figsize=(8, 6))
#     plt.hist(correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
#     plt.hist(correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')

#     # Add median lines
#     plt.axvline(overall_median_left, color='orange', linestyle='--', label=f'Median Left: {overall_median_left:.2f}')
#     plt.axvline(overall_median_right, color='blue', linestyle='--', label=f'Median Right: {overall_median_right:.2f}')
#     # plt.axvline(0, color='red', linestyle='--', label='Correlation = 0')  # Add line at correlation = 0

#     # Calculate and display IQR for left and right hands
#     iqr_left = np.percentile(correlations_left, 75) - np.percentile(correlations_left, 25)
#     iqr_right = np.percentile(correlations_right, 75) - np.percentile(correlations_right, 25)
#     plt.axvspan(np.percentile(correlations_left, 25), np.percentile(correlations_left, 75), color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
#     plt.axvspan(np.percentile(correlations_right, 25), np.percentile(correlations_right, 75), color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")

#     # Add labels, title, and legend
#     plt.title("Spearman Correlation Histogram (Overlayed Hands)")
#     plt.xlabel("Spearman Correlation")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # Plot histograms for overlayed left and right hands
# plot_correlation_histogram_overlay(spearman_hyperbolic_results)

# # Perform Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately
# def wilcoxon_test_spearman_corr(results_all):
#     """
#     Performs the Wilcoxon signed-rank test on Spearman correlation values for left and right hands separately.
#     Also calculates the median for each hand.

#     Parameters:
#         results_all (dict): Results containing Spearman correlations for each subject and hand.

#     Returns:
#         dict: Test statistics, p-values, and medians for left and right hands.
#     """
#     correlations_left = []
#     correlations_right = []

#     for subject, hands_data in results_all.items():
#         if 'left' in hands_data:
#             for reach_index in range(16):
#                 corr = hands_data['left'].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr):
#                     correlations_left.append(corr)
#         if 'right' in hands_data:
#             for reach_index in range(16):
#                 corr = hands_data['right'].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr):
#                     correlations_right.append(corr)

#     # Perform Wilcoxon signed-rank test for left hand
#     stat_left, p_value_left = wilcoxon(correlations_left)

#     # Perform Wilcoxon signed-rank test for right hand
#     stat_right, p_value_right = wilcoxon(correlations_right)

#     # Calculate medians
#     median_left = np.median(correlations_left) if correlations_left else np.nan
#     median_right = np.median(correlations_right) if correlations_right else np.nan

#     return {
#         "left": {"statistic": stat_left, "p_value": p_value_left, "median": median_left},
#         "right": {"statistic": stat_right, "p_value": p_value_right, "median": median_right}
#     }

# # Call the function to perform the Wilcoxon test
# wilcoxon_results = wilcoxon_test_spearman_corr(spearman_hyperbolic_results)

# # Print the results
# print("Wilcoxon Signed-Rank Test Results:")
# print(f"Left Hand: Statistic = {wilcoxon_results['left']['statistic']:.2f}, P-value = {wilcoxon_results['left']['p_value']:.4f}, Median = {wilcoxon_results['left']['median']:.2f}")
# print(f"Right Hand: Statistic = {wilcoxon_results['right']['statistic']:.2f}, P-value = {wilcoxon_results['right']['p_value']:.4f}, Median = {wilcoxon_results['right']['median']:.2f}")


# # Compare left vs right hands at the subject level using Wilcoxon signed-rank test on Spearman correlation values for durations and distances for each subject each reach index
# def compare_left_vs_right_hands(spearman_hyperbolic_results):
#     """
#     Compares the left and right hands at the subject level using the Wilcoxon signed-rank test
#     on paired subject Spearman correlations for durations and distances.

#     Parameters:
#         spearman_hyperbolic_results (dict): Dictionary containing Spearman correlation results for each subject.

#     Returns:
#         dict: Results of the Wilcoxon signed-rank test for durations and distances.
#     """
#     left_spearman_corrs = []
#     right_spearman_corrs = []

#     for subject, hands_data in spearman_hyperbolic_results.items():
#         if 'left' in hands_data and 'right' in hands_data:
#             left_corrs = [hands_data['left'][reach_index]['spearman_corr'] for reach_index in hands_data['left'] if not pd.isna(hands_data['left'][reach_index]['spearman_corr'])]
#             right_corrs = [hands_data['right'][reach_index]['spearman_corr'] for reach_index in hands_data['right'] if not pd.isna(hands_data['right'][reach_index]['spearman_corr'])]
            
#             if len(left_corrs) == len(right_corrs) and len(left_corrs) > 0:
#                 left_spearman_corrs.extend(left_corrs)
#                 right_spearman_corrs.extend(right_corrs)

#     # Perform Wilcoxon signed-rank test for Spearman correlations
#     stat_corrs, p_value_corrs = wilcoxon(left_spearman_corrs, right_spearman_corrs)

#     results = {
#         "spearman_correlations": {"statistic": stat_corrs, "p_value": p_value_corrs},
#         "data_points": {"left": len(left_spearman_corrs), "right": len(right_spearman_corrs)}
#     }

#     # Print the results of the Wilcoxon signed-rank test
#     print(f"Wilcoxon Signed-Rank Test Results:")
#     print(f"Spearman Correlations: Statistic = {stat_corrs:.2f}, P-value = {p_value_corrs:.4f}")
#     print(f"Data Points: Left Hand = {len(left_spearman_corrs)}, Right Hand = {len(right_spearman_corrs)}")

#     return results

# # Call the function to compare left vs right hands
# compare_left_vs_right_hands(spearman_hyperbolic_results)

# # -----------------------------------

# ## correlations grouped by reach index boxplot and histogram
# # Box plot of Spearman correlations for each reach index (rotated 90 degrees to the left)
# def plot_correlation_boxplot(results_all, hand):
#     """
#     Plots a box plot of Spearman correlations for each reach index, rotated 90 degrees to the left.
#     Y-axis represents reach indices (0 to 15), and X-axis represents Spearman correlations for each subject.
#     Displays the median value for each reach index and the overall median across all reach types.

#     Parameters:
#         results_all (dict): Results containing Spearman correlations for each subject and hand.
#         hand (str): Hand to plot ('right' or 'left').
#     """
#     reach_indices = range(16)
#     correlation_data = {reach_index: [] for reach_index in reach_indices}

#     # Collect data for each reach index
#     for subject in results_all.keys():
#         if hand in results_all[subject]:
#             for reach_index in reach_indices:
#                 corr = results_all[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr):
#                     correlation_data[reach_index].append(corr)

#     # Convert to DataFrame for box plot
#     correlation_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in correlation_data.items()]))

#     # Calculate overall median across all reach types
#     overall_median = correlation_df.median().median()

#     # Plot box plot (rotated)
#     plt.figure(figsize=(8, 6))
#     sns.boxplot(data=correlation_df, palette="coolwarm", orient="h", showmeans=False, width=0.6)
#     plt.axvline(0, color='red', linestyle='--', label='Correlation = 0')  # Add vertical line at correlation 0
#     plt.axvline(overall_median, color='purple', linestyle='--', label=f'Overall Median: {overall_median:.2f}')  # Add overall median line
#     plt.title(f"Spearman Correlation Box Plot ({hand.capitalize()} Hand)")
#     plt.ylabel("Reach Index")
#     plt.xlabel("Spearman Correlation")
#     plt.grid(alpha=0.5)

#     # Add median values to the plot
#     medians = correlation_df.median()
#     for i, median in enumerate(medians):
#         plt.text(median, i, f"{median:.2f}", verticalalignment='center', color='black', fontsize=8, horizontalalignment='right')

#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # Plot box plots for right and left hands
# plot_correlation_boxplot(spearman_hyperbolic_results, "right")
# plot_correlation_boxplot(spearman_hyperbolic_results, "left")

# # Plot histograms for median Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
# def plot_median_correlation_histogram_with_stats(results_all):
#     """
#     Plots histograms of median Spearman correlations for each reach index,
#     overlaying left and right hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

#     Parameters:
#         results_all (dict): Results containing Spearman correlations for each subject and hand.
#     """
#     median_correlations_left = []
#     median_correlations_right = []

#     # Collect median correlation for each reach index
#     for reach_index in range(16):
#         correlations_left = []
#         correlations_right = []
#         for subject in results_all.keys():
#             if 'left' in results_all[subject]:
#                 corr_left = results_all[subject]['left'].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr_left):
#                     correlations_left.append(corr_left)
#             if 'right' in results_all[subject]:
#                 corr_right = results_all[subject]['right'].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr_right):
#                     correlations_right.append(corr_right)
#         if correlations_left:
#             median_correlations_left.append(np.median(correlations_left))
#         if correlations_right:
#             median_correlations_right.append(np.median(correlations_right))

#     # Calculate statistics
#     median_left = np.median(median_correlations_left)
#     iqr_left = np.percentile(median_correlations_left, 75) - np.percentile(median_correlations_left, 25)
#     q1_left = np.percentile(median_correlations_left, 25)
#     q3_left = np.percentile(median_correlations_left, 75)

#     median_right = np.median(median_correlations_right)
#     iqr_right = np.percentile(median_correlations_right, 75) - np.percentile(median_correlations_right, 25)
#     q1_right = np.percentile(median_correlations_right, 25)
#     q3_right = np.percentile(median_correlations_right, 75)

#     # Perform Wilcoxon signed-rank test for each hand separately
#     stat_left, p_value_left = wilcoxon(median_correlations_left)
#     stat_right, p_value_right = wilcoxon(median_correlations_right)

#     # Plot histogram for median Spearman correlations
#     plt.figure(figsize=(8, 6))
#     plt.hist(median_correlations_left, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Left Hand')
#     plt.hist(median_correlations_right, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Right Hand')
#     plt.axvline(median_left, color='orange', linestyle='--', label=f"Median Left: {median_left:.2f}")
#     plt.axvline(median_right, color='blue', linestyle='--', label=f"Median Right: {median_right:.2f}")
#     plt.axvspan(q1_left, q3_left, color='orange', alpha=0.2, label=f"IQR Left: {iqr_left:.2f}")
#     plt.axvspan(q1_right, q3_right, color='blue', alpha=0.2, label=f"IQR Right: {iqr_right:.2f}")
#     plt.title("Histogram of Median Spearman Correlations by Hand")
#     plt.xlabel("Median Spearman Correlation")
#     plt.ylabel("Frequency")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # Print statistics
#     print(f"Left Hand: Median = {median_left:.2f}, IQR = {iqr_left:.2f}")
#     print(f"Right Hand: Median = {median_right:.2f}, IQR = {iqr_right:.2f}")
#     print(f"Wilcoxon Signed-Rank Test (Left Hand): Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}")
#     print(f"Wilcoxon Signed-Rank Test (Right Hand): Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}")

# plot_median_correlation_histogram_with_stats(spearman_hyperbolic_results)

# # Compare left vs right hands at the subject level using Wilcoxon signed-rank test on median Spearman correlations for durations and distances for each reach index
# def compare_left_vs_right_hands(results_all):
#     """
#     Compares the left and right hands at the subject level using the Wilcoxon signed-rank test
#     on median Spearman correlations for durations and distances.

#     Parameters:
#         results_all (dict): Results containing Spearman correlations for each subject and hand.

#     Returns:
#         dict: Results of the Wilcoxon signed-rank test, median correlations, and data points for left and right hands.
#     """
#     median_correlations_left = []
#     median_correlations_right = []


#     # Collect median correlation for each reach index
#     for reach_index in range(16):
#         correlations_left = []
#         correlations_right = []
#         for subject in results_all.keys():
#             if 'left' in results_all[subject]:
#                 corr_left = results_all[subject]['left'].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr_left):
#                     correlations_left.append(corr_left)
#             if 'right' in results_all[subject]:
#                 corr_right = results_all[subject]['right'].get(reach_index, {}).get("spearman_corr", np.nan)
#                 if not np.isnan(corr_right):
#                     correlations_right.append(corr_right)
#         if correlations_left:
#             median_correlations_left.append(np.median(correlations_left))
#         if correlations_right:
#             median_correlations_right.append(np.median(correlations_right))

#     # Perform Wilcoxon signed-rank test
#     stat, p_value = wilcoxon(median_correlations_left, median_correlations_right)

#     return {
#         "median_correlations_left": median_correlations_left,
#         "median_correlations_right": median_correlations_right,
#         "data_points_left": len(median_correlations_left),
#         "data_points_right": len(median_correlations_right),
#         "wilcoxon_statistic": stat,
#         "wilcoxon_p_value": p_value
#     }

# # Call the function to compare left vs right hands
# comparison_results = compare_left_vs_right_hands(spearman_hyperbolic_results)
# # Print the results of the Wilcoxon signed-rank test
# print(f"Wilcoxon Signed-Rank Test Results (Median Correlations): Statistic = {comparison_results['wilcoxon_statistic']:.2f}, P-value = {comparison_results['wilcoxon_p_value']:.4f}")
# print(f"Number of data points: Left Hand = {comparison_results['data_points_left']}, Right Hand = {comparison_results['data_points_right']}")

# -----------------------------------

## correlations grouped by subject boxplot and histogram
# Box plot of Spearman correlations for each subject
def plot_correlation_boxplot_per_subject(results_all, hand):
    """
    Plots a box plot of Spearman correlations for each subject.
    X-axis represents subjects, and Y-axis represents Spearman correlations across all reach indices.
    Displays the median value for each subject and the overall median.

    Parameters:
        results_all (dict): Results containing Spearman correlations for each subject and hand.
        hand (str): Hand to plot ('right' or 'left').
    """
    subjects = list(results_all.keys())
    correlation_data = {subject: [] for subject in subjects}

    # Collect data for each subject
    for subject in results_all.keys():
        if hand in results_all[subject]:
            for reach_index in range(16):
                corr = results_all[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
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

# Plot box plots for right and left hands
plot_correlation_boxplot_per_subject(spearman_hyperbolic_results, "right")
plot_correlation_boxplot_per_subject(spearman_hyperbolic_results, "left")

# Plot histograms for median Spearman correlations, overlaying hands, and report median, IQR, and Wilcoxon signed-rank test result by hand
def plot_median_correlation_histogram_with_stats(results_all):
    """
    Plots histograms of median Spearman correlations for each subject,
    overlaying left and right hands in different colors. Reports median, IQR, and Wilcoxon signed-rank test result by hand.

    Parameters:
        results_all (dict): Results containing Spearman correlations for each subject and hand.
    """
    median_correlations_left = []
    median_correlations_right = []

    # Collect median correlation for each subject
    for subject in results_all.keys():
        for hand in ['left', 'right']:
            if hand in results_all[subject]:
                correlations = [
                    results_all[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
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

    # Print statistics
    print(f"Left Hand: Median = {median_left:.2f}, IQR = {iqr_left:.2f}")
    print(f"Right Hand: Median = {median_right:.2f}, IQR = {iqr_right:.2f}")
    print(f"Wilcoxon Signed-Rank Test (Left Hand): Statistic = {stat_left:.2f}, P-value = {p_value_left:.4f}")
    print(f"Wilcoxon Signed-Rank Test (Right Hand): Statistic = {stat_right:.2f}, P-value = {p_value_right:.4f}")

plot_median_correlation_histogram_with_stats(spearman_hyperbolic_results)



# Compare left vs right hands using Wilcoxon signed-rank test on median Spearman correlations
def compare_median_correlations_left_vs_right(results_all):
    """
    Compares the left and right hands using the Wilcoxon signed-rank test
    on median Spearman correlations for durations and distances.

    Parameters:
        results_all (dict): Results containing Spearman correlations for each subject and hand.

    Returns:
        dict: Results of the Wilcoxon signed-rank test, median correlations, and data points for left and right hands.
    """
    median_correlations_left = []
    median_correlations_right = []

    # Collect median correlation for each subject
    for subject in results_all.keys():
        for hand in ['left', 'right']:
            if hand in results_all[subject]:
                correlations = [
                    results_all[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
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

    return {
        "median_correlations_left": median_correlations_left,
        "median_correlations_right": median_correlations_right,
        "data_points_left": len(median_correlations_left),
        "data_points_right": len(median_correlations_right),
        "wilcoxon_statistic": stat,
        "wilcoxon_p_value": p_value
    }

# Call the function to compare left vs right hands
comparison_results = compare_median_correlations_left_vs_right(spearman_hyperbolic_results)

# Print the results of the Wilcoxon signed-rank test
print(f"Wilcoxon Signed-Rank Test Results (Median Correlations): Statistic = {comparison_results['wilcoxon_statistic']:.2f}, P-value = {comparison_results['wilcoxon_p_value']:.4f}")
print(f"Number of data points: Left Hand = {comparison_results['data_points_left']}, Right Hand = {comparison_results['data_points_right']}")

