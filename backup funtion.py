
# method 1
# overlay 16 hyperbolic curves and intersection points for each subject acorss 16 reach as subplots, each with its own x and y axis
def overlay_hyperbolic_curves_all_subjects(all_combined_metrics, subjects, hand, metric_x, metric_y, add_diagonal=False):
    """
    Overlay all 16 hyperbolic curves and intersection points for each subject as subplots, 
    each with its own x and y axis.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subjects (list): List of subject identifiers.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        add_diagonal (bool): Whether to add diagonal lines and calculate intersection points.
    """
    num_reaches = 16
    max_subjects_per_row = 4
    num_subjects = len(subjects)
    num_rows = (num_subjects + max_subjects_per_row - 1) // max_subjects_per_row

    fig, axes = plt.subplots(num_rows, max_subjects_per_row, figsize=(6 * max_subjects_per_row, 6 * num_rows))
    axes = axes.flatten()

    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        colors = sns.color_palette("Blues", num_reaches)  # Generate a color palette from light blue to dark blue

        for reach_index in range(num_reaches):
            x_values = []
            y_values = []

            trials = all_combined_metrics[subject][hand][metric_x].keys()

            for trial in trials:
                trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                # Collect data for the current reach index
                if reach_index < len(trial_x) and reach_index < len(trial_y):
                    x_values.append(trial_x[reach_index])
                    y_values.append(trial_y[reach_index])

            # Remove NaN values
            x_values = np.array(x_values)
            y_values = np.array(y_values)
            valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
            x_values = x_values[valid_indices]
            y_values = y_values[valid_indices]

            # Apply hyperbolic regression
            if len(x_values) > 1 and len(y_values) > 1:
                def hyperbolic_model(x, a, b):
                    return a + b / x

                try:
                    params, _ = curve_fit(hyperbolic_model, x_values, y_values, p0=(0, 1))
                    a, b = params
                    fit_x = np.linspace(min(x_values), max(x_values), 100)
                    fit_y = hyperbolic_model(fit_x, a, b)
                    ax.plot(fit_x, fit_y, color=colors[reach_index], label=f'Reach {reach_index + 1}')

                    if add_diagonal:
                        # Calculate diagonal line
                        x_mean, x_sd = np.mean(x_values), np.std(x_values)
                        y_mean, y_sd = np.mean(y_values), np.std(y_values)
                        m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
                        x_diag = np.linspace(min(x_values), max(x_values), 100)
                        y_diag = m * x_diag

                        # Solve intersection point
                        A, B, C = m, -a, -b
                        discriminant = B**2 - 4 * A * C
                        if discriminant >= 0:
                            sqrt_disc = np.sqrt(discriminant)
                            root1 = (-B + sqrt_disc) / (2 * A)
                            root2 = (-B - sqrt_disc) / (2 * A)
                            positive_roots = [r for r in (root1, root2) if r > 0]
                            P_star = max(positive_roots) if positive_roots else np.nan
                            S_star = m * P_star if np.isfinite(P_star) else np.nan

                            if np.isfinite(P_star) and np.isfinite(S_star):
                                ax.scatter(P_star, S_star, color=colors[reach_index], edgecolor='black', zorder=5)

                except RuntimeError:
                    print(f"Hyperbolic regression failed for Reach {reach_index + 1}.")

        ax.set_title(f"{subject}, {hand.capitalize()}", fontsize=12)
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for ax in axes[num_subjects:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

overlay_hyperbolic_curves_all_subjects(all_combined_metrics, All_dates, 'left', 'accuracy', 'speed', add_diagonal=True)

# # Function to calculate motor acuity from hyperbolic regression and return distance, r values, and Spearman correlation
def get_motor_acuity_from_hyperbolic(all_combined_metrics, subject, hand, metric_x, metric_y, reach_indices, add_diagonal=False):
    """
    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        reach_indices (int or list): Reach index or list of indices to include.
        add_diagonal (bool): Whether to calculate diagonal line and intersection.

    Returns:
        dict: Dictionary containing distances, r values, Spearman correlation, and Spearman p-value.
    """
    if isinstance(reach_indices, int):
        reach_indices = [reach_indices]  # Convert single index to list

    x_values = []
    y_values = []

    trials = all_combined_metrics[subject][hand][metric_x].keys()

    for trial in trials:
        trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
        trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

        for reach_index in reach_indices:
            if reach_index < len(trial_x) and reach_index < len(trial_y):
                x_values.append(trial_x[reach_index])
                y_values.append(trial_y[reach_index])

    # Convert to arrays and remove NaN
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]

    results = {"distance": None, "R^2": None, "spearman_r": None, "spearman_p": None, "params": None, "intersection": None}

    # Apply hyperbolic regression
    if len(x_values) > 1 and len(y_values) > 1:
        def hyperbolic_model(x, a, b):
            return a + b / x

        try:
            params, _ = curve_fit(hyperbolic_model, x_values, y_values, p0=(0, 1))
            a, b = params
            # results["params"] = {"a": a, "b": b}

            # Calculate r (correlation coefficient)
            residuals = y_values - hyperbolic_model(x_values, a, b)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_values - np.mean(y_values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            results["R^2"] = r_squared

            if add_diagonal:
                # Calculate diagonal slope
                x_mean, x_sd = np.mean(x_values), np.std(x_values)
                y_mean, y_sd = np.mean(y_values), np.std(y_values)
                m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)

                # Solve intersection between y = m*x and y = a + b/x
                A, B, C = m, -a, -b
                discriminant = B**2 - 4 * A * C
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    root1 = (-B + sqrt_disc) / (2 * A)
                    root2 = (-B - sqrt_disc) / (2 * A)
                    positive_roots = [r for r in (root1, root2) if r > 0]
                    P_star = max(positive_roots) if positive_roots else np.nan
                    S_star = m * P_star if np.isfinite(P_star) else np.nan

                    if np.isfinite(P_star) and np.isfinite(S_star):
                        # results["intersection"] = (P_star, S_star)
                        results["distance"] = np.sqrt(P_star**2 + S_star**2)

        except RuntimeError:
            print("Hyperbolic regression failed.")

    # Calculate Spearman correlation
    if len(x_values) > 1 and len(y_values) > 1:
        spearman_r, spearman_p = spearmanr(x_values, y_values)
        results["spearman_r"] = spearman_r
        results["spearman_p"] = spearman_p

    return results

def calculate_motor_acuity_and_correlation(all_combined_metrics):
    """
    Calculate motor acuity and correlation for each subject, hand, and reach index.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.

    Returns:
        dict: Motor acuity values for each subject, hand, and reach index.
        dict: Correlation values for each subject, hand, and reach index.
    """
    motor_acuity = {}
    correlation = {}

    for subject in all_combined_metrics.keys():
        motor_acuity[subject] = {}
        correlation[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            motor_acuity[subject][hand] = {}
            correlation[subject][hand] = {}
            for reach_indices in range(16):
                results = get_motor_acuity_from_hyperbolic(all_combined_metrics, subject, hand, 'accuracy', 'speed', [reach_indices], add_diagonal=True)
                motor_acuity[subject][hand][reach_indices] = results.get("distance", None)
                correlation[subject][hand][reach_indices] = results.get("spearman_r", None)

    return motor_acuity, correlation

# Calculate motor acuity and correlation for each subject, hand, and reach index across trials, per subject have 16 motor acuity values
motor_acuity, correlation = calculate_motor_acuity_and_correlation(all_combined_metrics)

# # Plot histograms of correlation values for each reach type, separated by hand, across all subjects
def plot_correlation_histogram_per_reach(correlation):
    """
    Plot histograms of correlation values for each reach type, separated by hand, across all subjects.
    Also return the correlation values and medians for each reach type per hand.

    Parameters:
        correlation (dict): Correlation values for each subject, hand, and reach index.

    Returns:
        dict: Correlation values for each reach type per hand.
        dict: Median correlation values for each reach type per hand.
    """
    hands = ['right', 'left']
    num_reaches = 16
    reach_correlation_data = {reach_index: {hand: [] for hand in hands} for reach_index in range(num_reaches)}
    reach_correlation_medians = {reach_index: {hand: None for hand in hands} for reach_index in range(num_reaches)}

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for reach_index in range(num_reaches):
        ax = axes[reach_index]
        for hand in hands:
            all_correlation_values = []

            for subject in correlation.keys():
                if hand in correlation[subject]:
                    r_value = correlation[subject][hand].get(reach_index, None)
                    if r_value is not None:  # Exclude None values
                        all_correlation_values.append(r_value)

            # Store correlation values for the current reach type and hand
            reach_correlation_data[reach_index][hand] = all_correlation_values

            # Calculate and store the median
            if all_correlation_values:
                median_value = np.median(all_correlation_values)
                reach_correlation_medians[reach_index][hand] = median_value

                # Annotate the median on the plot
                color = 'darkblue' if hand == 'right' else 'darkorange'
                ax.axvline(median_value, color=color, linestyle='--', label=f"{hand.capitalize()} Median: {median_value:.2f}")

            # Plot histogram for the current reach type and hand
            ax.hist(
                all_correlation_values, bins=15, alpha=0.5, edgecolor='black',
                label=f"{hand.capitalize()} Hand"
            )

        ax.set_title(f"Reach {reach_index + 1}", fontsize=12)
        ax.set_xlabel("Correlation (r)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return reach_correlation_data, reach_correlation_medians

# Example usage
reach_correlation_data, reach_correlation_medians = plot_correlation_histogram_per_reach(correlation)


plt.figure(figsize=(10, 5))

# Extract median values for each hand across all reach indices
left_hand_medians = [reach_correlation_medians[reach_index]['left'] for reach_index in reach_correlation_medians if reach_correlation_medians[reach_index]['left'] is not None]
right_hand_medians = [reach_correlation_medians[reach_index]['right'] for reach_index in reach_correlation_medians if reach_correlation_medians[reach_index]['right'] is not None]

plt.hist(left_hand_medians, bins=8, alpha=0.6, label='Left Hand', color='skyblue')
plt.hist(right_hand_medians, bins=8, alpha=0.6, label='Right Hand', color='salmon')

plt.xlabel('Median Reach Correlation')
plt.ylabel('Frequency')
plt.title('Histogram of Reach Correlation Medians per Hand')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()


# Histogram of correlation across reaches for each subject and hand, with separate subplots for right and left hands
def plot_correlation_histogram_overlay(correlation):
    """
    Plot a histogram of correlation values across all reaches for each subject, with separate subplots for right and left hands.
    Also count the number of correlation values below 0 and above 0.

    Parameters:
        correlation (dict): Correlation values for each subject, hand, and reach index.
    """
    color_palette = sns.color_palette("husl", len(correlation.keys()))  # Generate colors for all subjects
    hands = ['right', 'left']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for hand_idx, hand in enumerate(hands):
        ax = axes[hand_idx]
        below_zero_count = 0
        above_zero_count = 0

        for idx, subject in enumerate(correlation.keys()):
            all_correlation_values = []

            if hand in correlation[subject]:
                for reach_index in correlation[subject][hand].keys():
                    r_value = correlation[subject][hand][reach_index]
                    if r_value is not None:  # Exclude None values
                        all_correlation_values.append(r_value)
                        if r_value < 0:
                            below_zero_count += 1
                        elif r_value > 0:
                            above_zero_count += 1

                # Plot histogram for the current subject and hand
                ax.hist(
                    all_correlation_values, bins=20, alpha=0.5, edgecolor='black',
                    color=color_palette[idx], label=f"{subject}"
                )

        ax.set_title(f"Histogram of Correlation Values ({hand.capitalize()} Hand)", fontsize=14)
        ax.set_xlabel("Correlation (r)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(alpha=0.3)

        # Display counts below and above 0
        ax.text(0.05, 0.95, f"Below 0: {below_zero_count}\nAbove 0: {above_zero_count}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='blue')

    plt.tight_layout()
    plt.show()

# Example usage
plot_correlation_histogram_overlay(correlation)

# Calculate the average motor acuity and average correlation across reaches only when r < 0
def calculate_average_motor_acuity_and_correlation(motor_acuity, correlation):
    """
    Calculate the average motor acuity and average correlation across reaches for each subject and hand, 
    considering only reaches where the correlation (r) is smaller than 0.

    Parameters:
        motor_acuity (dict): Motor acuity values for each subject, hand, and reach index.
        correlation (dict): Correlation values (r) for each subject, hand, and reach index.

    Returns:
        dict: Average motor acuity and correlation for each subject and hand.
    """
    results = {}

    for subject in motor_acuity.keys():
        results[subject] = {}
        for hand in motor_acuity[subject].keys():
            acuity_values = []
            correlation_values = []
            for reach_index in motor_acuity[subject][hand].keys():
                if correlation[subject][hand][reach_index] < 0:
                    acuity_values.append(motor_acuity[subject][hand][reach_index])
                    correlation_values.append(correlation[subject][hand][reach_index])
            # Calculate the averages only if there are valid values
            results[subject][hand] = {
                "average_motor_acuity(corr_<0)": np.mean(acuity_values) if acuity_values else np.nan,
                "average_correlation_<0": np.mean(correlation_values) if correlation_values else np.nan,
            }

    return results
average_results = calculate_average_motor_acuity_and_correlation(motor_acuity, correlation)

# Plot the average motor acuity and correlation values in two subplots (top and bottom)
def plot_average_motor_acuity_and_correlation(average_results):
    """
    Plot the average motor acuity and correlation values for each subject and hand in two subplots.

    Parameters:
        average_results (dict): Average motor acuity and correlation values for each subject and hand.
    """
    subjects = list(average_results.keys())
    hands = ['right', 'left']
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    bar_width = 0.35
    index = np.arange(len(subjects))

    # Top subplot: Motor Acuity
    for i, hand in enumerate(hands):
        acuity_values = [average_results[subject][hand]["average_motor_acuity(corr_<0)"] for subject in subjects]
        axes[0].bar(index + i * bar_width, acuity_values, bar_width, label=f"{hand.capitalize()} Motor Acuity", alpha=0.7)

    axes[0].set_ylabel("Motor Acuity", fontsize=12)
    axes[0].set_title("Average Motor Acuity by Subject and Hand", fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bottom subplot: Correlation
    for i, hand in enumerate(hands):
        correlation_values = [average_results[subject][hand]["average_correlation_<0"] for subject in subjects]
        axes[1].bar(index + i * bar_width, correlation_values, bar_width, label=f"{hand.capitalize()} Correlation", alpha=0.7)

    axes[1].set_xlabel("Subjects", fontsize=12)
    axes[1].set_ylabel("Correlation", fontsize=12)
    axes[1].set_title("Average Correlation by Subject and Hand", fontsize=14)
    axes[1].set_xticks(index + bar_width / 2)
    axes[1].set_xticklabels(subjects, fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
plot_average_motor_acuity_and_correlation(average_results)

# # # -------------------------------------------------------------------------------------------------------------------
# method 2
# Calculate the mean of speed and accuracy for each subject, each hand, and each reach type
def calculate_mean_speed_accuracy(all_combined_metrics, selected_trials=None):
    """
    Calculate the mean speed and accuracy for each subject, hand, and reach type.
    Optionally, calculate the mean only for selected trials.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        selected_trials (list, optional): List of trial indices to include. If None, include all trials.

    Returns:
        dict: Mean speed and accuracy values for each subject, hand, and reach type.
    """
    mean_results = {}
    for subject in all_combined_metrics.keys():
        mean_results[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            mean_results[subject][hand] = {}
            for metric in ['speed', 'accuracy']:
                mean_results[subject][hand][metric] = {}
                for reach_index in range(16):  # Assuming there are 16 reach types
                    values = []

                    for trial_index, trial in enumerate(all_combined_metrics[subject][hand][metric].keys()):
                        # Include only selected trials if specified
                        if selected_trials is not None and trial_index not in selected_trials:
                            continue

                        trial_values = np.array(all_combined_metrics[subject][hand][metric][trial])

                        # Ensure the reach index is valid
                        if reach_index < len(trial_values):
                            values.append(trial_values[reach_index])

                    # Remove NaN values and calculate the mean
                    values = np.array(values)
                    valid_values = values[~np.isnan(values)]
                    mean_results[subject][hand][metric][reach_index] = np.mean(valid_values) if len(valid_values) > 0 else np.nan

    return mean_results

# Example usage
# Select all trials
mean_speed_accuracy_all = calculate_mean_speed_accuracy(all_combined_metrics)

# Select first half of trials
selected_trials_first_half = list(range(0, 16))
mean_speed_accuracy_first_half = calculate_mean_speed_accuracy(all_combined_metrics, selected_trials=selected_trials_first_half)

# Select second half of trials
selected_trials_second_half = list(range(17, 32))
mean_speed_accuracy_second_half = calculate_mean_speed_accuracy(all_combined_metrics, selected_trials=selected_trials_second_half)

def plot_mean_accuracy_vs_speed_multiple_subjects(mean_speed_accuracy, subjects, hand, fit_hyperbolic=False):
    """
    Scatter plot for mean accuracy vs mean speed for the specified subjects and hand, with a maximum of 5 subjects per row.
    Optionally fits a hyperbolic regression curve.

    Parameters:
        mean_speed_accuracy (dict): Dictionary containing mean speed and accuracy values.
        subjects (list): List of subject identifiers.
        hand (str): Hand ('left' or 'right').
        fit_hyperbolic (bool): Whether to fit and plot a hyperbolic regression curve.
    """
    max_subjects_per_row = 5
    num_subjects = len(subjects)
    num_rows = (num_subjects + max_subjects_per_row - 1) // max_subjects_per_row

    fig, axes = plt.subplots(num_rows, min(num_subjects, max_subjects_per_row), figsize=(6 * min(num_subjects, max_subjects_per_row), 6 * num_rows), sharey=True)
    axes = axes.flatten() if num_subjects > 1 else [axes]

    for i, (ax, subject) in enumerate(zip(axes, subjects)):
        accuracy_values = list(mean_speed_accuracy[subject][hand]['accuracy'].values())
        speed_values = list(mean_speed_accuracy[subject][hand]['speed'].values())

        ax.scatter(accuracy_values, speed_values, color='blue', alpha=0.7)
        ax.set_title(f"Subject: {subject}, Hand: {hand.capitalize()}", fontsize=12)
        ax.set_xlabel("Mean Accuracy", fontsize=10)
        ax.set_ylabel("Mean Speed", fontsize=10)
        ax.grid(alpha=0.3)

        # Annotate each point with its reach index
        for j, (accuracy, speed) in enumerate(zip(accuracy_values, speed_values)):
            ax.text(accuracy, speed, str(j), fontsize=8, ha='right', va='bottom')

        # Fit and plot hyperbolic regression if enabled
        if fit_hyperbolic and len(accuracy_values) > 1 and len(speed_values) > 1:
            def hyperbolic_model(x, a, b):
                return a + b / x

            try:
                params, _ = curve_fit(hyperbolic_model, accuracy_values, speed_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(accuracy_values), max(accuracy_values), 100)
                fit_y = hyperbolic_model(fit_x, a, b)
                ax.plot(fit_x, fit_y, color='red', linestyle='--', label='Hyperbolic Fit')
                ax.legend(fontsize=8)
            except RuntimeError:
                ax.text(0.05, 0.85, "Fit failed", transform=ax.transAxes, fontsize=8, color='red')

    # Hide unused subplots
    for ax in axes[num_subjects:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plot_mean_accuracy_vs_speed_multiple_subjects(mean_speed_accuracy_all, All_dates, 'left', fit_hyperbolic=True)
plot_mean_accuracy_vs_speed_multiple_subjects(mean_speed_accuracy_first_half, ['07/22/HW'], 'left', fit_hyperbolic=True)
plot_mean_accuracy_vs_speed_multiple_subjects(mean_speed_accuracy_second_half, ['07/22/HW'], 'left', fit_hyperbolic=True)


def plot_mean_accuracy_vs_speed_all_subjects(mean_speed_accuracy, fit_hyperbolic=False):
    """
    Scatter plot for mean accuracy vs mean speed for all subjects, each subject as a subplot.
    Optionally fits a hyperbolic regression curve and adds a diagonal line with intersection points.
    Separate plots for left and right hands.

    Parameters:
        mean_speed_accuracy (dict): Dictionary containing mean speed and accuracy values.
        fit_hyperbolic (bool): Whether to fit and plot a hyperbolic regression curve.
    Returns:
        dict: Intersection points and Spearman correlations for each subject and hand.
    """
    hands = ['left', 'right']
    results = {hand: {} for hand in hands}

    for hand in hands:
        subjects = list(mean_speed_accuracy.keys())
        max_subjects_per_row = 4
        num_subjects = len(subjects)
        num_rows = (num_subjects + max_subjects_per_row - 1) // max_subjects_per_row

        fig, axes = plt.subplots(num_rows, min(num_subjects, max_subjects_per_row), figsize=(6 * min(num_subjects, max_subjects_per_row), 6 * num_rows), sharey=True)
        axes = axes.flatten() if num_subjects > 1 else [axes]

        for i, (ax, subject) in enumerate(zip(axes, subjects)):
            accuracy_values = list(mean_speed_accuracy[subject][hand]['accuracy'].values())
            speed_values = list(mean_speed_accuracy[subject][hand]['speed'].values())

            ax.scatter(accuracy_values, speed_values, color='blue', alpha=0.7)
            ax.set_title(f"{subject}, {hand.capitalize()}", fontsize=12)
            ax.set_xlabel("Mean Accuracy", fontsize=10)
            ax.set_ylabel("Mean Speed", fontsize=10)
            ax.grid(alpha=0.3)

            # Annotate each point with its reach index
            for j, (accuracy, speed) in enumerate(zip(accuracy_values, speed_values)):
                ax.text(accuracy, speed, str(j), fontsize=8, ha='right', va='bottom')

            # Calculate Spearman correlation
            if len(accuracy_values) > 1 and len(speed_values) > 1:
                spearman_corr, spearman_p = spearmanr(accuracy_values, speed_values)
                ax.text(0.05, 0.95, f"Spearman r: {spearman_corr:.2f}\np: {spearman_p:.4f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
                results[hand][subject] = {'spearman_corr': spearman_corr, 'spearman_p': spearman_p}
            else:
                results[hand][subject] = {'spearman_corr': np.nan, 'spearman_p': np.nan}

            # Fit and plot hyperbolic regression if enabled
            if fit_hyperbolic and len(accuracy_values) > 1 and len(speed_values) > 1:
                def hyperbolic_model(x, a, b):
                    return a + b / x

                try:
                    params, _ = curve_fit(hyperbolic_model, accuracy_values, speed_values, p0=(0, 1))
                    a, b = params
                    fit_x = np.linspace(min(accuracy_values), max(accuracy_values), 100)
                    fit_y = hyperbolic_model(fit_x, a, b)
                    ax.plot(fit_x, fit_y, color='red', linestyle='--', label='Hyperbolic Fit')

                    # Calculate diagonal line
                    x_mean, x_sd = np.mean(accuracy_values), np.std(accuracy_values)
                    y_mean, y_sd = np.mean(speed_values), np.std(speed_values)
                    m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
                    x_diag = np.linspace(min(accuracy_values), max(accuracy_values), 100)
                    y_diag = m * x_diag
                    ax.plot(x_diag, y_diag, color='green', linestyle='--', label='Diagonal Line')

                    # Solve for intersection points
                    A, B, C = m, -a, -b
                    discriminant = B**2 - 4 * A * C
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        root1 = (-B + sqrt_disc) / (2 * A)
                        root2 = (-B - sqrt_disc) / (2 * A)
                        positive_roots = [r for r in (root1, root2) if r > 0]
                        P_star = max(positive_roots) if positive_roots else np.nan
                        S_star = m * P_star if np.isfinite(P_star) else np.nan

                        if np.isfinite(P_star) and np.isfinite(S_star):
                            ax.scatter(P_star, S_star, color='purple', label='Intersection Point', zorder=5)
                            ax.text(P_star, S_star, f"({P_star:.2f}, {S_star:.2f})", color='purple', fontsize=8)
                            # results[hand][subject]['intersection'] = (P_star, S_star)
                            results[hand][subject]['distance'] = np.sqrt(P_star**2 + S_star**2)
                    else:
                        results[hand][subject]['distance'] = (np.nan)

                    ax.legend(fontsize=8)
                except RuntimeError:
                    ax.text(0.05, 0.85, "Fit failed", transform=ax.transAxes, fontsize=8, color='red')
                    results[hand][subject]['distance'] = (np.nan)

        # Hide unused subplots
        for ax in axes[num_subjects:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return results

# Example usage
results = plot_mean_accuracy_vs_speed_all_subjects(mean_speed_accuracy_second_half, fit_hyperbolic=True)

def plot_spearman_correlation_histogram(results):
    """
    Plot a histogram of Spearman correlation values for each hand as separate subplots.

    Parameters:
        results (dict): Dictionary containing Spearman correlation results.
    """
    hands = ['left', 'right']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for ax, hand in zip(axes, hands):
        for subject in results[hand].keys():
            ax.hist(results[hand][subject]['spearman_corr'], bins=15, alpha=0.7, label=subject)
        ax.set_xlabel('Spearman Correlation')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of Spearman Correlation for {hand.capitalize()} Hand')
        ax.legend(title='Subjects')

    plt.tight_layout()
    plt.show()

plot_spearman_correlation_histogram(results)

























# # # -------------------------------------------------------------------------------------------------------------------

# Scatter plot for duration vs trial index for selected reach types and calculate speed (1/median duration)
def scatter_duration_vs_trial_with_speed_and_heatmap(all_combined_metrics, subject, hand, selected_reaches=None):
    """
    Scatter plot for duration vs trial index for selected reach types for a specific subject and hand.
    Calculate and highlight speed (1/median duration) for each reach type.
    Add a subplot to show the heatmap of the reach speed for each reach type.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        selected_reaches (list, optional): List of reach indices to plot. If None, plot all reach types.
    """
    reach_durations = all_combined_metrics[subject][hand]['durations']
    num_reaches = 16
    trials = list(reach_durations.keys())
    colors = sns.color_palette("Blues", num_reaches)  # Light blue to dark blue

    if selected_reaches is None:
        selected_reaches = range(num_reaches)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
    scatter_ax, heatmap_ax = axes
    speeds = {}

    for reach_index in selected_reaches:
        durations = []
        for trial in trials:
            trial_durations = np.array(reach_durations[trial])
            if reach_index < len(trial_durations):
                durations.append(trial_durations[reach_index])
            else:
                durations.append(np.nan)  # Handle missing data

        # Calculate speed (1/median duration) for the reach type
        valid_durations = np.array(durations)[~np.isnan(durations)]
        if len(valid_durations) > 0:
            median_duration = np.median(sorted(valid_durations))
            speed = 1 / median_duration
            speeds[reach_index] = speed
            scatter_ax.scatter(range(len(trials)), durations, label=f"Reach {reach_index + 1} (Speed: {speed:.2f})", color=colors[reach_index])
            scatter_ax.axhline(median_duration, color=colors[reach_index], linestyle='--', alpha=0.7, label=f"Median Duration (Reach {reach_index + 1})")
        else:
            speeds[reach_index] = np.nan

    # Heatmap of reach speeds
    speed_values = [speeds.get(reach_index, np.nan) for reach_index in range(num_reaches)]
    speed_matrix = np.array(speed_values).reshape(4, 4)  # Reshape into 4x4 grid
    sns.heatmap(speed_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=heatmap_ax)
    heatmap_ax.set_title("Reach Speed Heatmap")
    heatmap_ax.set_xlabel("Reach Index (Columns)")
    heatmap_ax.set_ylabel("Reach Index (Rows)")

    scatter_ax.set_title(f"Duration vs Trial Index for Selected Reach Types\nSubject: {subject}, Hand: {hand.capitalize()}")
    scatter_ax.set_xlabel("Trial Index")
    scatter_ax.set_ylabel("Duration")
    scatter_ax.legend(title="Reach Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    scatter_ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return speeds

# Example usage
speeds = scatter_duration_vs_trial_with_speed_and_heatmap(all_combined_metrics, '07/22/HW', 'right')
print("Speeds (1/median duration) for each reach type:", speeds)

# Scatter plot for distance vs trial index for selected reach types and calculate precision (1/median error distance)
def scatter_distance_vs_trial_with_precision_and_heatmap(all_combined_metrics, subject, hand, selected_reaches=None):
    """
    Scatter plot for distance vs trial index for selected reach types for a specific subject and hand.
    Calculate and highlight precision (1/median error distance) for each reach type.
    Add a subplot to show the heatmap of the reach precision for each reach type.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        selected_reaches (list, optional): List of reach indices to plot. If None, plot all reach types.
    """
    reach_distances = all_combined_metrics[subject][hand]['distance']
    num_reaches = 16
    trials = list(reach_distances.keys())
    colors = sns.color_palette("Blues", num_reaches)  # Light blue to dark blue

    if selected_reaches is None:
        selected_reaches = range(num_reaches)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
    scatter_ax, heatmap_ax = axes
    precisions = {}

    for reach_index in selected_reaches:
        distances = []
        for trial in trials:
            trial_distances = np.array(reach_distances[trial])
            if reach_index < len(trial_distances):
                distances.append(trial_distances[reach_index])
            else:
                distances.append(np.nan)  # Handle missing data

        # Calculate precision (1/median error distance) for the reach type
        valid_distances = np.array(distances)[~np.isnan(distances)]
        if len(valid_distances) > 0:
            median_distance = np.median(sorted(valid_distances))
            precision = 1 / median_distance
            precisions[reach_index] = precision
            scatter_ax.scatter(range(len(trials)), distances, label=f"Reach {reach_index + 1} (Precision: {precision:.2f})", color=colors[reach_index])
            scatter_ax.axhline(median_distance, color=colors[reach_index], linestyle='--', alpha=0.7, label=f"Median Distance (Reach {reach_index + 1})")
        else:
            precisions[reach_index] = np.nan

    # Heatmap of reach precisions
    precision_values = [precisions.get(reach_index, np.nan) for reach_index in range(num_reaches)]
    precision_matrix = np.array(precision_values).reshape(4, 4)  # Reshape into 4x4 grid
    sns.heatmap(precision_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=heatmap_ax)
    heatmap_ax.set_title("Reach Precision Heatmap")
    heatmap_ax.set_xlabel("Reach Index (Columns)")
    heatmap_ax.set_ylabel("Reach Index (Rows)")

    scatter_ax.set_title(f"Distance vs Trial Index for Selected Reach Types\nSubject: {subject}, Hand: {hand.capitalize()}")
    scatter_ax.set_xlabel("Trial Index")
    scatter_ax.set_ylabel("Distance")
    scatter_ax.legend(title="Reach Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    scatter_ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return precisions

# Example usage
precisions = scatter_distance_vs_trial_with_precision_and_heatmap(all_combined_metrics, '07/22/HW', 'right')
print("Precisions (1/median error distance) for each reach type:", precisions)

# Scatter plot for duration and distance vs trial index for selected reach types and calculate speed and precision
def scatter_duration_distance_vs_trial_with_heatmaps(all_combined_metrics, subject, hand, selected_reaches=None):
    """
    Scatter plot for duration and distance vs trial index for selected reach types for a specific subject and hand.
    Calculate and highlight speed (1/median duration) and precision (1/median error distance) for each reach type.
    Add subplots to show the heatmaps of the reach speed and precision for each reach type.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        selected_reaches (list, optional): List of reach indices to plot. If None, plot all reach types.
    """
    reach_durations = all_combined_metrics[subject][hand]['durations']
    reach_distances = all_combined_metrics[subject][hand]['distance']
    num_reaches = 16
    trials = list(reach_durations.keys())
    colors = sns.color_palette("Blues", num_reaches)  # Light blue to dark blue

    if selected_reaches is None:
        selected_reaches = range(num_reaches)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'width_ratios': [3, 1]})
    duration_ax, duration_heatmap_ax, distance_ax, distance_heatmap_ax = axes.flatten()
    speeds = {}
    precisions = {}

    for reach_index in selected_reaches:
        durations = []
        distances = []
        for trial in trials:
            trial_durations = np.array(reach_durations[trial])
            trial_distances = np.array(reach_distances[trial])
            if reach_index < len(trial_durations):
                durations.append(trial_durations[reach_index])
            else:
                durations.append(np.nan)  # Handle missing data
            if reach_index < len(trial_distances):
                distances.append(trial_distances[reach_index])
            else:
                distances.append(np.nan)  # Handle missing data

        # Calculate speed (1/median duration) for the reach type
        valid_durations = np.array(durations)[~np.isnan(durations)]
        if len(valid_durations) > 0:
            median_duration = np.median(sorted(valid_durations))
            speed = 1 / median_duration
            speeds[reach_index] = speed
            duration_ax.scatter(range(len(trials)), durations, label=f"Reach {reach_index + 1} (Speed: {speed:.2f})", color=colors[reach_index])
            duration_ax.axhline(median_duration, color=colors[reach_index], linestyle='--', alpha=0.7, label=f"Median Duration (Reach {reach_index + 1})")
        else:
            speeds[reach_index] = np.nan

        # Calculate precision (1/median error distance) for the reach type
        valid_distances = np.array(distances)[~np.isnan(distances)]
        if len(valid_distances) > 0:
            median_distance = np.median(sorted(valid_distances))
            precision = 1 / median_distance
            precisions[reach_index] = precision
            distance_ax.scatter(range(len(trials)), distances, label=f"Reach {reach_index + 1} (Precision: {precision:.2f})", color=colors[reach_index])
            distance_ax.axhline(median_distance, color=colors[reach_index], linestyle='--', alpha=0.7, label=f"Median Distance (Reach {reach_index + 1})")
        else:
            precisions[reach_index] = np.nan

    # Heatmap of reach speeds
    speed_values = [speeds.get(reach_index, np.nan) for reach_index in range(num_reaches)]
    speed_matrix = np.array(speed_values).reshape(4, 4)  # Reshape into 4x4 grid
    sns.heatmap(speed_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=duration_heatmap_ax)
    duration_heatmap_ax.set_title("Reach Speed Heatmap")
    duration_heatmap_ax.set_xlabel("Reach Index (Columns)")
    duration_heatmap_ax.set_ylabel("Reach Index (Rows)")

    # Heatmap of reach precisions
    precision_values = [precisions.get(reach_index, np.nan) for reach_index in range(num_reaches)]
    precision_matrix = np.array(precision_values).reshape(4, 4)  # Reshape into 4x4 grid
    sns.heatmap(precision_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=distance_heatmap_ax)
    distance_heatmap_ax.set_title("Reach Precision Heatmap")
    distance_heatmap_ax.set_xlabel("Reach Index (Columns)")
    distance_heatmap_ax.set_ylabel("Reach Index (Rows)")

    duration_ax.set_title(f"Duration vs Trial Index for Selected Reach Types\nSubject: {subject}, Hand: {hand.capitalize()}")
    duration_ax.set_xlabel("Trial Index")
    duration_ax.set_ylabel("Duration")
    # duration_ax.legend(title="Reach Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    duration_ax.grid(alpha=0.3)

    distance_ax.set_title(f"Distance vs Trial Index for Selected Reach Types\nSubject: {subject}, Hand: {hand.capitalize()}")
    distance_ax.set_xlabel("Trial Index")
    distance_ax.set_ylabel("Distance")
    # distance_ax.legend(title="Reach Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    distance_ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return speeds, precisions

# Example usage
speeds, precisions = scatter_duration_distance_vs_trial_with_heatmaps(all_combined_metrics, '07/22/HW', 'right')
print("Speeds (1/median duration) for each reach type:", speeds)
print("Precisions (1/median error distance) for each reach type:", precisions)

plt.scatter(list(precisions.values()), list(speeds.values()), c='blue', alpha=0.7, edgecolor='k')
plt.ylabel('Speeds (1/median duration)')
plt.xlabel('Precisions (1/median error distance)')
plt.title('Speeds vs Precisions')
plt.grid(True)
plt.tight_layout()
plt.show()
# # # -------------------------------------------------------------------------------------------------------------------


def calculate_mean_and_sd_values(all_combined_metrics):
    """
    Calculate mean and standard deviation for accuracy and speed for all subjects and hands.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.

    Returns:
        dict: Mean and standard deviation for accuracy and speed for each subject and hand.
    """
    mean_sd_values = {}

    for subject in all_combined_metrics.keys():
        mean_sd_values[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            acc_dict = all_combined_metrics[subject][hand]['accuracy']
            speed_dict = all_combined_metrics[subject][hand]['speed']
            
            # Flatten all arrays into one, ignoring NaNs
            all_acc_values = np.concatenate([v.flatten() for v in acc_dict.values()])
            all_speed_values = np.concatenate([v.flatten() for v in speed_dict.values()])
            
            mean_acc = np.nanmean(all_acc_values)  # nanmean ignores NaNs
            sd_acc = np.nanstd(all_acc_values)  # nanstd ignores NaNs
            
            mean_speed = np.nanmean(all_speed_values)  # nanmean ignores NaNs
            sd_speed = np.nanstd(all_speed_values)  # nanstd ignores NaNs
            
            mean_sd_values[subject][hand] = {
                'accuracy': {'mean': mean_acc, 'sd': sd_acc},
                'speed': {'mean': mean_speed, 'sd': sd_speed}
            }
    
    return mean_sd_values

mean_sd_values = calculate_mean_and_sd_values(all_combined_metrics)

def overlay_hyperbolic_curves_all_subjects(all_combined_metrics, mean_sd_values, subjects, hand, metric_x, metric_y, add_diagonal=False):
    """
    Overlay all 16 hyperbolic curves and intersection points for each subject as subplots, 
    each with its own x and y axis.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        mean_sd_values (dict): Mean and standard deviation values for each subject and hand.
        subjects (list): List of subject identifiers.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        add_diagonal (bool): Whether to add diagonal lines and calculate intersection points.
    """
    num_reaches = 16
    max_subjects_per_row = 4
    num_subjects = len(subjects)
    num_rows = (num_subjects + max_subjects_per_row - 1) // max_subjects_per_row

    fig, axes = plt.subplots(num_rows, max_subjects_per_row, figsize=(6 * max_subjects_per_row, 6 * num_rows))
    axes = axes.flatten()

    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        colors = sns.color_palette("Blues", num_reaches)  # Generate a color palette from light blue to dark blue

        min_intersection = None
        max_intersection = None

        for reach_index in range(num_reaches):
            x_values = []
            y_values = []

            trials = all_combined_metrics[subject][hand][metric_x].keys()

            for trial in trials:
                trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                # Collect data for the current reach index
                if reach_index < len(trial_x) and reach_index < len(trial_y):
                    x_values.append(trial_x[reach_index])
                    y_values.append(trial_y[reach_index])

            # Remove NaN values
            x_values = np.array(x_values)
            y_values = np.array(y_values)
            valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
            x_values = x_values[valid_indices]
            y_values = y_values[valid_indices]

            # Apply hyperbolic regression
            if len(x_values) > 1 and len(y_values) > 1:
                def hyperbolic_model(x, a, b):
                    return a + b / x

                try:
                    params, _ = curve_fit(hyperbolic_model, x_values, y_values, p0=(0, 1))
                    a, b = params
                    fit_x = np.linspace(min(x_values), max(x_values), 100)
                    fit_y = hyperbolic_model(fit_x, a, b)
                    ax.plot(fit_x, fit_y, color=colors[reach_index], label=f'Reach {reach_index + 1}')

                    if add_diagonal:
                        # Use mean and standard deviation values for diagonal line
                        x_mean = mean_sd_values[subject][hand]['accuracy']['mean']
                        x_sd = mean_sd_values[subject][hand]['accuracy']['sd']
                        y_mean = mean_sd_values[subject][hand]['speed']['mean']
                        y_sd = mean_sd_values[subject][hand]['speed']['sd']
                        m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
                        x_diag = np.linspace(min(x_values), max(x_values), 100)
                        y_diag = m * x_diag

                        # Solve intersection point
                        A, B, C = m, -a, -b
                        discriminant = B**2 - 4 * A * C
                        if discriminant >= 0:
                            sqrt_disc = np.sqrt(discriminant)
                            root1 = (-B + sqrt_disc) / (2 * A)
                            root2 = (-B - sqrt_disc) / (2 * A)
                            positive_roots = [r for r in (root1, root2) if r > 0]
                            P_star = max(positive_roots) if positive_roots else np.nan
                            S_star = m * P_star if np.isfinite(P_star) else np.nan

                            if np.isfinite(P_star) and np.isfinite(S_star):
                                ax.scatter(P_star, S_star, color=colors[reach_index], edgecolor='black', zorder=5)

                                # Update min and max intersection points
                                if min_intersection is None or P_star < min_intersection[0]:
                                    min_intersection = (P_star, S_star)
                                if max_intersection is None or P_star > max_intersection[0]:
                                    max_intersection = (P_star, S_star)

                except RuntimeError:
                    print(f"Hyperbolic regression failed for Reach {reach_index + 1}.")

        # Draw a line connecting the min and max intersection points
        if min_intersection and max_intersection:
            ax.plot(
                [min_intersection[0], max_intersection[0]],
                [min_intersection[1], max_intersection[1]],
                color='purple', linestyle='--', label='Diagonal Line'
            )

        ax.set_title(f"{subject}, {hand.capitalize()}", fontsize=12)
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for ax in axes[num_subjects:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

overlay_hyperbolic_curves_all_subjects(all_combined_metrics, mean_sd_values, All_dates, 'right', 'accuracy', 'speed', add_diagonal=True)


# # # -------------------------------------------------------------------------------------------------------------------

# # Function to calculate motor acuity from hyperbolic regression and return distance, r values, and Spearman correlation
def get_motor_acuity_from_hyperbolic(all_combined_metrics, subject, hand, metric_x, metric_y, reach_indices, add_diagonal=False):
    """
    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        reach_indices (int or list): Reach index or list of indices to include.
        add_diagonal (bool): Whether to calculate diagonal line and intersection.

    Returns:
        dict: Dictionary containing distances, r values, Spearman correlation, and Spearman p-value.
    """
    if isinstance(reach_indices, int):
        reach_indices = [reach_indices]  # Convert single index to list

    x_values = []
    y_values = []

    trials = all_combined_metrics[subject][hand][metric_x].keys()

    for trial in trials:
        trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
        trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

        for reach_index in reach_indices:
            if reach_index < len(trial_x) and reach_index < len(trial_y):
                x_values.append(trial_x[reach_index])
                y_values.append(trial_y[reach_index])

    # Convert to arrays and remove NaN
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]

    results = {"distance": None, "R^2": None, "spearman_r": None, "spearman_p": None, "params": None, "intersection": None}

    # Apply hyperbolic regression
    if len(x_values) > 1 and len(y_values) > 1:
        def hyperbolic_model(x, a, b):
            return a + b / x

        try:
            params, _ = curve_fit(hyperbolic_model, x_values, y_values, p0=(0, 1))
            a, b = params
            # results["params"] = {"a": a, "b": b}

            # Calculate r (correlation coefficient)
            residuals = y_values - hyperbolic_model(x_values, a, b)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_values - np.mean(y_values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            results["R^2"] = r_squared

            if add_diagonal:
                # Calculate diagonal slope
                x_mean, x_sd = np.mean(x_values), np.std(x_values)
                y_mean, y_sd = np.mean(y_values), np.std(y_values)
                m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)

                # Solve intersection between y = m*x and y = a + b/x
                A, B, C = m, -a, -b
                discriminant = B**2 - 4 * A * C
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    root1 = (-B + sqrt_disc) / (2 * A)
                    root2 = (-B - sqrt_disc) / (2 * A)
                    positive_roots = [r for r in (root1, root2) if r > 0]
                    P_star = max(positive_roots) if positive_roots else np.nan
                    S_star = m * P_star if np.isfinite(P_star) else np.nan

                    if np.isfinite(P_star) and np.isfinite(S_star):
                        # results["intersection"] = (P_star, S_star)
                        results["distance"] = np.sqrt(P_star**2 + S_star**2)

        except RuntimeError:
            print("Hyperbolic regression failed.")

    # Calculate Spearman correlation
    if len(x_values) > 1 and len(y_values) > 1:
        spearman_r, spearman_p = spearmanr(x_values, y_values)
        results["spearman_r"] = spearman_r
        results["spearman_p"] = spearman_p

    return results

def calculate_motor_acuity_and_correlation(all_combined_metrics):
    """
    Calculate motor acuity and correlation for each subject, hand, and reach index.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.

    Returns:
        dict: Motor acuity values for each subject, hand, and reach index.
        dict: Correlation values for each subject, hand, and reach index.
    """
    motor_acuity = {}
    correlation = {}

    for subject in all_combined_metrics.keys():
        motor_acuity[subject] = {}
        correlation[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            motor_acuity[subject][hand] = {}
            correlation[subject][hand] = {}
            for reach_indices in range(16):
                results = get_motor_acuity_from_hyperbolic(all_combined_metrics, subject, hand, 'accuracy', 'speed', [reach_indices], add_diagonal=True)
                motor_acuity[subject][hand][reach_indices] = results.get("distance", None)
                correlation[subject][hand][reach_indices] = results.get("spearman_r", None)

    return motor_acuity, correlation

# Calculate motor acuity and correlation for each subject, hand, and reach index across trials, per subject have 16 motor acuity values
motor_acuity, correlation = calculate_motor_acuity_and_correlation(all_combined_metrics)

# Calculate the average motor acuity and average correlation across reaches only when r > 0
def calculate_average_motor_acuity_and_correlation(motor_acuity, correlation):
    """
    Calculate the average motor acuity and average correlation across reaches for each subject and hand, 
    considering only reaches where the correlation (r) is greater than 0.

    Parameters:
        motor_acuity (dict): Motor acuity values for each subject, hand, and reach index.
        correlation (dict): Correlation values (r) for each subject, hand, and reach index.

    Returns:
        dict: Average motor acuity and correlation for each subject and hand.
    """
    results = {}

    for subject in motor_acuity.keys():
        results[subject] = {}
        for hand in motor_acuity[subject].keys():
            acuity_values = []
            correlation_values = []
            for reach_index in motor_acuity[subject][hand].keys():
                if correlation[subject][hand][reach_index] > 0:
                    acuity_values.append(motor_acuity[subject][hand][reach_index])
                    correlation_values.append(correlation[subject][hand][reach_index])
            # Calculate the averages only if there are valid values
            results[subject][hand] = {
                "average_motor_acuity(corr_>0)": np.mean(acuity_values) if acuity_values else np.nan,
                "average_correlation_>0": np.mean(correlation_values) if correlation_values else np.nan,
            }

    return results
average_results = calculate_average_motor_acuity_and_correlation(motor_acuity, correlation)

# Plot the average motor acuity and correlation values in two subplots (top and bottom)
def plot_average_motor_acuity_and_correlation(average_results):
    """
    Plot the average motor acuity and correlation values for each subject and hand in two subplots.

    Parameters:
        average_results (dict): Average motor acuity and correlation values for each subject and hand.
    """
    subjects = list(average_results.keys())
    hands = ['right', 'left']
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    bar_width = 0.35
    index = np.arange(len(subjects))

    # Top subplot: Motor Acuity
    for i, hand in enumerate(hands):
        acuity_values = [average_results[subject][hand]["average_motor_acuity(corr_>0)"] for subject in subjects]
        axes[0].bar(index + i * bar_width, acuity_values, bar_width, label=f"{hand.capitalize()} Motor Acuity", alpha=0.7)

    axes[0].set_ylabel("Motor Acuity", fontsize=12)
    axes[0].set_title("Average Motor Acuity by Subject and Hand", fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bottom subplot: Correlation
    for i, hand in enumerate(hands):
        correlation_values = [average_results[subject][hand]["average_correlation_>0"] for subject in subjects]
        axes[1].bar(index + i * bar_width, correlation_values, bar_width, label=f"{hand.capitalize()} Correlation", alpha=0.7)

    axes[1].set_xlabel("Subjects", fontsize=12)
    axes[1].set_ylabel("Correlation", fontsize=12)
    axes[1].set_title("Average Correlation by Subject and Hand", fontsize=14)
    axes[1].set_xticks(index + bar_width / 2)
    axes[1].set_xticklabels(subjects, fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
plot_average_motor_acuity_and_correlation(average_results)


# Scatter plot for selected metrics with options for Spearman correlation, linear regression, or hyperbolic regression
def scatter_plot_with_options_per_reach(all_combined_metrics, subject, hand, metric_x, metric_y, add_diagonal=False):
    """
    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        add_diagonal (bool): Whether to add diagonal lines and calculate intersection points.

    Returns:
        dict: Dictionary containing distances and r values for each reach.
    """
    num_reaches = 16
    results = {}

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for reach_index in range(num_reaches):
        ax = axes[reach_index]
        x_values = []
        y_values = []
        trial_colors = []

        trials = all_combined_metrics[subject][hand][metric_x].keys()
        color_palette = sns.color_palette("Reds", len(trials))  # Generate a color palette from light to dark

        for i, trial in enumerate(trials):
            trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

            # Collect data for the current reach index
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

        ax.scatter(x_values, y_values, c=trial_colors, alpha=0.7, label=f"Reach {reach_index + 1}")

        reach_results = {"distance": None, "r": None}

        # Apply hyperbolic regression
        if len(x_values) > 1 and len(y_values) > 1:
            def hyperbolic_model(x, a, b):
                return a + b / x

            try:
                params, _ = curve_fit(hyperbolic_model, x_values, y_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(x_values), max(x_values), 100)
                fit_y = hyperbolic_model(fit_x, a, b)
                ax.plot(fit_x, fit_y, color='green', linestyle='--', label=f'Hyperbolic Fit (a={a:.2f}, b={b:.2f})')

                # Calculate r (correlation coefficient)
                residuals = y_values - hyperbolic_model(x_values, a, b)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_values - np.mean(y_values))**2)
                r_squared = 1 - (ss_res / ss_tot)
                r = np.sqrt(r_squared) if b > 0 else -np.sqrt(r_squared)
                reach_results["r"] = r
                ax.text(0.05, 0.95, f"r: {r:.2f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

                if add_diagonal:
                    # Calculate diagonal line
                    x_mean, x_sd = np.mean(x_values), np.std(x_values)
                    y_mean, y_sd = np.mean(y_values), np.std(y_values)
                    m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
                    x_diag = np.linspace(min(x_values), max(x_values), 100)
                    y_diag = m * x_diag
                    ax.plot(x_diag, y_diag, color='blue', linestyle='--', label='Diagonal Line')

                    # Calculate intersection point
                    A, B, C = m, -a, -b
                    discriminant = B**2 - 4 * A * C
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        root1 = (-B + sqrt_disc) / (2 * A)
                        root2 = (-B - sqrt_disc) / (2 * A)
                        positive_roots = [r for r in (root1, root2) if r > 0]
                        P_star = max(positive_roots) if positive_roots else np.nan
                        S_star = m * P_star if np.isfinite(P_star) else np.nan

                        if np.isfinite(P_star) and np.isfinite(S_star):
                            ax.scatter(P_star, S_star, color='purple', label='Intersection Point', zorder=5)
                            ax.text(P_star, S_star, f"({P_star:.2f}, {S_star:.2f})", color='purple', fontsize=8)

                            # Calculate distance from (0, 0) to the intersection point
                            distance = np.sqrt(P_star**2 + S_star**2)
                            reach_results["distance"] = distance
                            ax.text(P_star, S_star - 0.1, f"Dist: {distance:.2f}", color='purple', fontsize=8)
            except RuntimeError:
                print(f"Hyperbolic regression failed for Reach {reach_index + 1}.")

        ax.set_title(f"Reach {reach_index + 1}")
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())
        # ax.legend()
        results[reach_index] = reach_results

    fig.suptitle(f"{metric_x.capitalize()} vs {metric_y.capitalize()} for Subject: {subject}, Hand: {hand.capitalize()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return results

results = scatter_plot_with_options_per_reach(all_combined_metrics, '07/22/HW', 'right', 'accuracy', 'speed', add_diagonal=True)




# Scatter plot for selected metrics with options for Spearman correlation, linear regression, or hyperbolic regression
def scatter_plot_with_options(all_combined_metrics, subject, hand, metric_x, metric_y, reach_indices, add_diagonal=False):
    """
    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        reach_indices (int or list): Reach index or list of indices to include.
        add_diagonal (bool): Whether to add diagonal lines and calculate intersection points.

    Returns:
        dict: Dictionary containing distances and r values.
    """
    if isinstance(reach_indices, int):
        reach_indices = [reach_indices]  # Convert single index to list

    x_values = []
    y_values = []
    trial_colors = []

    trials = all_combined_metrics[subject][hand][metric_x].keys()
    color_palette = sns.color_palette("Reds", len(trials))  # Generate a color palette from light to dark

    for i, trial in enumerate(trials):
        trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
        trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

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

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c=trial_colors, alpha=0.7, label=f"{reach_indices}")

    results = {"distance": None, "r": None}

    # Apply hyperbolic regression
    if len(x_values) > 1 and len(y_values) > 1:
        def hyperbolic_model(x, a, b):
            return a + b / x

        try:
            params, _ = curve_fit(hyperbolic_model, x_values, y_values, p0=(0, 1))
            a, b = params
            fit_x = np.linspace(min(x_values), max(x_values), 100)
            fit_y = hyperbolic_model(fit_x, a, b)
            plt.plot(fit_x, fit_y, color='green', linestyle='--', label=f'Hyperbolic Fit (a={a:.2f}, b={b:.2f})')

            # Calculate r (correlation coefficient)
            residuals = y_values - hyperbolic_model(x_values, a, b)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_values - np.mean(y_values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            r = np.sqrt(r_squared) if b > 0 else -np.sqrt(r_squared)
            results["r"] = r
            plt.text(0.05, 0.95, f"r: {r:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            if add_diagonal:
                # Calculate diagonal line
                x_mean, x_sd = np.mean(x_values), np.std(x_values)
                y_mean, y_sd = np.mean(y_values), np.std(y_values)
                m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
                x_diag = np.linspace(min(x_values), max(x_values), 100)
                y_diag = m * x_diag
                plt.plot(x_diag, y_diag, color='blue', linestyle='--', label='Diagonal Line')

                # Calculate intersection point
                A, B, C = m, -a, -b
                discriminant = B**2 - 4 * A * C
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    root1 = (-B + sqrt_disc) / (2 * A)
                    root2 = (-B - sqrt_disc) / (2 * A)
                    positive_roots = [r for r in (root1, root2) if r > 0]
                    P_star = max(positive_roots) if positive_roots else np.nan
                    S_star = m * P_star if np.isfinite(P_star) else np.nan

                    if np.isfinite(P_star) and np.isfinite(S_star):
                        plt.scatter(P_star, S_star, color='purple', label='Intersection Point', zorder=5)
                        plt.text(P_star, S_star, f"({P_star:.2f}, {S_star:.2f})", color='purple', fontsize=8)

                        # Calculate distance from (0, 0) to the intersection point
                        distance = np.sqrt(P_star**2 + S_star**2)
                        results["distance"] = distance
                        plt.text(P_star, S_star - 0.1, f"Dist: {distance:.2f}", color='purple', fontsize=8)
        except RuntimeError:
            print("Hyperbolic regression failed.")

    plt.xlabel(metric_x.capitalize())
    plt.ylabel(metric_y.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

results = scatter_plot_with_options(all_combined_metrics, '07/22/HW', 'right', 'accuracy', 'speed', [0], add_diagonal=True)





# Histogram of correlation across reaches for each subject and hand, with separate subplots for right and left hands
def plot_correlation_histogram_overlay(correlation):
    """
    Plot a histogram of correlation values across all reaches for each subject, with separate subplots for right and left hands.
    Also count the number of correlation values below 0 and above 0.

    Parameters:
        correlation (dict): Correlation values for each subject, hand, and reach index.
    """
    color_palette = sns.color_palette("husl", len(correlation.keys()))  # Generate colors for all subjects
    hands = ['right', 'left']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for hand_idx, hand in enumerate(hands):
        ax = axes[hand_idx]
        below_zero_count = 0
        above_zero_count = 0

        for idx, subject in enumerate(correlation.keys()):
            all_correlation_values = []

            if hand in correlation[subject]:
                for reach_index in correlation[subject][hand].keys():
                    r_value = correlation[subject][hand][reach_index]
                    if r_value is not None:  # Exclude None values
                        all_correlation_values.append(r_value)
                        if r_value < 0:
                            below_zero_count += 1
                        elif r_value > 0:
                            above_zero_count += 1

                # Plot histogram for the current subject and hand
                ax.hist(
                    all_correlation_values, bins=20, alpha=0.5, edgecolor='black',
                    color=color_palette[idx], label=f"{subject}"
                )

        ax.set_title(f"Histogram of Correlation Values ({hand.capitalize()} Hand)", fontsize=14)
        ax.set_xlabel("Correlation (r)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(alpha=0.3)

        # Display counts below and above 0
        ax.text(0.05, 0.95, f"Below 0: {below_zero_count}\nAbove 0: {above_zero_count}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='blue')

    plt.tight_layout()
    plt.show()

# Example usage
plot_correlation_histogram_overlay(correlation)

# Histogram of correlation across reaches for each subject and hand, with separate subplots for each subject and hand
def plot_correlation_histogram_subplots(correlation):
    """
    Plot a histogram of correlation values across all reaches for each subject and hand, with separate subplots for each subject and hand.

    Parameters:
        correlation (dict): Correlation values for each subject, hand, and reach index.
    """
    subjects = list(correlation.keys())
    hands = ['right', 'left']
    num_subjects = len(subjects)
    num_hands = len(hands)

    fig, axes = plt.subplots(num_subjects, num_hands, figsize=(12, 6 * num_subjects), sharey=True, squeeze=False)

    for row, subject in enumerate(subjects):
        for col, hand in enumerate(hands):
            ax = axes[row, col]
            below_zero_count = 0
            above_zero_count = 0
            all_correlation_values = []

            if hand in correlation[subject]:
                for reach_index in correlation[subject][hand].keys():
                    r_value = correlation[subject][hand][reach_index]
                    if r_value is not None:  # Exclude None values
                        all_correlation_values.append(r_value)
                        if r_value < 0:
                            below_zero_count += 1
                        elif r_value > 0:
                            above_zero_count += 1

                # Plot histogram for the current subject and hand
                ax.hist(
                    all_correlation_values, bins=20, alpha=0.5, edgecolor='black',
                    color='skyblue', label=f"{hand.capitalize()} Hand"
                )

            ax.set_title(f"Subject: {subject}, Hand: {hand.capitalize()}", fontsize=14)
            ax.set_xlabel("Correlation (r)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(alpha=0.3)

            # Display counts below and above 0
            ax.text(0.05, 0.95, f"Below 0: {below_zero_count}\nAbove 0: {above_zero_count}",
                    transform=ax.transAxes, fontsize=12, verticalalignment='top', color='blue')

    plt.tight_layout()
    plt.show()

# Example usage
plot_correlation_histogram_subplots(correlation)

# # # -------------------------------------------------------------------------------------------------------------------



# # # -------------------------------------------------------------------------------------------------------------------

# Calculate the median of speed and accuracy for each subject, each hand, and each reach type
def calculate_median_speed_accuracy(all_combined_metrics, selected_trials=None):
    """
    Calculate the median speed and accuracy for each subject, hand, and reach type.
    Optionally, calculate the median only for selected trials.

    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        selected_trials (list, optional): List of trial indices to include. If None, include all trials.

    Returns:
        dict: Median speed and accuracy values for each subject, hand, and reach type.
    """
    median_results = {}
    for subject in all_combined_metrics.keys():
        median_results[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            median_results[subject][hand] = {}
            for metric in ['speed', 'accuracy']:
                median_results[subject][hand][metric] = {}
                for reach_index in range(16):  # Assuming there are 16 reach types
                    values = []

                    for trial_index, trial in enumerate(all_combined_metrics[subject][hand][metric].keys()):
                        # Include only selected trials if specified
                        if selected_trials is not None and trial_index not in selected_trials:
                            continue

                        trial_values = np.array(all_combined_metrics[subject][hand][metric][trial])

                        # Ensure the reach index is valid
                        if reach_index < len(trial_values):
                            values.append(trial_values[reach_index])

                    # Remove NaN values and calculate the median
                    values = np.array(values)
                    valid_values = values[~np.isnan(values)]
                    median_results[subject][hand][metric][reach_index] = np.median(valid_values) if len(valid_values) > 0 else np.nan

    return median_results

# Example usage
# Select all trials
median_speed_accuracy_all = calculate_median_speed_accuracy(all_combined_metrics)

# Select first half of trials
selected_trials_first_half = list(range(0, 16))
median_speed_accuracy_first_half = calculate_median_speed_accuracy(all_combined_metrics, selected_trials=selected_trials_first_half)

# Select second half of trials
selected_trials_second_half = list(range(17, 32))
median_speed_accuracy_second_half = calculate_median_speed_accuracy(all_combined_metrics, selected_trials=selected_trials_second_half)

def plot_median_accuracy_vs_speed_multiple_subjects(median_speed_accuracy, subjects, hand, fit_hyperbolic=False):
    """
    Scatter plot for median accuracy vs median speed for the specified subjects and hand, with a maximum of 5 subjects per row.
    Optionally fits a hyperbolic regression curve.

    Parameters:
        median_speed_accuracy (dict): Dictionary containing median speed and accuracy values.
        subjects (list): List of subject identifiers.
        hand (str): Hand ('left' or 'right').
        fit_hyperbolic (bool): Whether to fit and plot a hyperbolic regression curve.
    """
    max_subjects_per_row = 5
    num_subjects = len(subjects)
    num_rows = (num_subjects + max_subjects_per_row - 1) // max_subjects_per_row

    fig, axes = plt.subplots(num_rows, min(num_subjects, max_subjects_per_row), figsize=(6 * min(num_subjects, max_subjects_per_row), 6 * num_rows), sharey=True)
    axes = axes.flatten() if num_subjects > 1 else [axes]

    for i, (ax, subject) in enumerate(zip(axes, subjects)):
        accuracy_values = list(median_speed_accuracy[subject][hand]['accuracy'].values())
        speed_values = list(median_speed_accuracy[subject][hand]['speed'].values())

        ax.scatter(accuracy_values, speed_values, color='blue', alpha=0.7)
        ax.set_title(f"Subject: {subject}, Hand: {hand.capitalize()}", fontsize=12)
        ax.set_xlabel("Median Accuracy", fontsize=10)
        ax.set_ylabel("Median Speed", fontsize=10)
        ax.grid(alpha=0.3)

        # Annotate each point with its reach index
        for j, (accuracy, speed) in enumerate(zip(accuracy_values, speed_values)):
            ax.text(accuracy, speed, str(j), fontsize=8, ha='right', va='bottom')

        # Fit and plot hyperbolic regression if enabled
        if fit_hyperbolic and len(accuracy_values) > 1 and len(speed_values) > 1:
            def hyperbolic_model(x, a, b):
                return a + b / x

            try:
                params, _ = curve_fit(hyperbolic_model, accuracy_values, speed_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(accuracy_values), max(accuracy_values), 100)
                fit_y = hyperbolic_model(fit_x, a, b)
                ax.plot(fit_x, fit_y, color='red', linestyle='--', label='Hyperbolic Fit')
                ax.legend(fontsize=8)
            except RuntimeError:
                ax.text(0.05, 0.85, "Fit failed", transform=ax.transAxes, fontsize=8, color='red')

    # Hide unused subplots
    for ax in axes[num_subjects:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plot_median_accuracy_vs_speed_multiple_subjects(median_speed_accuracy_all, ['07/22/HW'], 'right', fit_hyperbolic=True)
plot_median_accuracy_vs_speed_multiple_subjects(median_speed_accuracy_first_half, ['07/22/HW'], 'right', fit_hyperbolic=True)
plot_median_accuracy_vs_speed_multiple_subjects(median_speed_accuracy_second_half, ['07/22/HW'], 'right', fit_hyperbolic=True)


# # -------------------------------------------------------------------------------------------------------------------

# Calculate z-scores for all_combined_metrics and save as all_combined_metrics_z
# Z-score per trial
def calculate_z_scores(all_combined_metrics):
    all_combined_metrics_z = {}
    for subject in all_combined_metrics.keys():
        all_combined_metrics_z[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            all_combined_metrics_z[subject][hand] = {}
            for metric in all_combined_metrics[subject][hand].keys():
                metric_data = all_combined_metrics[subject][hand][metric]
                z_scored_data = {}
                for trial, values in metric_data.items():
                    values = np.array(values)
                    if len(values) > 1:  # Ensure there are enough values to calculate z-scores
                        z_scores = (values - np.nanmean(values)) / np.nanstd(values)
                    else:
                        z_scores = np.full_like(values, np.nan)
                    z_scored_data[trial] = z_scores
                all_combined_metrics_z[subject][hand][metric] = z_scored_data
    return all_combined_metrics_z

# Generate z-scored metrics
all_combined_metrics_z = calculate_z_scores(all_combined_metrics)

# Scatter plot for selected metrics with options for Spearman correlation, linear regression, or hyperbolic regression
def scatter_plot_with_options(all_combined_metrics, subject, hand, metric_x, metric_y, reach_indices, plot_type='scatter', add_diagonal=False):
    """
    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        reach_indices (int or list): Reach index or list of indices to include.
        plot_type (str): Type of plot ('scatter', 'spearman', 'linear_regression', 'hyperbolic_regression').
        add_diagonal (bool): Whether to add diagonal lines and calculate intersection points.
    """
    if isinstance(reach_indices, int):
        reach_indices = [reach_indices]  # Convert single index to list

    x_values = []
    y_values = []
    trial_colors = []

    trials = all_combined_metrics[subject][hand][metric_x].keys()
    color_palette = sns.color_palette("Reds", len(trials))  # Generate a color palette from light to dark

    for i, trial in enumerate(trials):
        trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
        trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

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

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c=trial_colors, alpha=0.7, label=f"{reach_indices}")

    if plot_type == 'spearman':
        # Calculate and display Spearman correlation
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
            plt.title(f"Spearman r: {correlation:.2f}, p: {p_value:.4f}")
        else:
            plt.title("Insufficient data for Spearman correlation")

    elif plot_type == 'linear_regression':
        # Apply linear regression
        if len(x_values) > 1 and len(y_values) > 1:
            x_values_reshaped = x_values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x_values_reshaped, y_values)
            y_pred = model.predict(x_values_reshaped)

            # Plot regression line
            plt.plot(x_values, y_pred, color='red', label=f'Linear Fit (R²={model.score(x_values_reshaped, y_values):.2f})')
            plt.title(f"{metric_y.capitalize()} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {metric_x.capitalize()}")
        else:
            plt.title("Insufficient data for Linear Regression")

    elif plot_type == 'hyperbolic_regression':
        # Apply hyperbolic regression
        if len(x_values) > 1 and len(y_values) > 1:
            def hyperbolic_model(x, a, b):
                return a + b / x

            try:
                params, _ = curve_fit(hyperbolic_model, x_values, y_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(x_values), max(x_values), 100)
                fit_y = hyperbolic_model(fit_x, a, b)
                plt.plot(fit_x, fit_y, color='green', linestyle='--', label=f'Hyperbolic Fit (a={a:.2f}, b={b:.2f})')
                plt.title(f"{metric_y.capitalize()} vs {metric_x.capitalize()} with Hyperbolic Regression")

                if add_diagonal:
                    # Calculate diagonal line
                    x_mean, x_sd = np.mean(x_values), np.std(x_values)
                    y_mean, y_sd = np.mean(y_values), np.std(y_values)
                    m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
                    x_diag = np.linspace(min(x_values), max(x_values), 100)
                    y_diag = m * x_diag
                    plt.plot(x_diag, y_diag, color='blue', linestyle='--', label='Diagonal Line')

                    # Calculate intersection point
                    A, B, C = m, -a, -b
                    discriminant = B**2 - 4 * A * C
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        root1 = (-B + sqrt_disc) / (2 * A)
                        root2 = (-B - sqrt_disc) / (2 * A)
                        positive_roots = [r for r in (root1, root2) if r > 0]
                        P_star = max(positive_roots) if positive_roots else np.nan
                        S_star = m * P_star if np.isfinite(P_star) else np.nan

                        if np.isfinite(P_star) and np.isfinite(S_star):
                            plt.scatter(P_star, S_star, color='purple', label='Intersection Point', zorder=5)
                            plt.text(P_star, S_star, f"({P_star:.2f}, {S_star:.2f})", color='purple', fontsize=8)

                            # Calculate distance from (0, 0) to the intersection point
                            distance = np.sqrt(P_star**2 + S_star**2)
                            plt.text(P_star, S_star - 0.1, f"Dist: {distance:.2f}", color='purple', fontsize=8)
            except RuntimeError:
                plt.title("Hyperbolic regression failed")
        else:
            plt.title("Insufficient data for Hyperbolic Regression")

    else:
        plt.title(f"{metric_y.capitalize()} vs {metric_x.capitalize()} Scatter Plot")

    plt.xlabel(metric_x.capitalize())
    plt.ylabel(metric_y.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
scatter_plot_with_options(all_combined_metrics, '07/22/HW', 'right', 'distance', 'speed', [0], plot_type='scatter')  # Scatter plot / list(range(16))
scatter_plot_with_options(all_combined_metrics, '07/22/HW', 'right', 'distance', 'speed', [0], plot_type='spearman')  # Spearman correlation
scatter_plot_with_options(all_combined_metrics, '07/22/HW', 'right', 'distance', 'speed', [0], plot_type='linear_regression')  # Linear regression
scatter_plot_with_options(all_combined_metrics, '07/22/HW', 'right', 'accuracy', 'speed', [1], plot_type='hyperbolic_regression', add_diagonal=True)  # Hyperbolic regression with diagonal

# Scatter plot for selected metrics with options for Spearman correlation, linear regression, or hyperbolic regression
def scatter_plot_with_options_subplots(all_combined_metrics, subject, hand, metric_x, metric_y, reach_indices):
    """
    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        reach_indices (int or list): Reach index or list of indices to include.
    Returns:
        dict: Distance from (0.0, 0.0) to the intersection point for each subplot.
    """
    if isinstance(reach_indices, int):
        reach_indices = [reach_indices]  # Convert single index to list

    x_values = []
    y_values = []
    trial_colors = []

    trials = all_combined_metrics[subject][hand][metric_x].keys()
    color_palette = sns.color_palette("Reds", len(trials))  # Generate a color palette from light to dark

    for i, trial in enumerate(trials):
        trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
        trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    distances = {}

    # Scatter plot
    axes[0].scatter(x_values, y_values, c=trial_colors, alpha=0.7, label=f"Reach {reach_index + 1}")
    axes[0].set_title(f"{metric_y.capitalize()} vs {metric_x.capitalize()} Scatter Plot")
    axes[0].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[0].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[0].legend()

    # Spearman correlation
    if len(x_values) > 1 and len(y_values) > 1:
        correlation, p_value = spearmanr(x_values, y_values)
        axes[1].scatter(x_values, y_values, c=trial_colors, alpha=0.7, label=f"Reach {reach_index + 1}")
        axes[1].set_title(f"Spearman r: {correlation:.2f}, p: {p_value:.4f}")
    else:
        axes[1].set_title("Insufficient data for Spearman correlation")
    axes[1].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[1].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[1].legend()

    # Linear regression
    if len(x_values) > 1 and len(y_values) > 1:
        x_values_reshaped = x_values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_values_reshaped, y_values)
        y_pred = model.predict(x_values_reshaped)
        axes[2].scatter(x_values, y_values, c=trial_colors, alpha=0.7, label=f"Reach {reach_index + 1}")
        axes[2].plot(x_values, y_pred, color='red', label=f'Linear Fit (R²={model.score(x_values_reshaped, y_values):.2f})')
        axes[2].set_title(f"{metric_y.capitalize()} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {metric_x.capitalize()}")
    else:
        axes[2].set_title("Insufficient data for Linear Regression")
    axes[2].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[2].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[2].legend()

    # Hyperbolic regression with diagonal
    if len(x_values) > 1 and len(y_values) > 1:
        def hyperbolic_model(x, a, b):
            return a + b / x

        try:
            params, _ = curve_fit(hyperbolic_model, 1/x_values, y_values, p0=(0, 1))
            a, b = params
            fit_x = np.linspace(min(1/x_values), max(1/x_values), 100)
            fit_y = hyperbolic_model(fit_x, a, b)
            axes[3].scatter(1/x_values, y_values, c=trial_colors, alpha=0.7, label=f"Reach {reach_index + 1}")
            axes[3].plot(fit_x, fit_y, color='green', linestyle='--', label=f'Hyperbolic Fit (a={a:.2f}, b={b:.2f})')

            # Calculate diagonal line
            x_mean, x_sd = np.mean(1/x_values), np.std(1/x_values)
            y_mean, y_sd = np.mean(y_values), np.std(y_values)
            m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
            x_diag = np.linspace(min(1/x_values), max(1/x_values), 100)
            y_diag = m * x_diag
            axes[3].plot(x_diag, y_diag, color='blue', linestyle='--', label='Diagonal Line')

            # Calculate intersection point
            A, B, C = m, -a, -b
            discriminant = B**2 - 4 * A * C
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                root1 = (-B + sqrt_disc) / (2 * A)
                root2 = (-B - sqrt_disc) / (2 * A)
                positive_roots = [r for r in (root1, root2) if r > 0]
                P_star = max(positive_roots) if positive_roots else np.nan
                S_star = m * P_star if np.isfinite(P_star) else np.nan

                if np.isfinite(P_star) and np.isfinite(S_star):
                    axes[3].scatter(P_star, S_star, color='purple', label='Intersection Point', zorder=5)
                    axes[3].text(P_star, S_star, f"({P_star:.2f}, {S_star:.2f})", color='purple', fontsize=8)

                    # Calculate distance from (0, 0) to the intersection point
                    distance = np.sqrt(P_star**2 + S_star**2)
                    distances['hyperbolic'] = distance
                    axes[3].text(P_star, S_star - 0.1, f"Dist: {distance:.2f}", color='purple', fontsize=8)
        except RuntimeError:
            axes[3].set_title("Hyperbolic regression failed")
    else:
        axes[3].set_title("Insufficient data for Hyperbolic Regression")
    axes[3].set_xlabel("Accuracy (Bad → Good)")
    axes[3].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    return distances

# Example usage
distances = scatter_plot_with_options_subplots(all_combined_metrics, '07/22/HW', 'right', 'distance', 'speed', 1)


def scatter_plot_with_linear_regression_for_reaches_subplots(all_combined_metrics, subject, hand, metric_x, metric_y):
    num_reaches = 16
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharey=True)
    axes = axes.flatten()

    for reach_index, ax in enumerate(axes):
        x_values = []
        y_values = []

        trials = all_combined_metrics[subject][hand][metric_x].keys()
        for trial in trials:
            trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

            if reach_index < len(trial_x) and reach_index < len(trial_y):
                x_values.append(trial_x[reach_index])
                y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        ax.scatter(x_values, y_values, alpha=0.7, label='Data Points')

        # Apply linear regression
        if len(x_values) > 1 and len(y_values) > 1:
            x_values_reshaped = x_values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x_values_reshaped, y_values)
            y_pred = model.predict(x_values_reshaped)

            # Plot regression line
            ax.plot(x_values, y_pred, color='red', label=f'Linear Fit (R²={model.score(x_values_reshaped, y_values):.2f})')

        ax.set_title(f"Reach {reach_index + 1}")
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())
        ax.legend()

    plt.tight_layout()
    plt.show()

# Example usage
scatter_plot_with_linear_regression_for_reaches_subplots(all_combined_metrics_z, '07/22/HW', 'right', 'distance', 'speed')

# # -------------------------------------------------------------------------------------------------------------------


# Scatter plot for selected metrics with options for Spearman correlation, linear regression, or hyperbolic regression
def scatter_plot_with_options_subplots(all_combined_metrics, subjects, hand, metric_x, metric_y, trial_index, reach_index):
    """
    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subjects (list): List of subject identifiers.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        trial_index (int): Trial index to include.
        reach_index (int): Reach index to include.
    Returns:
        dict: Distance from (0.0, 0.0) to the intersection point for each subplot.
    """
    x_values = []
    y_values = []

    for subject in subjects:
        if subject not in all_combined_metrics or hand not in all_combined_metrics[subject]:
            continue

        trials = all_combined_metrics[subject][hand][metric_x].keys()
        trial_list = list(trials)

        if trial_index < len(trial_list):
            trial = trial_list[trial_index]
            trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    distances = {}

    # Scatter plot
    axes[0].scatter(x_values, y_values, color='red', alpha=0.7, label=f"Trial {trial_index + 1}, Reach {reach_index + 1}")
    axes[0].set_title(f"{metric_y.capitalize()} vs {metric_x.capitalize()} Scatter Plot")
    axes[0].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[0].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[0].legend()

    # Spearman correlation
    if len(x_values) > 1 and len(y_values) > 1:
        correlation, p_value = spearmanr(x_values, y_values)
        axes[1].scatter(x_values, y_values, color='red', alpha=0.7, label=f"Trial {trial_index + 1}, Reach {reach_index + 1}")
        axes[1].set_title(f"Spearman r: {correlation:.2f}, p: {p_value:.4f}")
    else:
        axes[1].set_title("Insufficient data for Spearman correlation")
    axes[1].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[1].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[1].legend()

    # Linear regression
    if len(x_values) > 1 and len(y_values) > 1:
        x_values_reshaped = x_values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_values_reshaped, y_values)
        y_pred = model.predict(x_values_reshaped)
        axes[2].scatter(x_values, y_values, color='red', alpha=0.7, label=f"Trial {trial_index + 1}, Reach {reach_index + 1}")
        axes[2].plot(x_values, y_pred, color='blue', label=f'Linear Fit (R²={model.score(x_values_reshaped, y_values):.2f})')
        axes[2].set_title(f"{metric_y.capitalize()} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {metric_x.capitalize()}")
    else:
        axes[2].set_title("Insufficient data for Linear Regression")
    axes[2].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[2].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[2].legend()

    # Hyperbolic regression with diagonal
    if len(x_values) > 1 and len(y_values) > 1:
        def hyperbolic_model(x, a, b):
            return a + b / x

        try:
            params, _ = curve_fit(hyperbolic_model, 1/x_values, y_values, p0=(0, 1))
            a, b = params
            fit_x = np.linspace(min(1/x_values), max(1/x_values), 100)
            fit_y = hyperbolic_model(fit_x, a, b)
            axes[3].scatter(1/x_values, y_values, color='red', alpha=0.7, label=f"Trial {trial_index + 1}, Reach {reach_index + 1}")
            axes[3].plot(fit_x, fit_y, color='green', linestyle='--', label=f'Hyperbolic Fit (a={a:.2f}, b={b:.2f})')

            # Calculate diagonal line
            x_mean, x_sd = np.mean(1/x_values), np.std(1/x_values)
            y_mean, y_sd = np.mean(y_values), np.std(y_values)
            m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
            x_diag = np.linspace(min(1/x_values), max(1/x_values), 100)
            y_diag = m * x_diag
            axes[3].plot(x_diag, y_diag, color='blue', linestyle='--', label='Diagonal Line')

            # Calculate intersection point
            A, B, C = m, -a, -b
            discriminant = B**2 - 4 * A * C
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                root1 = (-B + sqrt_disc) / (2 * A)
                root2 = (-B - sqrt_disc) / (2 * A)
                positive_roots = [r for r in (root1, root2) if r > 0]
                P_star = max(positive_roots) if positive_roots else np.nan
                S_star = m * P_star if np.isfinite(P_star) else np.nan

                if np.isfinite(P_star) and np.isfinite(S_star):
                    axes[3].scatter(P_star, S_star, color='purple', label='Intersection Point', zorder=5)
                    axes[3].text(P_star, S_star, f"({P_star:.2f}, {S_star:.2f})", color='purple', fontsize=8)

                    # Calculate distance from (0, 0) to the intersection point
                    distance = np.sqrt(P_star**2 + S_star**2)
                    distances['hyperbolic'] = distance
                    axes[3].text(P_star, S_star - 0.1, f"Dist: {distance:.2f}", color='purple', fontsize=8)
        except RuntimeError:
            axes[3].set_title("Hyperbolic regression failed")
    else:
        axes[3].set_title("Insufficient data for Hyperbolic Regression")
    axes[3].set_xlabel("Accuracy (Bad → Good)")
    axes[3].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    return distances

# Example usage
distances = scatter_plot_with_options_subplots(all_combined_metrics, All_dates, 'right', 'distance', 'speed', 0, 1)


def scatter_plot_trials_separate(all_combined_metrics, subjects, hand, metric_x, metric_y, trial_indices, reach_index):
    """
    Parameters:
        all_combined_metrics (dict): Combined metrics data.
        subjects (list): List of subject identifiers.
        hand (str): Hand ('left' or 'right').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        trial_indices (list): List of trial indices to include.
        reach_index (int): Reach index to include.
    Returns:
        dict: Distance from (0.0, 0.0) to the intersection point for hyperbolic regression for each trial.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    distances = {}

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']  # color cycle for multiple trials

    for i, trial_index in enumerate(trial_indices):
        x_values = []
        y_values = []
        color = colors[i % len(colors)]

        for subject in subjects:
            if subject not in all_combined_metrics or hand not in all_combined_metrics[subject]:
                continue

            trials = list(all_combined_metrics[subject][hand][metric_x].keys())

            if trial_index < len(trials):
                trial = trials[trial_index]
                trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                if reach_index < len(trial_x) and reach_index < len(trial_y):
                    x_values.append(trial_x[reach_index])
                    y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        label = f"Trial {trial_index + 1}, Reach {reach_index + 1}"

        # Scatter plot
        axes[0].scatter(x_values, y_values, color=color, alpha=0.7, label=label)

        # Spearman correlation
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
            axes[1].scatter(x_values, y_values, color=color, alpha=0.7, label=f"{label} (r={correlation:.2f})")

        # Linear regression
        if len(x_values) > 1 and len(y_values) > 1:
            x_reshaped = x_values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x_reshaped, y_values)
            y_pred = model.predict(x_reshaped)
            axes[2].scatter(x_values, y_values, color=color, alpha=0.7)
            axes[2].plot(x_values, y_pred, color=color, linestyle='--', label=f"{label} Fit")

        # Hyperbolic regression
        if len(x_values) > 1 and len(y_values) > 1:
            try:
                def hyperbolic_model(x, a, b):
                    return a + b / x

                params, _ = curve_fit(hyperbolic_model, 1 / x_values, y_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(1 / x_values), max(1 / x_values), 100)
                fit_y = hyperbolic_model(fit_x, a, b)
                # axes[3].scatter(1 / x_values, y_values, color=color, alpha=0.7)
                axes[3].plot(fit_x, fit_y, color=color, linestyle='--', label=f"{label} Hyper Fit")

                # Diagonal line
                x_mean, x_sd = np.mean(1 / x_values), np.std(1 / x_values)
                y_mean, y_sd = np.mean(y_values), np.std(y_values)
                m = (y_mean + 3 * y_sd) / (x_mean + 3 * x_sd)
                x_diag = np.linspace(min(1 / x_values), max(1 / x_values), 100)
                y_diag = m * x_diag
                axes[3].plot(x_diag, y_diag, color='black', linestyle='--', label='Diagonal')

                # Intersection
                A, B, C = m, -a, -b
                discriminant = B ** 2 - 4 * A * C
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    root1 = (-B + sqrt_disc) / (2 * A)
                    root2 = (-B - sqrt_disc) / (2 * A)
                    positive_roots = [r for r in (root1, root2) if r > 0]
                    P_star = max(positive_roots) if positive_roots else np.nan
                    S_star = m * P_star if np.isfinite(P_star) else np.nan

                    if np.isfinite(P_star) and np.isfinite(S_star):
                        axes[3].scatter(P_star, S_star, color=color, edgecolor='black', s=50, zorder=5)
                        distance = np.sqrt(P_star ** 2 + S_star ** 2)
                        distances[f"trial_{trial_index + 1}"] = distance

            except RuntimeError:
                axes[3].set_title("Hyperbolic regression failed for some trials")

    axes[0].set_title(f"{metric_y.capitalize()} vs {metric_x.capitalize()} Scatter Plot")
    axes[0].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[0].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[0].legend()

    axes[1].set_title("Spearman Correlation")
    axes[1].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[1].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[1].legend()

    axes[2].set_title("Linear Regression")
    axes[2].set_xlabel(f"{metric_x.capitalize()} (Good → Bad)")
    axes[2].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    axes[2].legend()

    axes[3].set_title("Hyperbolic Regression with Diagonal")
    axes[3].set_xlabel("Accuracy (Bad → Good)")
    axes[3].set_ylabel(f"{metric_y.capitalize()} (Slow → Fast)")
    # axes[3].legend()

    plt.tight_layout()
    plt.show()

    return distances

distances = scatter_plot_trials_separate(
    all_combined_metrics, All_dates, 'right', 
    'distance', 'speed', trial_indices=list(range(31)), reach_index=3
)
# Print distances in a more readable format
for trial, distance in distances.items():
    print(f"{trial}, Distance: {distance:.2f}")

# Scatter plot for distances from scatter_plot_trials_separate and calculate Spearman correlation
def scatter_plot_distances_from_trials_with_spearman(distances):
    # Extract trial indices and distances
    trial_indices = list(distances.keys())
    distance_values = np.array(list(distances.values()))

    # Remove outliers using z-score
    z_scores = np.abs(zscore(distance_values))
    valid_indices = z_scores < 3  # Keep only non-outlier values
    filtered_trial_indices = np.array(trial_indices)[valid_indices]
    filtered_distance_values = distance_values[valid_indices]

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_trial_indices, filtered_distance_values, alpha=0.7, color='blue')
    plt.title("Scatter Plot of Distances from Trials (Outliers Removed)")
    plt.xlabel("Trial Index")
    plt.xticks(ticks=range(len(filtered_trial_indices)), labels=range(len(filtered_trial_indices)))
    plt.ylabel("Distance")
    plt.grid(alpha=0.3)

    # Calculate Spearman correlation
    if len(filtered_trial_indices) > 1 and len(filtered_distance_values) > 1:
        correlation, p_value = spearmanr(filtered_trial_indices, filtered_distance_values)
        plt.text(0.05, 0.95, f"Spearman r: {correlation:.2f}\np: {p_value:.4f}",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='red')
    else:
        plt.text(0.05, 0.95, "Not enough data for Spearman correlation",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='red')

    plt.tight_layout()
    plt.show()

# Example usage
scatter_plot_distances_from_trials_with_spearman(distances)

# # -------------------------------------------------------------------------------------------------------------------

# Perform linear regression for each subject, each hand, and each reach type
def perform_linear_regression(all_combined_metrics):
    regression_results = {}
    for subject in all_combined_metrics.keys():
        regression_results[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            regression_results[subject][hand] = {}
            for reach_index in range(16):  # Assuming there are 16 reach types
                distances = []
                speeds = []

                for trial in all_combined_metrics[subject][hand]['distance'].keys():
                    trial_distances = np.array(all_combined_metrics[subject][hand]['distance'][trial])
                    trial_speeds = np.array(all_combined_metrics[subject][hand]['speed'][trial])

                    # Ensure the reach index is valid
                    if reach_index < len(trial_distances) and reach_index < len(trial_speeds):
                        distances.append(trial_distances[reach_index])
                        speeds.append(trial_speeds[reach_index])

                # Remove NaN values
                distances = np.array(distances)
                speeds = np.array(speeds)
                valid_indices = ~np.isnan(distances) & ~np.isnan(speeds)
                distances = distances[valid_indices].reshape(-1, 1)
                speeds = speeds[valid_indices]

                # Perform linear regression if there are enough valid data points
                if len(distances) > 1:
                    model = LinearRegression()
                    model.fit(distances, speeds)
                    regression_results[subject][hand][reach_index] = {
                        'slope': model.coef_[0],
                        'intercept': model.intercept_,
                        'r_squared': model.score(distances, speeds)
                    }
                else:
                    regression_results[subject][hand][reach_index] = {
                        'slope': np.nan,
                        'intercept': np.nan,
                        'r_squared': np.nan
                    }

    return regression_results

# Example usage
regression_results = perform_linear_regression(all_combined_metrics)


# Plot regression results overlaying data points
def plot_regression_results_and_return_medians(regression_results, all_combined_metrics, selected_subjects=None):
    # If no specific subjects are selected, plot for all subjects
    if selected_subjects is None:
        selected_subjects = list(regression_results.keys())
    elif isinstance(selected_subjects, str):
        selected_subjects = [selected_subjects]  # Convert single subject to list

    all_medians = {}

    for subject in selected_subjects:
        if subject not in regression_results:
            print(f"Subject {subject} not found in regression results.")
            continue

        all_medians[subject] = {}

        for hand in regression_results[subject].keys():
            all_medians[subject][hand] = {}

            fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
            axes = axes.flatten()

            for reach_index in range(16):  # Assuming there are 16 reach types
                ax = axes[reach_index]
                distances = []
                speeds = []

                for trial in all_combined_metrics[subject][hand]['distance'].keys():
                    trial_distances = np.array(all_combined_metrics[subject][hand]['distance'][trial])
                    trial_speeds = np.array(all_combined_metrics[subject][hand]['speed'][trial])

                    # Ensure the reach index is valid
                    if reach_index < len(trial_distances) and reach_index < len(trial_speeds):
                        distances.append(trial_distances[reach_index])
                        speeds.append(trial_speeds[reach_index])

                # Remove NaN values
                distances = np.array(distances)
                speeds = np.array(speeds)
                valid_indices = ~np.isnan(distances) & ~np.isnan(speeds)
                distances = distances[valid_indices]
                speeds = speeds[valid_indices]

                # Scatter plot of data points
                ax.scatter(distances, speeds, alpha=0.7, label='Data Points')

                # Calculate and plot medians
                if len(distances) > 0 and len(speeds) > 0:
                    median_x = np.median(distances)
                    median_y = np.median(speeds)
                    all_medians[subject][hand][reach_index] = (median_x, median_y)
                    ax.scatter(median_x, median_y, color='blue', s=100, label='Median', zorder=5)
                    ax.text(median_x, median_y, f"({median_x:.2f}, {median_y:.2f})", 
                            fontsize=8, color='blue', ha='center', va='bottom')

                # Overlay regression line if available
                if reach_index in regression_results[subject][hand]:
                    reg_result = regression_results[subject][hand][reach_index]
                    if not np.isnan(reg_result['slope']):
                        x_vals = np.linspace(min(distances), max(distances), 100)
                        y_vals = reg_result['slope'] * x_vals + reg_result['intercept']
                        ax.plot(x_vals, y_vals, color='red', label='Regression Line')

                        # Display R-squared value
                        ax.text(0.05, 0.95, f"R²: {reg_result['r_squared']:.2f}", 
                                transform=ax.transAxes, fontsize=10, verticalalignment='top')

                ax.set_title(f"Reach {reach_index + 1}")
                ax.set_xlabel("Distance")
                ax.set_ylabel("Speed")
                ax.legend()

            fig.suptitle(f"Regression Results for Subject: {subject}, Hand: {hand.capitalize()}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    return all_medians

# Example usage
medians = plot_regression_results_and_return_medians(regression_results, all_combined_metrics, selected_subjects=['07/22/HW'])
print(medians)

# Scatter plot the medians from plot_regression_results_and_return_medians and calculate Spearman correlation
def scatter_plot_medians_with_spearman(medians, subject, hand):
    """
    Scatter plot the medians for a specific subject and hand, and calculate Spearman correlation.

    Parameters:
        medians (dict): Dictionary containing median values for each subject, hand, and reach index.
        subject (str): Subject identifier.
        hand (str): Hand ('left' or 'right').
    """
    if subject not in medians or hand not in medians[subject]:
        print(f"No median data available for subject: {subject}, hand: {hand}")
        return

    reach_indices = []
    median_distances = []
    median_speeds = []

    for reach_index, (median_distance, median_speed) in medians[subject][hand].items():
        reach_indices.append(reach_index + 1)  # Reach indices are 1-based
        median_distances.append(median_distance)
        median_speeds.append(median_speed)

    # Calculate Spearman correlation
    if len(median_distances) > 1 and len(median_speeds) > 1:
        correlation, p_value = spearmanr(median_distances, median_speeds)
    else:
        correlation, p_value = np.nan, np.nan

    plt.figure(figsize=(8, 6))
    plt.scatter(median_distances, median_speeds, color='blue', alpha=0.7)

    # Annotate each point with its reach index
    for i, reach_index in enumerate(reach_indices):
        plt.text(median_distances[i], median_speeds[i], str(reach_index), fontsize=9, ha='right', va='bottom')

    plt.title(f"Scatter Plot of Medians for Subject: {subject}, Hand: {hand.capitalize()}\n"
              f"Spearman r: {correlation:.2f}, p: {p_value:.4f}")
    plt.xlabel("Median Distance")
    plt.ylabel("Median Speed")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
scatter_plot_medians_with_spearman(medians, '07/22/HW', 'right')


# Perform linear regression for each subject, each hand, and each reach type using the formula S(P) = a + b / accuracy
def perform_hyperbolic_regression(all_combined_metrics):
    regression_results = {}
    for subject in all_combined_metrics.keys():
        regression_results[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            regression_results[subject][hand] = {}
            for reach_index in range(16):  # Assuming there are 16 reach types
                accuracies = []
                speeds = []

                for trial in all_combined_metrics[subject][hand]['accuracy'].keys():
                    trial_accuracies = np.array(all_combined_metrics[subject][hand]['accuracy'][trial])
                    trial_speeds = np.array(all_combined_metrics[subject][hand]['speed'][trial])

                    # Ensure the reach index is valid
                    if reach_index < len(trial_accuracies) and reach_index < len(trial_speeds):
                        accuracies.append(trial_accuracies[reach_index])
                        speeds.append(trial_speeds[reach_index])

                # Remove NaN values
                accuracies = np.array(accuracies)
                speeds = np.array(speeds)
                valid_indices = ~np.isnan(accuracies) & ~np.isnan(speeds)
                accuracies = accuracies[valid_indices]
                speeds = speeds[valid_indices]

                # Perform hyperbolic regression if there are enough valid data points
                if len(accuracies) > 1:
                    def hyperbolic_model(x, a, b):
                        return a + b / x

                    try:
                        params, _ = curve_fit(hyperbolic_model, accuracies, speeds, p0=(0, 1))
                        a, b = params
                        r_squared = 1 - (np.sum((speeds - hyperbolic_model(accuracies, a, b))**2) /
                                         np.sum((speeds - np.mean(speeds))**2))
                        regression_results[subject][hand][reach_index] = {
                            'a': a,
                            'b': b,
                            'r_squared': r_squared
                        }
                    except RuntimeError:
                        regression_results[subject][hand][reach_index] = {
                            'a': np.nan,
                            'b': np.nan,
                            'r_squared': np.nan
                        }
                else:
                    regression_results[subject][hand][reach_index] = {
                        'a': np.nan,
                        'b': np.nan,
                        'r_squared': np.nan
                    }

    return regression_results

# Example usage
hyperbolic_regression_results = perform_hyperbolic_regression(all_combined_metrics)

# Plot hyperbolic regression results overlaying data points
def plot_hyperbolic_regression_results(hyperbolic_regression_results, all_combined_metrics, selected_subjects=None):
    # If no specific subjects are selected, plot for all subjects
    if selected_subjects is None:
        selected_subjects = list(hyperbolic_regression_results.keys())
    elif isinstance(selected_subjects, str):
        selected_subjects = [selected_subjects]  # Convert single subject to list

    for subject in selected_subjects:
        if subject not in hyperbolic_regression_results:
            print(f"Subject {subject} not found in hyperbolic regression results.")
            continue

        for hand in hyperbolic_regression_results[subject].keys():
            fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
            axes = axes.flatten()

            for reach_index in range(16):  # Assuming there are 16 reach types
                ax = axes[reach_index]
                accuracies = []
                speeds = []

                for trial in all_combined_metrics[subject][hand]['accuracy'].keys():
                    trial_accuracies = np.array(all_combined_metrics[subject][hand]['accuracy'][trial])
                    trial_speeds = np.array(all_combined_metrics[subject][hand]['speed'][trial])

                    # Ensure the reach index is valid
                    if reach_index < len(trial_accuracies) and reach_index < len(trial_speeds):
                        accuracies.append(trial_accuracies[reach_index])
                        speeds.append(trial_speeds[reach_index])

                # Remove NaN values
                accuracies = np.array(accuracies)
                speeds = np.array(speeds)
                valid_indices = ~np.isnan(accuracies) & ~np.isnan(speeds)
                accuracies = accuracies[valid_indices]
                speeds = speeds[valid_indices]

                # Scatter plot of data points
                ax.scatter(accuracies, speeds, alpha=0.7, label='Data Points')

                # Overlay hyperbolic regression curve if available
                if reach_index in hyperbolic_regression_results[subject][hand]:
                    reg_result = hyperbolic_regression_results[subject][hand][reach_index]
                    if not np.isnan(reg_result['a']) and not np.isnan(reg_result['b']):
                        x_vals = np.linspace(min(accuracies), max(accuracies), 100)
                        y_vals = reg_result['a'] + reg_result['b'] / x_vals
                        ax.plot(x_vals, y_vals, color='red', label='Hyperbolic Fit')

                        # Display R-squared value
                        ax.text(0.05, 0.95, f"R²: {reg_result['r_squared']:.2f}", 
                                transform=ax.transAxes, fontsize=10, verticalalignment='top')

                ax.set_title(f"Reach {reach_index + 1}")
                ax.set_xlabel("Accuracy")
                ax.set_ylabel("Speed")
                ax.legend()

            fig.suptitle(f"Hyperbolic Regression Results for Subject: {subject}, Hand: {hand.capitalize()}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

# Example usage
plot_hyperbolic_regression_results(hyperbolic_regression_results, all_combined_metrics, selected_subjects=['07/22/HW'])

def plot_hyperbolic_regression_results(hyperbolic_regression_results, all_combined_metrics, selected_subjects=None):
    """
    Plot hyperbolic regression results for selected subjects, overlaying data points, regression curves, 
    and diagonal lines. Also calculates and marks the intersection point of the regression curve and diagonal line.

    Parameters:
        hyperbolic_regression_results (dict): Results of hyperbolic regression for each subject, hand, and reach type.
        all_combined_metrics (dict): Combined metrics data for all subjects, hands, and trials.
        selected_subjects (list or str, optional): List of subjects to plot. If None, plots for all subjects.
    """
    # If no specific subjects are selected, plot for all subjects
    if selected_subjects is None:
        selected_subjects = list(hyperbolic_regression_results.keys())
    elif isinstance(selected_subjects, str):
        selected_subjects = [selected_subjects]  # Convert single subject to list

    # Iterate over each selected subject
    for subject in selected_subjects:
        if subject not in hyperbolic_regression_results:
            print(f"Subject {subject} not found in hyperbolic regression results.")
            continue

        # Iterate over each hand for the subject
        for hand in hyperbolic_regression_results[subject].keys():
            # Create a 4x4 grid of subplots for the 16 reach types
            fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
            axes = axes.flatten()

            # Iterate over each reach type
            for reach_index in range(16):  # Assuming there are 16 reach types
                ax = axes[reach_index]
                accuracies = []
                speeds = []

                # Collect accuracy and speed data for the current reach type
                for trial in all_combined_metrics[subject][hand]['accuracy'].keys():
                    trial_accuracies = np.array(all_combined_metrics[subject][hand]['accuracy'][trial])
                    trial_speeds = np.array(all_combined_metrics[subject][hand]['speed'][trial])

                    # Ensure the reach index is valid
                    if reach_index < len(trial_accuracies) and reach_index < len(trial_speeds):
                        accuracies.append(trial_accuracies[reach_index])
                        speeds.append(trial_speeds[reach_index])

                # Remove NaN values
                accuracies = np.array(accuracies)
                speeds = np.array(speeds)
                valid_indices = ~np.isnan(accuracies) & ~np.isnan(speeds)
                accuracies = accuracies[valid_indices]
                speeds = speeds[valid_indices]

                # Scatter plot of data points
                ax.scatter(accuracies, speeds, alpha=0.7, label='Data Points')

                # Overlay hyperbolic regression curve if available
                if reach_index in hyperbolic_regression_results[subject][hand]:
                    reg_result = hyperbolic_regression_results[subject][hand][reach_index]
                    if not np.isnan(reg_result['a']) and not np.isnan(reg_result['b']):
                        # Generate x values for the regression curve
                        x_vals = np.linspace(min(accuracies), max(accuracies), 100)
                        # Calculate y values using the hyperbolic regression formula
                        y_vals = reg_result['a'] + reg_result['b'] / x_vals
                        ax.plot(x_vals, y_vals, color='red', label='Hyperbolic Fit')

                        # Display R-squared value on the plot
                        ax.text(0.05, 0.95, f"R²: {reg_result['r_squared']:.2f}", 
                                transform=ax.transAxes, fontsize=10, verticalalignment='top')

                # Calculate and plot diagonal line
                if len(accuracies) > 0 and len(speeds) > 0:
                    # Calculate mean and standard deviation for accuracies and speeds
                    accuracy_mean = np.mean(accuracies)
                    accuracy_sd = np.std(accuracies)
                    speed_mean = np.mean(speeds)
                    speed_sd = np.std(speeds)

                    # Calculate slope (m) for the diagonal line
                    m = (speed_mean + 3 * speed_sd) / (accuracy_mean + 3 * accuracy_sd)
                    # Generate x and y values for the diagonal line
                    x_diag = np.linspace(0, max(accuracies), 1000)
                    y_diag = m * x_diag
                    ax.plot(x_diag, y_diag, color='blue', linestyle='--', label='Diagonal Line (S = mP)')

                    # Solve for the intersection point of the regression curve and diagonal line
                    a = reg_result['a']
                    b = reg_result['b']

                    # Quadratic coefficients for m*P^2 - a*P - b = 0
                    A = m
                    B = -a
                    C = -b

                    # Calculate discriminant and roots
                    discriminant = B**2 - 4 * A * C
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        root1 = (-B + sqrt_disc) / (2 * A)
                        root2 = (-B - sqrt_disc) / (2 * A)

                        # Select positive root(s)
                        positive_roots = [r for r in (root1, root2) if r > 0]
                        P_star = max(positive_roots) if positive_roots else np.nan
                        S_star = m * P_star if np.isfinite(P_star) else np.nan
                    else:
                        P_star = np.nan
                        S_star = np.nan
                    
                    # Plot intersection point if valid
                    if np.isfinite(P_star) and np.isfinite(S_star):
                        ax.scatter(P_star, S_star, color='purple', label='Intersection Point', zorder=5)
                        ax.text(P_star, S_star, f"({P_star:.2f}, {S_star:.2f})", color='purple', fontsize=8)
                else:
                    print(f"Not enough data to calculate diagonal line for Subject: {subject}, Hand: {hand}, Reach Index: {reach_index}")

                # Set plot title and labels
                ax.set_title(f"Reach {reach_index + 1}")
                ax.set_xlabel("Accuracy")
                ax.set_ylabel("Speed")
                ax.legend()

            # Set the overall title for the figure
            fig.suptitle(f"Hyperbolic Regression Results for Subject: {subject}, Hand: {hand.capitalize()}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

# Example usage
plot_hyperbolic_regression_results(hyperbolic_regression_results, all_combined_metrics, selected_subjects=['07/22/HW'])


def plot_hyperbolic_regression_results(hyperbolic_regression_results, all_combined_metrics, selected_subjects=None):
    """
    Plot hyperbolic regression results for selected subjects, overlaying data points, regression curves, 
    and diagonal lines. Also calculates and marks the intersection point of the regression curve and diagonal line.

    Parameters:
        hyperbolic_regression_results (dict): Results of hyperbolic regression for each subject, hand, and reach type.
        all_combined_metrics (dict): Combined metrics data for all subjects, hands, and trials.
        selected_subjects (list or str, optional): List of subjects to plot. If None, plots for all subjects.
    """
    # Dictionary to store lengths from (0, 0) to the intersection point
    intersection_lengths = {}

    # If no specific subjects are selected, plot for all subjects
    if selected_subjects is None:
        selected_subjects = list(hyperbolic_regression_results.keys())
    elif isinstance(selected_subjects, str):
        selected_subjects = [selected_subjects]  # Convert single subject to list

    # Iterate over each selected subject
    for subject in selected_subjects:
        if subject not in hyperbolic_regression_results:
            print(f"Subject {subject} not found in hyperbolic regression results.")
            continue

        intersection_lengths[subject] = {}

        # Iterate over each hand for the subject
        for hand in hyperbolic_regression_results[subject].keys():
            intersection_lengths[subject][hand] = {}

            # Create a 4x4 grid of subplots for the 16 reach types
            fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
            axes = axes.flatten()

            # Iterate over each reach type
            for reach_index in range(16):  # Assuming there are 16 reach types
                ax = axes[reach_index]
                accuracies = []
                speeds = []

                # Collect accuracy and speed data for the current reach type
                for trial in all_combined_metrics[subject][hand]['accuracy'].keys():
                    trial_accuracies = np.array(all_combined_metrics[subject][hand]['accuracy'][trial])
                    trial_speeds = np.array(all_combined_metrics[subject][hand]['speed'][trial])

                    # Ensure the reach index is valid
                    if reach_index < len(trial_accuracies) and reach_index < len(trial_speeds):
                        accuracies.append(trial_accuracies[reach_index])
                        speeds.append(trial_speeds[reach_index])

                # Remove NaN values
                accuracies = np.array(accuracies)
                speeds = np.array(speeds)
                valid_indices = ~np.isnan(accuracies) & ~np.isnan(speeds)
                accuracies = accuracies[valid_indices]
                speeds = speeds[valid_indices]

                # Scatter plot of data points
                ax.scatter(accuracies, speeds, alpha=0.7, label='Data Points')

                # Overlay hyperbolic regression curve if available
                if reach_index in hyperbolic_regression_results[subject][hand]:
                    reg_result = hyperbolic_regression_results[subject][hand][reach_index]
                    if not np.isnan(reg_result['a']) and not np.isnan(reg_result['b']):
                        # Generate x values for the regression curve
                        x_vals = np.linspace(min(accuracies), max(accuracies), 100)
                        # Calculate y values using the hyperbolic regression formula
                        y_vals = reg_result['a'] + reg_result['b'] / x_vals
                        ax.plot(x_vals, y_vals, color='red', label='Hyperbolic Fit')

                        # Display R-squared value on the plot
                        ax.text(0.05, 0.95, f"R²: {reg_result['r_squared']:.2f}", 
                                transform=ax.transAxes, fontsize=10, verticalalignment='top')

                # Calculate and plot diagonal line
                if len(accuracies) > 0 and len(speeds) > 0:
                    # Calculate mean and standard deviation for accuracies and speeds
                    accuracy_mean = np.mean(accuracies)
                    accuracy_sd = np.std(accuracies)
                    speed_mean = np.mean(speeds)
                    speed_sd = np.std(speeds)

                    # Calculate slope (m) for the diagonal line
                    m = (speed_mean + 3 * speed_sd) / (accuracy_mean + 3 * accuracy_sd)
                    # Generate x and y values for the diagonal line
                    x_diag = np.linspace(0, max(accuracies), 1000)
                    y_diag = m * x_diag
                    ax.plot(x_diag, y_diag, color='blue', linestyle='--', label='Diagonal Line (S = mP)')

                    # Solve for the intersection point of the regression curve and diagonal line
                    a = reg_result['a']
                    b = reg_result['b']

                    # Quadratic coefficients for m*P^2 - a*P - b = 0
                    A = m
                    B = -a
                    C = -b

                    # Calculate discriminant and roots
                    discriminant = B**2 - 4 * A * C
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        root1 = (-B + sqrt_disc) / (2 * A)
                        root2 = (-B - sqrt_disc) / (2 * A)

                        # Select positive root(s)
                        positive_roots = [r for r in (root1, root2) if r > 0]
                        P_star = max(positive_roots) if positive_roots else np.nan
                        S_star = m * P_star if np.isfinite(P_star) else np.nan
                    else:
                        P_star = np.nan
                        S_star = np.nan
                    
                    # Plot intersection point if valid
                    if np.isfinite(P_star) and np.isfinite(S_star):
                        ax.scatter(P_star, S_star, color='purple', label='Intersection Point', zorder=5)
                        ax.text(P_star, S_star, f"({P_star:.2f}, {S_star:.2f})", color='purple', fontsize=8)

                        # Calculate length from (0, 0) to the intersection point
                        length = np.sqrt(P_star**2 + S_star**2)
                        intersection_lengths[subject][hand][reach_index] = length
                    else:
                        intersection_lengths[subject][hand][reach_index] = np.nan
                else:
                    print(f"Not enough data to calculate diagonal line for Subject: {subject}, Hand: {hand}, Reach Index: {reach_index}")
                    intersection_lengths[subject][hand][reach_index] = np.nan

                # Set plot title and labels
                ax.set_title(f"Reach {reach_index + 1}")
                ax.set_xlabel("Accuracy")
                ax.set_ylabel("Speed")
                ax.legend()

            # Set the overall title for the figure
            fig.suptitle(f"Hyperbolic Regression Results for Subject: {subject}, Hand: {hand.capitalize()}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    return intersection_lengths

# Example usage
intersection_lengths = plot_hyperbolic_regression_results(hyperbolic_regression_results, all_combined_metrics, selected_subjects=['07/22/HW'])

def plot_combined_hyperbolic_regression(hyperbolic_regression_results, all_combined_metrics, selected_subjects=None, plot_data_points=True):
    """
    Overlay hyperbolic regression curves and intersection points for all reach types into one plot.
    Different shades of blue indicate each reach type.

    Parameters:
        hyperbolic_regression_results (dict): Results of hyperbolic regression for each subject, hand, and reach type.
        all_combined_metrics (dict): Combined metrics data for all subjects, hands, and trials.
        selected_subjects (list or str, optional): List of subjects to plot. If None, plots for all subjects.
        plot_data_points (bool, optional): Whether to plot data points. Default is True.
    """
    # If no specific subjects are selected, plot for all subjects
    if selected_subjects is None:
        selected_subjects = list(hyperbolic_regression_results.keys())
    elif isinstance(selected_subjects, str):
        selected_subjects = [selected_subjects]  # Convert single subject to list

    for subject in selected_subjects:
        if subject not in hyperbolic_regression_results:
            print(f"Subject {subject} not found in hyperbolic regression results.")
            continue

        for hand in hyperbolic_regression_results[subject].keys():
            plt.figure(figsize=(10, 8))
            colors = sns.color_palette("Blues", 16)  # Generate 16 shades of blue

            for reach_index in range(16):  # Assuming there are 16 reach types
                accuracies = []
                speeds = []

                # Collect accuracy and speed data for the current reach type
                for trial in all_combined_metrics[subject][hand]['accuracy'].keys():
                    trial_accuracies = np.array(all_combined_metrics[subject][hand]['accuracy'][trial])
                    trial_speeds = np.array(all_combined_metrics[subject][hand]['speed'][trial])

                    if reach_index < len(trial_accuracies) and reach_index < len(trial_speeds):
                        accuracies.append(trial_accuracies[reach_index])
                        speeds.append(trial_speeds[reach_index])

                # Remove NaN values
                accuracies = np.array(accuracies)
                speeds = np.array(speeds)
                valid_indices = ~np.isnan(accuracies) & ~np.isnan(speeds)
                accuracies = accuracies[valid_indices]
                speeds = speeds[valid_indices]

                # Plot data points if enabled
                if plot_data_points:
                    plt.scatter(accuracies, speeds, alpha=0.5, color=colors[reach_index], label=f"Reach {reach_index + 1}")

                # Overlay hyperbolic regression curve if available
                if reach_index in hyperbolic_regression_results[subject][hand]:
                    reg_result = hyperbolic_regression_results[subject][hand][reach_index]
                    if not np.isnan(reg_result['a']) and not np.isnan(reg_result['b']):
                        x_vals = np.linspace(min(accuracies), max(accuracies), 100)
                        y_vals = reg_result['a'] + reg_result['b'] / x_vals
                        plt.plot(x_vals, y_vals, color=colors[reach_index])

                        # Calculate and plot intersection point
                        a = reg_result['a']
                        b = reg_result['b']
                        accuracy_mean = np.mean(accuracies)
                        accuracy_sd = np.std(accuracies)
                        speed_mean = np.mean(speeds)
                        speed_sd = np.std(speeds)
                        m = (speed_mean + 3 * speed_sd) / (accuracy_mean + 3 * accuracy_sd)

                        # Solve for intersection point
                        A = m
                        B = -a
                        C = -b
                        discriminant = B**2 - 4 * A * C
                        if discriminant >= 0:
                            sqrt_disc = np.sqrt(discriminant)
                            root1 = (-B + sqrt_disc) / (2 * A)
                            root2 = (-B - sqrt_disc) / (2 * A)
                            positive_roots = [r for r in (root1, root2) if r > 0]
                            P_star = max(positive_roots) if positive_roots else np.nan
                            S_star = m * P_star if np.isfinite(P_star) else np.nan

                            if np.isfinite(P_star) and np.isfinite(S_star):
                                plt.scatter(P_star, S_star, color=colors[reach_index], edgecolor='black', zorder=5, label=f"motor acuity {reach_index + 1}")

            plt.title(f"Combined Hyperbolic Regression for Subject: {subject}, Hand: {hand.capitalize()}", fontsize=16)
            plt.xlabel("Accuracy", fontsize=14)
            plt.ylabel("Speed", fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title="Reach Types")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

# Example usage
plot_combined_hyperbolic_regression(hyperbolic_regression_results, all_combined_metrics, selected_subjects=['08/02/AR'], plot_data_points=False)

for subejct in All_dates:
    plot_combined_hyperbolic_regression(hyperbolic_regression_results, all_combined_metrics, selected_subjects=[subejct], plot_data_points=False)

# # -------------------------------------------------------------------------------------------------------------------
# --- SPEARMAN CORRELATION FOR SPEED VS ACCURACY ---
# Calculate Spearman correlation for speed vs accuracy for all reach types across all trials
def calculate_spearman_correlation_speed_vs_accuracy_all_reaches(all_combined_metrics):
    spearman_results = {}
    for subject in all_combined_metrics.keys():
        spearman_results[subject] = {}
        for hand in all_combined_metrics[subject].keys():
            speeds = all_combined_metrics[subject][hand]['speed']
            accuracies = all_combined_metrics[subject][hand]['accuracy']
            
            reach_spearman_results = {}
            
            for reach_index in range(16):  # Assuming there are 16 reach types
                reach_speeds = []
                reach_accuracies = []
                
                for trial in speeds.keys():
                    trial_speeds = np.array(speeds[trial])
                    trial_accuracies = np.array(accuracies[trial])
                    
                    # Ensure the reach index is valid
                    if reach_index < len(trial_speeds) and reach_index < len(trial_accuracies):
                        reach_speeds.append(trial_speeds[reach_index])
                        reach_accuracies.append(trial_accuracies[reach_index])
                
                # Remove NaN values
                reach_speeds = np.array(reach_speeds)
                reach_accuracies = np.array(reach_accuracies)
                valid_indices = ~np.isnan(reach_speeds) & ~np.isnan(reach_accuracies)
                reach_speeds = reach_speeds[valid_indices]
                reach_accuracies = reach_accuracies[valid_indices]
                
                # Calculate Spearman correlation
                if len(reach_speeds) > 1 and len(reach_accuracies) > 1:
                    correlation, p_value = spearmanr(reach_speeds, reach_accuracies)
                else:
                    correlation, p_value = np.nan, np.nan
                
                reach_spearman_results[reach_index] = {
                    'correlation': correlation,
                    'p_value': p_value
                }
            
            spearman_results[subject][hand] = reach_spearman_results
    
    return spearman_results

# Example usage
spearman_results_all_reaches = calculate_spearman_correlation_speed_vs_accuracy_all_reaches(all_combined_metrics)

# Plot heatmap of Spearman correlation results for each subject and hand as subplots
def plot_spearman_heatmap_all_subjects(spearman_results_all_reaches, selected_subjects=None):
    # If no specific subjects are selected, plot for all subjects
    if selected_subjects is None:
        selected_subjects = list(spearman_results_all_reaches.keys())
    elif isinstance(selected_subjects, str):
        selected_subjects = [selected_subjects]  # Convert single subject to list

    hands = ['left','right']
    num_subjects = len(selected_subjects)
    num_hands = len(hands)
    max_subjects_per_column = min(4, num_subjects)  # Adjust rows based on the number of subjects
    num_columns = (num_subjects + max_subjects_per_column - 1) // max_subjects_per_column

    fig, axes = plt.subplots(max_subjects_per_column, num_columns * num_hands, 
                             figsize=(6 * num_columns * num_hands, 6 * max_subjects_per_column), squeeze=False)

    for idx, subject in enumerate(selected_subjects):
        col_offset = (idx // max_subjects_per_column) * num_hands
        row = idx % max_subjects_per_column

        for j, hand in enumerate(hands):
            col = col_offset + j
            if hand not in spearman_results_all_reaches[subject]:
                axes[row, col].axis('off')
                continue

            # Extract correlation values and p-values for the subject and hand
            reach_spearman_results = spearman_results_all_reaches[subject][hand]
            correlation_values = [reach_spearman_results[reach_index]['correlation'] for reach_index in range(16)]
            p_values = [reach_spearman_results[reach_index]['p_value'] for reach_index in range(16)]

            # Rearrange correlation values based on hand
            if hand == 'right':
                rearranged_indices = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
            elif hand == 'left':
                rearranged_indices = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            else:
                rearranged_indices = list(range(16))  # Default to no rearrangement

            correlation_grid = np.array([correlation_values[i] for i in rearranged_indices]).reshape(4, 4)
            p_value_grid = np.array([p_values[i] for i in rearranged_indices]).reshape(4, 4)

            # Plot heatmap
            im = axes[row, col].imshow(correlation_grid, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)

            # Annotate heatmap with correlation values
            for x in range(4):
                for y in range(4):
                    value = correlation_grid[x, y]
                    p_value = p_value_grid[x, y]
                    color = 'red' if p_value < 0.05 else 'black'  # Red text if significant
                    axes[row, col].text(y, x, f"{value:.2f}", ha='center', va='center', color=color)

            # Set axis labels and ticks
            axes[row, col].set_xticks(range(4))
            axes[row, col].set_xticklabels(range(1, 5))
            axes[row, col].set_yticks(range(4))
            axes[row, col].set_yticklabels(range(1, 5))
            axes[row, col].set_title(f"Subject: {subject}, Hand: {hand.capitalize()}")

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Spearman Correlation')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# Example usage for a single subject
plot_spearman_heatmap_all_subjects(spearman_results_all_reaches, selected_subjects='07/22/HW')
# Example usage for multiple subjects
plot_spearman_heatmap_all_subjects(spearman_results_all_reaches)

# Plot heatmap of Spearman correlation results for each subject and hand as subplots
def plot_spearman_heatmap_all_subjects_hand(spearman_results_all_reaches, selected_hand=None):
    subjects = list(spearman_results_all_reaches.keys())
    hands = ['right', 'left'] if selected_hand is None else [selected_hand]
    num_subjects = len(subjects)
    num_hands = len(hands)
    max_subjects_per_column = 4
    num_columns = (num_subjects + max_subjects_per_column - 1) // max_subjects_per_column

    fig, axes = plt.subplots(max_subjects_per_column, num_columns * num_hands, 
                             figsize=(6 * num_columns * num_hands, 6 * max_subjects_per_column), squeeze=False)

    for idx, subject in enumerate(subjects):
        col_offset = (idx // max_subjects_per_column) * num_hands
        row = idx % max_subjects_per_column

        for j, hand in enumerate(hands):
            col = col_offset + j
            if hand not in spearman_results_all_reaches[subject]:
                axes[row, col].axis('off')
                continue

            # Extract correlation values and p-values for the subject and hand
            reach_spearman_results = spearman_results_all_reaches[subject][hand]
            correlation_values = [reach_spearman_results[reach_index]['correlation'] for reach_index in range(16)]
            p_values = [reach_spearman_results[reach_index]['p_value'] for reach_index in range(16)]

            # Rearrange correlation values based on hand
            if hand == 'right':
                rearranged_indices = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
            elif hand == 'left':
                rearranged_indices = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            else:
                rearranged_indices = list(range(16))  # Default to no rearrangement

            correlation_grid = np.array([correlation_values[i] for i in rearranged_indices]).reshape(4, 4)
            p_value_grid = np.array([p_values[i] for i in rearranged_indices]).reshape(4, 4)

            # Plot heatmap
            im = axes[row, col].imshow(correlation_grid, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)

            # Annotate heatmap with correlation values
            for x in range(4):
                for y in range(4):
                    value = correlation_grid[x, y]
                    p_value = p_value_grid[x, y]
                    color = 'red' if p_value < 0.05 else 'black'  # Red text if significant
                    axes[row, col].text(y, x, f"{value:.2f}", ha='center', va='center', color=color)

            # Set axis labels and ticks
            axes[row, col].set_xticks(range(4))
            axes[row, col].set_xticklabels(range(1, 5))
            axes[row, col].set_yticks(range(4))
            axes[row, col].set_yticklabels(range(1, 5))
            axes[row, col].set_title(f"Subject: {subject}, Hand: {hand.capitalize()}")

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Spearman Correlation')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# Example usage
plot_spearman_heatmap_all_subjects_hand(spearman_results_all_reaches, selected_hand='left')
# # -------------------------------------------------------------------------------------------------------------------

# Scatter plot for durations vs distance for all trials, overlaying them with different colors
utils5.plot_durations_vs_distance_hand(all_combined_metrics, '07/22/SC', 'right')

# Scatter plot for durations vs distance for all trials, each hand as a subplot
utils5.plot_durations_vs_distance_hands(all_combined_metrics, '07/22/HW')

# Scatter plot for speed vs accuracy for all trials, each hand as a subplot
utils5.plot_speed_vs_accuracy(all_combined_metrics, '07/22/SC')




# Scatter plot for speed vs accuracy for a single reach, each hand as a subplot
def plot_speed_vs_accuracy_single_reach(all_combined_metrics, subject, reach_index):
    hands = ['left', 'right']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, hand in zip(axes, hands):
        trials = all_combined_metrics[subject][hand]['speed'].keys()
        colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

        all_speeds = []
        all_accuracies = []

        for i, trial_path in enumerate(trials):
            speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
            accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

            # Ensure the reach index is valid
            if reach_index >= len(speeds) or reach_index >= len(accuracies):
                print(f"Invalid reach index {reach_index} for trial {trial_path}.")
                continue

            # Scatter plot for the specified reach
            ax.scatter(speeds[reach_index], accuracies[reach_index], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')
            # Overlay line from (0.0, 0.0) to each point
            for speed, accuracy in zip(all_speeds, all_accuracies):
                ax.plot([0.0, speed], [0.0, accuracy], color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            all_speeds.append(speeds[reach_index])
            all_accuracies.append(accuracies[reach_index])

        # Calculate x and y limits based on data
        valid_data = [(s, a) for s, a in zip(all_speeds, all_accuracies) if not np.isnan(s) and not np.isnan(a)]
        if valid_data:
            valid_speeds, valid_accuracies = zip(*valid_data)
            # x_min, x_max = np.percentile(valid_speeds, [1, 90])
            # y_min, y_max = np.percentile(valid_accuracies, [0.5, 90])
            x_min, x_max = min(valid_speeds), max(valid_speeds)
            y_min, y_max = min(valid_accuracies), max(valid_accuracies)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)

            # Calculate and display Spearman correlation
            spearman_corr, _ = spearmanr(valid_speeds, valid_accuracies)
            ax.text(0.05, 0.95, f"Spearman r: {spearman_corr:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')

        else:
            print(f"Error: No valid data points for axis limits for {hand} hand.")
            continue

        ax.set_title(f"Speed vs Accuracy for {subject} ({hand.capitalize()} Hand, Reach {reach_index + 1})", fontsize=14)
        ax.set_xlabel("Speed (1/Duration)\n(Slow → Fast)", fontsize=14)
        ax.set_ylabel("Accuracy (1/Distance)\n(Bad → Good)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()


plot_speed_vs_accuracy_single_reach(all_combined_metrics, '07/22/HW', 4)



# Scatter plot for speed vs accuracy for a single reach, each hand as a subplot
utils5.plot_speed_vs_accuracy_single_reach(all_combined_metrics, '07/22/HW', 4)

# Scatter plot for speed vs accuracy for all reaches, each hand as a separate figure, 4x4 layout for each reach
utils5.plot_speed_vs_accuracy_all_reaches(all_combined_metrics, '07/22/HW')

# Scatter plot for motor_acuity vs sparc for all reaches, each hand as a separate figure, 4x4 layout for each reach
utils5.plot_motor_acuity_vs_sparc_all_reaches(all_combined_metrics, '07/22/HW')



# This function plots a heatmap for motor acuity values for a specific subject and hand.
def plot_motor_acuity_heatmap(all_combined_metrics, subject, hand):
    motor_acuity_data = all_combined_metrics[subject][hand]['motor_acuity']
    num_trials = len(motor_acuity_data)
    heatmap_data = np.full((num_trials, 16), np.nan)  # Initialize with NaN for missing values

    for trial_idx, (trial_number, motor_acuity_values) in enumerate(sorted(motor_acuity_data.items())):
        heatmap_data[trial_idx, :] = motor_acuity_values  # Fill the row with motor acuity values for each trial

    # Calculate outlier thresholds
    lower_thresh = np.percentile(heatmap_data[~np.isnan(heatmap_data)], 1)
    upper_thresh = np.percentile(heatmap_data[~np.isnan(heatmap_data)], 99)

    # Mask outliers in the heatmap data
    heatmap_data_masked = np.clip(heatmap_data, lower_thresh, upper_thresh)

    # Plot the heatmap for the motor acuity values
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_data_masked, aspect='auto', cmap='viridis', interpolation='nearest')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Motor Acuity (Clipped to Outlier Thresholds)')

    # Set axis labels and ticks
    ax.set_xlabel("Block Membership (1 to 16)")
    ax.set_ylabel("Trial Index")
    ax.set_xticks(np.arange(16))
    ax.set_xticklabels(np.arange(1, 17))
    ax.set_yticks(np.arange(num_trials))
    # ax.set_yticklabels([f"Trial {trial}" for trial in sorted(motor_acuity_data.keys())])

    # Add title
    ax.set_title(f"Motor Acuity Heatmap for Subject: {subject}, Hand: {hand}")

    plt.tight_layout()
    plt.show()

# Example usage
plot_motor_acuity_heatmap(all_combined_metrics, '06/24/DR', 'right')


# This function plots a heatmap for motor acuity values for each subject and hand as subplots.
def plot_motor_acuity_heatmap_all_subjects(all_combined_metrics):
    subjects = list(all_combined_metrics.keys())
    hands = ['right', 'left']
    num_subjects = len(subjects)
    num_hands = len(hands)

    fig, axes = plt.subplots(num_subjects, num_hands, figsize=(12, 6 * num_subjects), squeeze=False)

    for i, subject in enumerate(subjects):
        for j, hand in enumerate(hands):
            if hand not in all_combined_metrics[subject]:
                axes[i, j].axis('off')
                continue

            motor_acuity_data = all_combined_metrics[subject][hand]['motor_acuity']
            num_trials = len(motor_acuity_data)
            heatmap_data = np.full((num_trials, 16), np.nan)  # Initialize with NaN for missing values

            for trial_idx, (trial_number, motor_acuity_values) in enumerate(sorted(motor_acuity_data.items())):
                heatmap_data[trial_idx, :] = motor_acuity_values  # Fill the row with motor acuity values for each trial

            # Calculate outlier thresholds
            lower_thresh = np.percentile(heatmap_data[~np.isnan(heatmap_data)], 1)
            upper_thresh = np.percentile(heatmap_data[~np.isnan(heatmap_data)], 99)

            # Mask outliers in the heatmap data
            heatmap_data_masked = np.clip(heatmap_data, lower_thresh, upper_thresh)

            # Plot the heatmap for the motor acuity values
            im = axes[i, j].imshow(heatmap_data_masked, aspect='auto', cmap='viridis', interpolation='nearest')

            # Set axis labels and ticks
            axes[i, j].set_xlabel("Block Membership (1 to 16)")
            axes[i, j].set_ylabel("Trial Index")
            axes[i, j].set_xticks(np.arange(16))
            axes[i, j].set_xticklabels(np.arange(1, 17))
            axes[i, j].set_yticks(np.arange(num_trials))

            # Add title
            axes[i, j].set_title(f"Subject: {subject}, Hand: {hand}")

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Motor Acuity (Clipped to Outlier Thresholds)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# Example usage
plot_motor_acuity_heatmap_all_subjects(all_combined_metrics)









# This function plots a heatmap for each subject and hand based on the Block_Distance data.
def plot_heatmap_by_subject(Block_Distance):
    for subject, hands_data in Block_Distance.items():
        for hand, distances_data in hands_data.items():
            num_images = len(distances_data)
            heatmap_data = np.full((num_images, 16), np.nan)  # Initialize with NaN for missing values

            for img_idx, (image_number, distances) in enumerate(sorted(distances_data.items())):
                heatmap_data[img_idx, :] = distances  # Fill the row with distances for each image

            # Calculate outlier thresholds
            lower_thresh = np.percentile(heatmap_data[~np.isnan(heatmap_data)], 1)
            upper_thresh = np.percentile(heatmap_data[~np.isnan(heatmap_data)], 99)

            # Mask outliers in the heatmap data
            heatmap_data_masked = np.clip(heatmap_data, lower_thresh, upper_thresh)

            # Plot the heatmap for the current subject and hand
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(heatmap_data_masked, aspect='auto', cmap='viridis', interpolation='nearest')

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Distance (Clipped to Outlier Thresholds)')

            # Set axis labels and ticks
            ax.set_xlabel("Block Membership (1 to 16)")
            ax.set_ylabel("Image Index")
            ax.set_xticks(np.arange(16))
            ax.set_xticklabels(np.arange(1, 17))
            ax.set_yticks(np.arange(num_images))
            ax.set_yticklabels([f"Image {img}" for img in sorted(distances_data.keys())])

            # Add title
            ax.set_title(f"Subject: {subject}, Hand: {hand}")

            plt.tight_layout()
            plt.show()

# Example usage
plot_heatmap_by_subject(Block_Distance)







# Example metric for x-axis, can be 'durations', 'distance', 'speed', or 'accuracy', or 'motor_acuity'
# Example metric for y-axis, can be 'sparc', 'ldlj'

# # -------------------------------------------------------------------------------------------------------------------

# Scatter plot for correlation between two metrics for each reach type for one subject and specific hand
def plot_metric_correlation_hand(all_combined_metrics, subject, hand, metric_x, metric_y):
    num_reaches = 16
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for reach_index in range(num_reaches):
        ax = axes[reach_index]
        x_values = []
        y_values = []

        trials = all_combined_metrics[subject][hand][metric_x].keys()
        for trial in trials:
            trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

            # Ensure the reach index is valid
            if reach_index < len(trial_x) and reach_index < len(trial_y):
                x_values.append(trial_x[reach_index])
                y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Scatter plot
        ax.scatter(x_values, y_values, label=f'{hand.capitalize()} Hand', alpha=0.7)

        # Calculate and display Spearman correlation and p-value
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
            ax.text(0.05, 0.95, f"r: {correlation:.2f}\np: {p_value:.4f}", 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

        ax.set_title(f"Reach {reach_index + 1}")
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())

    fig.suptitle(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation for Subject: {subject}, Hand: {hand.capitalize()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_metric_correlation_hand(all_combined_metrics, '07/22/HW', 'right', 'durations', 'sparc')

# Scatter plot for correlation between two metrics for each reach type for multiple subjects and specific hand
def plot_metric_correlation_multi_subject(all_combined_metrics, subjects, hand, metric_x, metric_y):
    num_reaches = 16
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for reach_index in range(num_reaches):
        ax = axes[reach_index]
        x_values = []
        y_values = []

        for subject in subjects:
            trials = all_combined_metrics[subject][hand][metric_x].keys()
            for trial in trials:
                trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                # Ensure the reach index is valid
                if reach_index < len(trial_x) and reach_index < len(trial_y):
                    x_values.append(trial_x[reach_index])
                    y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Scatter plot
        ax.scatter(x_values, y_values, alpha=0.7)

        # Calculate and display Spearman correlation and p-value
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
            ax.text(0.05, 0.95, f"r: {correlation:.2f}\np: {p_value:.4f}", 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

        ax.set_title(f"Reach {reach_index + 1}")
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())

    fig.suptitle(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation for Multiple Subjects, Hand: {hand.capitalize()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_metric_correlation_multi_subject(all_combined_metrics, All_dates, 'right', 'durations', 'sparc')
# # -------------------------------------------------------------------------------------------------------------------

# Scatter plot for correlation between two metrics for each reach type for one subject and specific hand with hyperbolic fit
def plot_metric_correlation_hand(all_combined_metrics, subject, hand, metric_x, metric_y):
    num_reaches = 16
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for reach_index in range(num_reaches):
        ax = axes[reach_index]
        x_values = []
        y_values = []

        trials = all_combined_metrics[subject][hand][metric_x].keys()
        for trial in trials:
            trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

            # Ensure the reach index is valid
            if reach_index < len(trial_x) and reach_index < len(trial_y):
                x_values.append(trial_x[reach_index])
                y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Scatter plot
        ax.scatter(x_values, y_values, label=f'{hand.capitalize()} Hand', alpha=0.7)

        # Fit and plot hyperbolic model
        if len(x_values) > 1 and len(y_values) > 1:
            def hyperbolic(x, a, b):
                return a + b / x

            try:
                params, _ = curve_fit(hyperbolic, x_values, y_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(x_values), max(x_values), 100)
                fit_y = hyperbolic(fit_x, a, b)
                ax.plot(fit_x, fit_y, color='green', linestyle='--', label='Hyperbolic Fit')
            except RuntimeError:
                ax.text(0.05, 0.85, "Fit failed", transform=ax.transAxes, fontsize=8, color='red')

        # Calculate and display Spearman correlation and p-value
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
            ax.text(0.05, 0.95, f"r: {correlation:.2f}\np: {p_value:.4f}", 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

        ax.set_title(f"Reach {reach_index + 1}")
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())

    fig.suptitle(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation for Subject: {subject}, Hand: {hand.capitalize()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_metric_correlation_hand(all_combined_metrics, '07/22/HW', 'right', 'speed', 'sparc')

# Scatter plot for correlation between two metrics for each reach type for multiple subjects and specific hand with hyperbolic fit
def plot_metric_correlation_multi_subject(all_combined_metrics, subjects, hand, metric_x, metric_y):
    num_reaches = 16
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for reach_index in range(num_reaches):
        ax = axes[reach_index]
        x_values = []
        y_values = []

        for subject in subjects:
            trials = all_combined_metrics[subject][hand][metric_x].keys()
            for trial in trials:
                trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                # Ensure the reach index is valid
                if reach_index < len(trial_x) and reach_index < len(trial_y):
                    x_values.append(trial_x[reach_index])
                    y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Scatter plot
        ax.scatter(x_values, y_values, alpha=0.7)

        # Fit and plot hyperbolic model
        if len(x_values) > 1 and len(y_values) > 1:
            def hyperbolic(x, a, b):
                return a + b / x

            try:
                params, _ = curve_fit(hyperbolic, x_values, y_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(x_values), max(x_values), 100)
                fit_y = hyperbolic(fit_x, a, b)
                ax.plot(fit_x, fit_y, color='green', linestyle='--', label='Hyperbolic Fit')
            except RuntimeError:
                ax.text(0.05, 0.85, "Fit failed", transform=ax.transAxes, fontsize=8, color='red')

        # Calculate and display Spearman correlation and p-value
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
            ax.text(0.05, 0.95, f"r: {correlation:.2f}\np: {p_value:.4f}", 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

        ax.set_title(f"Reach {reach_index + 1}")
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())

    fig.suptitle(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation for Multiple Subjects, Hand: {hand.capitalize()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_metric_correlation_multi_subject(all_combined_metrics, All_dates, 'right', 'durations', 'sparc')

# # -------------------------------------------------------------------------------------------------------------------

# Scatter plot for correlation between two metrics for grouped reach types for one subject and specific hand with hyperbolic fit
def plot_metric_correlation_hand_grouped(all_combined_metrics, subject, hand, metric_x, metric_y):
    reach_groups = {
        "Group 1": [0, 4, 8, 12],
        "Group 2": [1, 5, 9, 13],
        "Group 3": [2, 6, 10, 14],
        "Group 4": [3, 7, 11, 15]
    }
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (group_name, reach_indices) in enumerate(reach_groups.items()):
        ax = axes[idx]
        x_values = []
        y_values = []

        trials = all_combined_metrics[subject][hand][metric_x].keys()
        for trial in trials:
            trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

            # Collect data for the grouped reach indices
            for reach_index in reach_indices:
                if reach_index < len(trial_x) and reach_index < len(trial_y):
                    x_values.append(trial_x[reach_index])
                    y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Scatter plot
        ax.scatter(x_values, y_values, label=f'{hand.capitalize()} Hand', alpha=0.7)

        # Fit and plot hyperbolic model
        if len(x_values) > 1 and len(y_values) > 1:
            def hyperbolic(x, a, b):
                return a + b / x

            try:
                params, _ = curve_fit(hyperbolic, x_values, y_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(x_values), max(x_values), 100)
                fit_y = hyperbolic(fit_x, a, b)
                ax.plot(fit_x, fit_y, color='green', linestyle='--', label='Hyperbolic Fit')
            except RuntimeError:
                ax.text(0.05, 0.85, "Fit failed", transform=ax.transAxes, fontsize=8, color='red')

        # Calculate and display Spearman correlation and p-value
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
            ax.text(0.05, 0.95, f"r: {correlation:.2f}\np: {p_value:.4f}", 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

        ax.set_title(f"{group_name}")
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())

    fig.suptitle(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation for Subject: {subject}, Hand: {hand.capitalize()} (Grouped)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_metric_correlation_hand_grouped(all_combined_metrics, '07/22/HW', 'left', 'accuracy', 'sparc')


# Scatter plot for correlation between two metrics for grouped reach types for multiple subjects and specific hand with hyperbolic fit
def plot_metric_correlation_multi_subject_grouped(all_combined_metrics, subjects, hand, metric_x, metric_y):
    reach_groups = {
        "Group 1": [0, 4, 8, 12],
        "Group 2": [1, 5, 9, 13],
        "Group 3": [2, 6, 10, 14],
        "Group 4": [3, 7, 11, 15]
    }
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (group_name, reach_indices) in enumerate(reach_groups.items()):
        ax = axes[idx]
        x_values = []
        y_values = []

        for subject in subjects:
            trials = all_combined_metrics[subject][hand][metric_x].keys()
            for trial in trials:
                trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                # Collect data for the grouped reach indices
                for reach_index in reach_indices:
                    if reach_index < len(trial_x) and reach_index < len(trial_y):
                        x_values.append(trial_x[reach_index])
                        y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Scatter plot
        ax.scatter(x_values, y_values, alpha=0.7)

        # Fit and plot hyperbolic model
        if len(x_values) > 1 and len(y_values) > 1:
            def hyperbolic(x, a, b):
                return a + b / x

            try:
                params, _ = curve_fit(hyperbolic, x_values, y_values, p0=(0, 1))
                a, b = params
                fit_x = np.linspace(min(x_values), max(x_values), 100)
                fit_y = hyperbolic(fit_x, a, b)
                ax.plot(fit_x, fit_y, color='green', linestyle='--', label='Hyperbolic Fit')
            except RuntimeError:
                ax.text(0.05, 0.85, "Fit failed", transform=ax.transAxes, fontsize=8, color='red')

        # Calculate and display Spearman correlation and p-value
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
            ax.text(0.05, 0.95, f"r: {correlation:.2f}\np: {p_value:.4f}", 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

        ax.set_title(f"{group_name}")
        ax.set_xlabel(metric_x.capitalize())
        ax.set_ylabel(metric_y.capitalize())

    fig.suptitle(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation for Multiple Subjects, Hand: {hand.capitalize()} (Grouped)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_metric_correlation_multi_subject_grouped(all_combined_metrics, All_dates, 'right', 'motor_acuity', 'sparc')

# # -------------------------------------------------------------------------------------------------------------------

# Scatter plot for correlation between two metrics for all reach types combined for one subject and specific hand
def plot_metric_correlation_hand_combined(all_combined_metrics, subject, hand, metric_x, metric_y):
    x_values = []
    y_values = []

    trials = all_combined_metrics[subject][hand][metric_x].keys()
    for trial in trials:
        trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
        trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

        # Collect data for all reach indices
        x_values.extend(trial_x)
        y_values.extend(trial_y)

    # Remove NaN values
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, label=f'{hand.capitalize()} Hand', alpha=0.7)

    # Fit and plot hyperbolic model
    if len(x_values) > 1 and len(y_values) > 1:
        def hyperbolic(x, a, b):
            return a + b / x

        try:
            params, _ = curve_fit(hyperbolic, x_values, y_values, p0=(0, 1))
            a, b = params
            fit_x = np.linspace(min(x_values), max(x_values), 100)
            fit_y = hyperbolic(fit_x, a, b)
            plt.plot(fit_x, fit_y, color='green', linestyle='--', label='Hyperbolic Fit')
        except RuntimeError:
            plt.text(0.05, 0.85, "Fit failed", transform=plt.gca().transAxes, fontsize=8, color='red')

    # Calculate and display Spearman correlation and p-value
    if len(x_values) > 1 and len(y_values) > 1:
        correlation, p_value = spearmanr(x_values, y_values)
        plt.text(0.05, 0.95, f"r: {correlation:.2f}\np: {p_value:.4f}", 
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.title(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation for Subject: {subject}, Hand: {hand.capitalize()} (Combined)", fontsize=14)
    plt.xlabel(metric_x.capitalize())
    plt.ylabel(metric_y.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_metric_correlation_hand_combined(all_combined_metrics, '07/22/HW', 'right', 'speed', 'sparc')


# Scatter plot for correlation between two metrics for all reach types combined for multiple subjects and specific hand
def plot_metric_correlation_multi_subject_combined(all_combined_metrics, subjects, hand, metric_x, metric_y):
    x_values = []
    y_values = []

    for subject in subjects:
        trials = all_combined_metrics[subject][hand][metric_x].keys()
        for trial in trials:
            trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

            # Collect data for all reach indices
            x_values.extend(trial_x)
            y_values.extend(trial_y)

    # Remove NaN values
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.7)

    # Fit and plot hyperbolic model
    if len(x_values) > 1 and len(y_values) > 1:
        def hyperbolic(x, a, b):
            return a + b / x

        try:
            params, _ = curve_fit(hyperbolic, x_values, y_values, p0=(0, 1))
            a, b = params
            fit_x = np.linspace(min(x_values), max(x_values), 100)
            fit_y = hyperbolic(fit_x, a, b)
            plt.plot(fit_x, fit_y, color='green', linestyle='--', label='Hyperbolic Fit')
        except RuntimeError:
            plt.text(0.05, 0.85, "Fit failed", transform=plt.gca().transAxes, fontsize=8, color='red')

    # Calculate and display Spearman correlation and p-value
    if len(x_values) > 1 and len(y_values) > 1:
        correlation, p_value = spearmanr(x_values, y_values)
        plt.text(0.05, 0.95, f"r: {correlation:.2f}\np: {p_value:.4f}", 
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.title(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation for Multiple Subjects, Hand: {hand.capitalize()} (Combined)", fontsize=14)
    plt.xlabel(metric_x.capitalize())
    plt.ylabel(metric_y.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_metric_correlation_multi_subject_combined(all_combined_metrics, All_dates, 'left', 'motor_acuity', 'sparc')

# # -------------------------------------------------------------------------------------------------------------------

# Heatmap for correlation between two metrics for each reach type for one subject and specific hand
def plot_metric_correlation_heatmap_hand(all_combined_metrics, subject, hand, metric_x, metric_y):
    num_reaches = 16
    correlation_matrix = np.full((4, 4), np.nan)
    p_value_matrix = np.full((4, 4), np.nan)

    for reach_index in range(num_reaches):
        x_values = []
        y_values = []

        trials = all_combined_metrics[subject][hand][metric_x].keys()
        for trial in trials:
            trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
            trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

            # Ensure the reach index is valid
            if reach_index < len(trial_x) and reach_index < len(trial_y):
                x_values.append(trial_x[reach_index])
                y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Calculate Spearman correlation and p-value
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
        else:
            correlation, p_value = np.nan, np.nan

        # Map reach index to 4x4 grid
        row, col = divmod(reach_index, 4)
        correlation_matrix[row, col] = correlation
        p_value_matrix[row, col] = p_value

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Annotate heatmap with correlation values and p-values
    for i in range(4):
        for j in range(4):
            value = correlation_matrix[i, j]
            p_value = p_value_matrix[i, j]
            if not np.isnan(value):
                color = 'red' if p_value < 0.05 else 'black'
                ax.text(j, i, f"{value:.2f}\n(p={p_value:.4f})", ha='center', va='center', color=color, fontsize=8)

    ax.set_title(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation\nSubject: {subject}, Hand: {hand.capitalize()}")
    ax.set_xlabel("Reach Index (Columns)")
    ax.set_ylabel("Reach Index (Rows)")
    plt.colorbar(im, ax=ax, label='Spearman Correlation')
    plt.tight_layout()
    plt.show()

plot_metric_correlation_heatmap_hand(all_combined_metrics, '07/22/HW', 'right', 'durations', 'sparc')

# Heatmap for correlation between two metrics for each reach type for multiple subjects and specific hand
def plot_metric_correlation_heatmap_multi_subject(all_combined_metrics, subjects, hand, metric_x, metric_y):
    num_reaches = 16
    correlation_matrix = np.full((4, 4), np.nan)
    p_value_matrix = np.full((4, 4), np.nan)

    for reach_index in range(num_reaches):
        x_values = []
        y_values = []

        for subject in subjects:
            trials = all_combined_metrics[subject][hand][metric_x].keys()
            for trial in trials:
                trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                # Ensure the reach index is valid
                if reach_index < len(trial_x) and reach_index < len(trial_y):
                    x_values.append(trial_x[reach_index])
                    y_values.append(trial_y[reach_index])

        # Remove NaN values
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Calculate Spearman correlation and p-value
        if len(x_values) > 1 and len(y_values) > 1:
            correlation, p_value = spearmanr(x_values, y_values)
        else:
            correlation, p_value = np.nan, np.nan

        # Map reach index to 4x4 grid
        row, col = divmod(reach_index, 4)
        correlation_matrix[row, col] = correlation
        p_value_matrix[row, col] = p_value

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Annotate heatmap with correlation values and p-values
    for i in range(4):
        for j in range(4):
            value = correlation_matrix[i, j]
            p_value = p_value_matrix[i, j]
            if not np.isnan(value):
                color = 'red' if p_value < 0.05 else 'black'
                ax.text(j, i, f"{value:.2f}\n(p={p_value:.4f})", ha='center', va='center', color=color, fontsize=8)

    ax.set_title(f"{metric_x.capitalize()} vs {metric_y.capitalize()} Correlation\nMultiple Subjects, Hand: {hand.capitalize()}")
    ax.set_xlabel("Reach Index (Columns)")
    ax.set_ylabel("Reach Index (Rows)")
    plt.colorbar(im, ax=ax, label='Spearman Correlation')
    plt.tight_layout()
    plt.show()

plot_metric_correlation_heatmap_multi_subject(all_combined_metrics, All_dates, 'right', 'distance', 'sparc')

# # -------------------------------------------------------------------------------------------------------------------

metrics_to_plot = [
    ('speed', 'sparc'),
    ('accuracy', 'sparc'),
    ('motor_acuity', 'sparc'),
    ('speed', 'ldlj'),
    ('accuracy', 'ldlj'),
    ('motor_acuity', 'ldlj')   
]
# --- Plot heatmaps for multiple metric correlations for a single subject and hand ---
def plot_metric_correlation_heatmap_hand_combined(all_combined_metrics, subject, hand, metrics):
    num_reaches = 16
    num_metrics = len(metrics)
    correlation_matrices = []
    p_value_matrices = []

    # Calculate correlation and p-value matrices for each metric pair
    for metric_x, metric_y in metrics:
        correlation_matrix = np.full((4, 4), np.nan)
        p_value_matrix = np.full((4, 4), np.nan)

        for reach_index in range(num_reaches):
            x_values = []
            y_values = []

            trials = all_combined_metrics[subject][hand][metric_x].keys()
            for trial in trials:
                trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                # Ensure the reach index is valid
                if reach_index < len(trial_x) and reach_index < len(trial_y):
                    x_values.append(trial_x[reach_index])
                    y_values.append(trial_y[reach_index])

            # Remove NaN values
            x_values = np.array(x_values)
            y_values = np.array(y_values)
            valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
            x_values = x_values[valid_indices]
            y_values = y_values[valid_indices]

            # Calculate Spearman correlation and p-value
            if len(x_values) > 1 and len(y_values) > 1:
                correlation, p_value = spearmanr(x_values, y_values)
            else:
                correlation, p_value = np.nan, np.nan

            # Map reach index to 4x4 grid
            row, col = divmod(reach_index, 4)
            correlation_matrix[row, col] = correlation
            p_value_matrix[row, col] = p_value

        correlation_matrices.append(correlation_matrix)
        p_value_matrices.append(p_value_matrix)

    # Plot all heatmaps in a single figure with subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 8), sharey=True)

    for idx, (ax, (metric_x, metric_y)) in enumerate(zip(axes, metrics)):
        im = ax.imshow(correlation_matrices[idx], cmap='coolwarm', vmin=-1, vmax=1)

        # Annotate heatmap with correlation values and p-values
        for i in range(4):
            for j in range(4):
                value = correlation_matrices[idx][i, j]
                p_value = p_value_matrices[idx][i, j]
                if not np.isnan(value):
                    color = 'red' if p_value < 0.05 else 'black'
                    ax.text(j, i, f"{value:.2f}\n(p={p_value:.4f})", ha='center', va='center', color=color, fontsize=8)

        ax.set_title(f"{metric_x.capitalize()} vs {metric_y.capitalize()}")
        ax.set_xlabel("Reach Index (Columns)")
        ax.set_xticks(range(4))
        ax.set_xticklabels(range(1, 5))
        if idx == 0:
            ax.set_ylabel(f"{subject}, {hand.capitalize()}\nReach Index (Rows)")
            ax.set_yticks(range(4))
            ax.set_yticklabels(range(1, 5))

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Spearman Correlation')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

plot_metric_correlation_heatmap_hand_combined(all_combined_metrics, '07/22/HW', 'left', metrics_to_plot)

# --- Plot heatmaps for multiple metric correlations for multiple subjects and hand ---
def plot_metric_correlation_heatmap_multi_subject_combined(all_combined_metrics, subjects, hand, metrics):
    num_reaches = 16
    num_metrics = len(metrics)
    correlation_matrices = []
    p_value_matrices = []

    # Calculate correlation and p-value matrices for each metric pair
    for metric_x, metric_y in metrics:
        correlation_matrix = np.full((4, 4), np.nan)
        p_value_matrix = np.full((4, 4), np.nan)

        for reach_index in range(num_reaches):
            x_values = []
            y_values = []

            for subject in subjects:
                trials = all_combined_metrics[subject][hand][metric_x].keys()
                for trial in trials:
                    trial_x = np.array(all_combined_metrics[subject][hand][metric_x][trial])
                    trial_y = np.array(all_combined_metrics[subject][hand][metric_y][trial])

                    # Ensure the reach index is valid
                    if reach_index < len(trial_x) and reach_index < len(trial_y):
                        x_values.append(trial_x[reach_index])
                        y_values.append(trial_y[reach_index])

            # Remove NaN values
            x_values = np.array(x_values)
            y_values = np.array(y_values)
            valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
            x_values = x_values[valid_indices]
            y_values = y_values[valid_indices]

            # Calculate Spearman correlation and p-value
            if len(x_values) > 1 and len(y_values) > 1:
                correlation, p_value = spearmanr(x_values, y_values)
            else:
                correlation, p_value = np.nan, np.nan

            # Map reach index to 4x4 grid
            row, col = divmod(reach_index, 4)
            correlation_matrix[row, col] = correlation
            p_value_matrix[row, col] = p_value

        correlation_matrices.append(correlation_matrix)
        p_value_matrices.append(p_value_matrix)

    # Plot all heatmaps in a single figure with subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 8), sharey=True)

    for idx, (ax, (metric_x, metric_y)) in enumerate(zip(axes, metrics)):
        im = ax.imshow(correlation_matrices[idx], cmap='coolwarm', vmin=-1, vmax=1)

        # Annotate heatmap with correlation values and p-values
        for i in range(4):
            for j in range(4):
                value = correlation_matrices[idx][i, j]
                p_value = p_value_matrices[idx][i, j]
                if not np.isnan(value):
                    color = 'red' if p_value < 0.05 else 'black'
                    ax.text(j, i, f"{value:.2f}\n(p={p_value:.4f})", ha='center', va='center', color=color, fontsize=8)

        ax.set_title(f"{metric_x.capitalize()} vs {metric_y.capitalize()}")
        ax.set_xlabel("Reach Index (Columns)")
        ax.set_xticks(range(4))
        ax.set_xticklabels(range(1, 5))
        if idx == 0:
            ax.set_ylabel(f"Subjects Combined, {hand.capitalize()}\nReach Index (Rows)")
            ax.set_yticks(range(4))
            ax.set_yticklabels(range(1, 5))

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Spearman Correlation')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

plot_metric_correlation_heatmap_multi_subject_combined(all_combined_metrics, All_dates, 'right', metrics_to_plot)

# # -------------------------------------------------------------------------------------------------------------------














# # Calculate total data points for reach durations and block accuracy
# total_reach_durations = sum(len(reach_metrics['reach_durations'][subject][hand]) 
#                             for subject in reach_metrics['reach_durations'] 
#                             for hand in reach_metrics['reach_durations'][subject])

# total_block_accuracy = sum(len(Block_Distance[subject][hand]) 
#                            for subject in Block_Distance 
#                            for hand in Block_Distance[subject])

# print(f"Total data points in reach durations: {total_reach_durations}")
# print(f"Total data points in block accuracy: {total_block_accuracy}")



























# # PART 3: 

# # SPARC 
# # Plot a heatmap of SPARC values for a specific subject and hand, with outliers annotated.
# utils3.plot_sparc_heatmap_with_outliers(reach_sparc_test_windows_1, '07/23/AK', 'right')

# # Plot a heatmap of actual average SPARC values across trials for all subjects and a specific hand,
# # with an additional subplot for the average across all subjects.
# utils3.plot_average_sparc_value_heatmap_with_subject_average(reach_sparc_test_windows_1, 'right')

# # Plot a heatmap of ranked SPARC values for a specific subject and hand.
# utils3.plot_ranked_sparc_heatmap(reach_sparc_test_windows_1, '07/23/AK', 'right')

# # Plot a heatmap of average ranked SPARC values across trials across all subjects for a specific hand.
# utils3.plot_average_sparc_ranking_heatmap_across_all_dates(reach_sparc_test_windows_1, 'left')

# # LDLJ
# # Plot a heatmap of actual average LDLJ values across trials for all subjects and a specific hand,
# # with an additional subplot for the average across all subjects.
# utils3.plot_average_ldlj_value_heatmap_with_subject_average(reach_TW_metrics, 'left')

# # Plot a heatmap of average ranked LDLJ values across trials across all subjects for a specific hand.
# utils3.plot_average_ldlj_ranking_heatmap_across_all_dates(reach_TW_metrics, 'right')


# # Plot a heatmap of overall average SPARC values for a specific hand, rearranged into a 4x4 grid.
# utils3.plot_average_sparc_value_heatmap_with_subject_average_4x4(reach_sparc_test_windows_1, 'left')

# # Plot a heatmap of overall average LDLJ values for a specific hand, rearranged into a 4x4 grid.
# utils3.plot_average_ldlj_value_heatmap_with_subject_average_4x4(reach_TW_metrics, 'left')

# # Plot a violin plot of SPARC values for all participants, each participant in one color, excluding outliers.
# def plot_sparc_violin_all_participants_no_outliers(reach_sparc_test_windows, hand):
#     # Prepare data for all participants
#     all_data = []
#     all_labels = []
#     all_colors = []
#     participants = list(reach_sparc_test_windows.keys())
#     color_palette = sns.color_palette("husl", len(participants))

#     for idx, participant in enumerate(participants):
#         sparc_data = reach_sparc_test_windows[participant][hand]
#         sparc_matrix = np.array([values for values in sparc_data.values()])
#         trials, reaches = sparc_matrix.shape

#         # Remove outliers using z-score
#         sparc_matrix = sparc_matrix[(np.abs(zscore(sparc_matrix, axis=None)) < 3).all(axis=1)]

#         # Append data and labels
#         for reach_idx in range(reaches):
#             reach_values = sparc_matrix[:, reach_idx]
#             all_data.extend(reach_values)
#             all_labels.extend([reach_idx + 1] * len(reach_values))
#             all_colors.extend([color_palette[idx]] * len(reach_values))

#     # Create DataFrame for plotting
#     data = pd.DataFrame({
#         'SPARC Values': all_data,
#         'Reach': all_labels,
#         'Participant': all_colors
#     })

#     # Plot violin plot
#     plt.figure(figsize=(12, 8))
#     sns.violinplot(
#         x='Reach', 
#         y='SPARC Values', 
#         data=data, 
#         inner=None, 
#         scale='width', 
#         palette='light:#d3d3d3'  # Light grey fill for violins
#     )

#     # Overlay value dots for each participant
#     for idx, participant in enumerate(participants):
#         sparc_data = reach_sparc_test_windows[participant][hand]
#         sparc_matrix = np.array([values for values in sparc_data.values()])
#         trials, reaches = sparc_matrix.shape

#         # Remove outliers using z-score
#         sparc_matrix = sparc_matrix[(np.abs(zscore(sparc_matrix, axis=None)) < 3).all(axis=1)]

#         for reach_idx in range(reaches):
#             reach_values = sparc_matrix[:, reach_idx]
#             plt.scatter(
#                 [reach_idx] * len(reach_values),
#                 reach_values,
#                 c=[color_palette[idx]] * len(reach_values),
#                 alpha=0.6,
#                 label=participant if reach_idx == 0 else ""
#             )

#     plt.xlabel('Reach')
#     plt.ylabel('SPARC Values')
#     plt.title(f'Violin Plot of SPARC Values for {hand.capitalize()} Hand (All Participants, No Outliers)')
#     plt.legend(title='Participants', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()

# plot_sparc_violin_all_participants_no_outliers(reach_sparc_test_windows_1, 'left')


# # Plot a violin plot of LDLJ values for all participants, each participant in one color, excluding outliers.
# def plot_ldlj_violin_all_participants_no_outliers(reach_TW_metrics, hand):
#     # Prepare data for all participants
#     all_data = []
#     all_labels = []
#     all_colors = []
#     participants = list(reach_TW_metrics['reach_LDLJ'].keys())
#     color_palette = sns.color_palette("husl", len(participants))

#     for idx, participant in enumerate(participants):
#         ldlj_data = reach_TW_metrics['reach_LDLJ'][participant][hand]
#         ldlj_matrix = np.array([values for values in ldlj_data.values()])
#         trials, reaches = ldlj_matrix.shape

#         # Remove outliers using z-score
#         ldlj_matrix = ldlj_matrix[(np.abs(zscore(ldlj_matrix, axis=None)) < 3).all(axis=1)]

#         # Append data and labels
#         for reach_idx in range(reaches):
#             reach_values = ldlj_matrix[:, reach_idx]
#             all_data.extend(reach_values)
#             all_labels.extend([reach_idx + 1] * len(reach_values))
#             all_colors.extend([color_palette[idx]] * len(reach_values))

#     # Create DataFrame for plotting
#     data = pd.DataFrame({
#         'LDLJ Values': all_data,
#         'Reach': all_labels,
#         'Participant': all_colors
#     })

#     # Plot violin plot
#     plt.figure(figsize=(12, 8))
#     sns.violinplot(
#         x='Reach', 
#         y='LDLJ Values', 
#         data=data, 
#         inner=None, 
#         scale='width', 
#         palette='light:#d3d3d3'  # Light grey fill for violins
#     )

#     # Overlay value dots for each participant
#     for idx, participant in enumerate(participants):
#         ldlj_data = reach_TW_metrics['reach_LDLJ'][participant][hand]
#         ldlj_matrix = np.array([values for values in ldlj_data.values()])
#         trials, reaches = ldlj_matrix.shape

#         # Remove outliers using z-score
#         ldlj_matrix = ldlj_matrix[(np.abs(zscore(ldlj_matrix, axis=None)) < 3).all(axis=1)]

#         for reach_idx in range(reaches):
#             reach_values = ldlj_matrix[:, reach_idx]
#             plt.scatter(
#                 [reach_idx] * len(reach_values),
#                 reach_values,
#                 c=[color_palette[idx]] * len(reach_values),
#                 alpha=0.6,
#                 label=participant if reach_idx == 0 else ""
#             )

#     plt.xlabel('Reach')
#     plt.ylabel('LDLJ Values')
#     plt.title(f'Violin Plot of LDLJ Values for {hand.capitalize()} Hand (All Participants, No Outliers)')
#     plt.legend(title='Participants', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()

# plot_ldlj_violin_all_participants_no_outliers(reach_TW_metrics, 'right')

# # Plot mean and standard deviation of LDLJ values for each subject and hand as separate box plots for each hand
# def plot_subject_hand_ldlj_boxplot_with_data_points(reach_TW_metrics):
#     hands = ['right', 'left']
#     data = []

#     for hand in hands:
#         all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
#         for subject in all_subjects:
#             if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
#                 continue
#             ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
            
#             # Collect all data points, mean, and standard deviation for each subject and hand
#             mean_value = ldlj_matrix.mean()
#             sd_value = ldlj_matrix.std()
#             for value in ldlj_matrix.flatten():
#                 data.append({'Hand': hand.capitalize(), 'LDLJ Value': value, 'Mean LDLJ': mean_value, 'SD': sd_value, 'Subject': subject})

#     df = pd.DataFrame(data)

#     # Create subplots for left and right hands
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
#     for ax, hand in zip(axes, hands):
#         sns.boxplot(
#             ax=ax, x='Subject', y='LDLJ Value', 
#             data=df[df['Hand'] == hand.capitalize()], 
#             palette='Set2', showfliers=False
#         )
#         sns.stripplot(
#             ax=ax, x='Subject', y='LDLJ Value', 
#             data=df[df['Hand'] == hand.capitalize()], 
#             dodge=True, alpha=0.6, marker='o', size=8, palette='husl'
#         )
        
#         # Add error bars for standard deviation and annotate mean values
#         for i, subject in enumerate(df[df['Hand'] == hand.capitalize()]['Subject'].unique()):
#             mean = df[(df['Hand'] == hand.capitalize()) & (df['Subject'] == subject)]['Mean LDLJ'].values[0]
#             sd = df[(df['Hand'] == hand.capitalize()) & (df['Subject'] == subject)]['SD'].values[0]
#             ax.errorbar(
#                 x=i, y=mean, yerr=sd, fmt='none', c='black', capsize=5, label='SD' if i == 0 else ""
#             )
#             ax.text(
#                 x=i, y=mean, s=f'{mean:.2f}', color='black', ha='center', va='bottom', fontsize=10
#             )

#         ax.set_title(f'LDLJ Values for {hand.capitalize()} Hand (Including Data Points)', fontsize=14)
#         ax.set_xlabel('Subject', fontsize=12)
#         ax.set_ylabel('LDLJ Value', fontsize=12)
#         ax.legend(loc='upper right')

#     plt.tight_layout()
#     plt.show()

# plot_subject_hand_ldlj_boxplot_with_data_points(reach_TW_metrics)

# # Plot mean and standard deviation of LDLJ values for each subject and hand as separate box plots for each hand
# # Averaging across specific reach groups: (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (4, 8, 12, 16)
# def plot_subject_hand_ldlj_boxplot_with_grouped_means(reach_TW_metrics):
#     hands = ['right', 'left']
#     data = []

#     for hand in hands:
#         all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
#         for subject in all_subjects:
#             if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
#                 continue
#             ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
            
#             # Group reaches into (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (4, 8, 12, 16)
#             grouped_means = []
#             for group in range(4):
#                 group_indices = range(group, ldlj_matrix.shape[1], 4)
#                 group_values = ldlj_matrix[:, group_indices].flatten()
#                 grouped_means.append(group_values.mean())

#             # Collect grouped means and standard deviations for each subject and hand
#             for group_idx, mean_value in enumerate(grouped_means, start=1):
#                 data.append({
#                     'Hand': hand.capitalize(),
#                     'Group': f'Group {group_idx}',
#                     'Mean LDLJ': mean_value,
#                     'Subject': subject
#                 })

#     df = pd.DataFrame(data)

#     # Create subplots for left and right hands
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
#     for ax, hand in zip(axes, hands):
#         sns.boxplot(
#             ax=ax, x='Group', y='Mean LDLJ', 
#             data=df[df['Hand'] == hand.capitalize()], 
#             palette='Set2', showfliers=False
#         )
#         sns.stripplot(
#             ax=ax, x='Group', y='Mean LDLJ', 
#             data=df[df['Hand'] == hand.capitalize()], 
#             dodge=True, alpha=0.6, marker='o', size=8, hue='Subject', palette='husl'
#         )
        
#         ax.set_title(f'LDLJ Grouped Means for {hand.capitalize()} Hand', fontsize=14)
#         ax.set_xlabel('Reach Group', fontsize=12)
#         ax.set_ylabel('Mean LDLJ Value', fontsize=12)
#         ax.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

#     plt.tight_layout()
#     plt.show()

# plot_subject_hand_ldlj_boxplot_with_grouped_means(reach_TW_metrics)

# # Plot a heatmap of overall average LDLJ values for a specific hand, averaging across columns and showing as a subplot.
# def plot_average_ldlj_value_heatmap_with_column_average(reach_TW_metrics, hand):
#     all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
#     all_average_values = []

#     for subject in all_subjects:
#         if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
#             continue
#         ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
#         all_average_values.append(ldlj_matrix.mean(axis=0))

#     all_average_values = np.array(all_average_values)
#     overall_average = all_average_values.mean(axis=0)

#     # Rearrange the overall average into the specified order
#     if hand.lower() == 'right':
#         rearranged_indices = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
#     elif hand.lower() == 'left':
#         rearranged_indices = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
#     else:
#         raise ValueError("Invalid hand specified. Use 'right' or 'left'.")

#     rearranged_average = overall_average[rearranged_indices]

#     # Reshape the rearranged average into a 4x4 grid
#     grid_size = 4
#     rearranged_average_reshaped = rearranged_average.reshape(grid_size, grid_size)

#     # Calculate column averages
#     column_averages = rearranged_average_reshaped.mean(axis=0)
#     print(column_averages)

#     # Create subplots for the heatmap and column averages
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})

#     # Heatmap subplot
#     ax1 = axes[0]
#     im = ax1.imshow(rearranged_average_reshaped, aspect='auto', cmap='viridis', interpolation='nearest')
#     ax1.set_title(f"Overall Average LDLJ Values for Hand: {hand.capitalize()}", fontsize=14, fontweight='bold')
#     ax1.set_xticks(range(grid_size))
#     ax1.set_xticklabels(range(1, grid_size + 1))
#     ax1.set_yticks(range(grid_size))
#     ax1.set_yticklabels(range(1, grid_size + 1))
#     ax1.set_xlabel("Reach Index (Columns)")
#     ax1.set_ylabel("Reach Index (Rows)")
#     plt.colorbar(im, ax=ax1, orientation='vertical', label='LDLJ Values')

#     # Annotate the heatmap with LDLJ values
#     for i in range(grid_size):
#         for j in range(grid_size):
#             ax1.text(j, i, f'{rearranged_average_reshaped[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

#     # Column averages subplot
#     ax2 = axes[1]
#     ax2.barh(range(grid_size), column_averages, color='skyblue')
#     ax2.set_yticks(range(grid_size))
#     ax2.set_yticklabels(range(1, grid_size + 1))
#     ax2.set_xlabel("Average LDLJ Value")
#     ax2.set_title("Column Averages", fontsize=12, fontweight='bold')

#     plt.tight_layout()
#     plt.show()

# plot_average_ldlj_value_heatmap_with_column_average(reach_TW_metrics, 'left')

# # Calculate mean, median, IQR, and SD for both hands and return it as sparc_parameters
# def calculate_sparc_statistics(reach_sparc_test_windows):
#     sparc_parameters = {}
#     participants = list(reach_sparc_test_windows.keys())
#     hands = ['right', 'left']

#     for participant in participants:
#         sparc_parameters[participant] = {}
#         for hand in hands:
#             sparc_data = reach_sparc_test_windows[participant][hand]
#             sparc_matrix = np.array([values for values in sparc_data.values()])
            
#             # Remove outliers using z-score
#             sparc_matrix = sparc_matrix[(np.abs(zscore(sparc_matrix, axis=None)) < 3).all(axis=1)]
            
#             # Flatten the matrix to calculate statistics
#             sparc_values = sparc_matrix.flatten()
            
#             # Calculate statistics
#             mean = np.mean(sparc_values)
#             median = np.median(sparc_values)
#             iqr = np.percentile(sparc_values, 75) - np.percentile(sparc_values, 25)
#             sd = np.std(sparc_values)
#             q1 = np.percentile(sparc_values, 25)
#             q3 = np.percentile(sparc_values, 75)
            
#             # Store statistics for the participant and hand
#             sparc_parameters[participant][hand] = {
#                 'mean': mean,
#                 'median': median,
#                 'iqr': iqr,
#                 'sd': sd,
#                 'q1': q1,
#                 'q3': q3
#             }
    
#     return sparc_parameters

# sparc_parameters = calculate_sparc_statistics(reach_sparc_test_windows_1)

# # Calculate mean, median, IQR, and SD for both hands and return it as ldlj_parameters
# def calculate_ldlj_statistics(reach_TW_metrics):
#     ldlj_parameters = {}
#     participants = list(reach_TW_metrics['reach_LDLJ'].keys())
#     hands = ['right', 'left']

#     for participant in participants:
#         ldlj_parameters[participant] = {}
#         for hand in hands:
#             ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][participant][hand].values()])
            
#             # Remove outliers using z-score
#             ldlj_matrix = ldlj_matrix[(np.abs(zscore(ldlj_matrix, axis=None)) < 3).all(axis=1)]
            
#             # Flatten the matrix to calculate statistics
#             ldlj_values = ldlj_matrix.flatten()
            
#             # Calculate statistics
#             mean = np.mean(ldlj_values)
#             median = np.median(ldlj_values)
#             iqr = np.percentile(ldlj_values, 75) - np.percentile(ldlj_values, 25)
#             sd = np.std(ldlj_values)
#             q1 = np.percentile(ldlj_values, 25)
#             q3 = np.percentile(ldlj_values, 75)
            
#             # Store statistics for the participant and hand
#             ldlj_parameters[participant][hand] = {
#                 'mean': mean,
#                 'median': median,
#                 'iqr': iqr,
#                 'sd': sd,
#                 'q1': q1,
#                 'q3': q3
#             }
    
#     return ldlj_parameters

# ldlj_parameters = calculate_ldlj_statistics(reach_TW_metrics)

# # Plot mean and standard deviation for SPARC and LDLJ for both hands as box plots with paired t-test
# def plot_mean_sd_boxplot_with_ttest(sparc_parameters, ldlj_parameters):
#     data = []
#     for metric, parameters in [('SPARC', sparc_parameters), ('LDLJ', ldlj_parameters)]:
#         for participant, hand_data in parameters.items():
#             for hand, stats in hand_data.items():
#                 data.append({
#                     'Metric': metric,
#                     'Hand': hand.capitalize(),
#                     'Mean': stats['mean'],
#                     'SD': stats['sd'],
#                     'Participant': participant
#                 })

#     df = pd.DataFrame(data)
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8))
#     palette = sns.color_palette("husl", len(df['Participant'].unique()))
#     participant_colors = {p: palette[i] for i, p in enumerate(df['Participant'].unique())}

#     for ax, metric in zip(axes, ['SPARC', 'LDLJ']):
#         sns.boxplot(ax=ax, x='Hand', y='Mean', data=df[df['Metric'] == metric], palette='Set2', showfliers=False)
#         sns.stripplot(
#             ax=ax, x='Hand', y='Mean', data=df[df['Metric'] == metric], dodge=True, 
#             palette=participant_colors, alpha=0.6, marker='o', hue='Participant', size=8
#         )
#         ax.set_title(f'Mean Values of {metric} for Both Hands')
#         ax.set_ylabel(f'{metric} Mean Value')
#         ax.set_ylim(df[df['Metric'] == metric]['Mean'].min() - 0.1, df[df['Metric'] == metric]['Mean'].max() + 0.1)
#         ax.set_xlabel('Hand')
#         ax.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

#         # Perform paired t-test between hands
#         right_hand_means = df[(df['Metric'] == metric) & (df['Hand'] == 'Right')]['Mean']
#         left_hand_means = df[(df['Metric'] == metric) & (df['Hand'] == 'Left')]['Mean']
#         t_stat, p_value = ttest_rel(right_hand_means, left_hand_means)

#         # Annotate t-test result on the plot title
#         ax.set_title(f'Mean Values of {metric} for Both Hands\nPaired t-test: t={t_stat:.2f}, p={p_value:.3e}')

#     plt.tight_layout()
#     plt.show()

# plot_mean_sd_boxplot_with_ttest(sparc_parameters, ldlj_parameters)


# # SPARC vs LDLJ Correlation
# # Plot scatter plot of LDLJ vs. SPARC values for a specific subject and hand
# utils3.plot_ldlj_sparc_correlation_by_trial(
#     reach_TW_metrics=reach_TW_metrics,
#     reach_sparc_test_windows_1=reach_sparc_test_windows_1,
#     subject='07/23/AK',
#     hand='right'
# )

# # Plot scatter plot of LDLJ vs. SPARC values for a specific subject and hand
# utils3.plot_ldlj_sparc_scatter_by_trial(
#     reach_TW_metrics=reach_TW_metrics,
#     reach_sparc_test_windows_1=reach_sparc_test_windows_1,
#     subject='07/23/AK',
#     hand='right'
# )

# # --- PLOT EACH SEGMENT SPEED AS SUBPLOT WITH LDLJ AND SPARC VALUES ---
# utils3.plot_all_segments_with_ldlj_and_sparc(Figure_folder, test_windows_6, results, reach_TW_metrics, reach_sparc_test_windows_1)

# # Plot scatter plot of LDLJ vs. SPARC values for a specific subject and hand
# utils3.plot_ldlj_sparc_correlation_by_trial(
#     reach_TW_metrics=reach_TW_metrics,
#     reach_sparc_test_windows_1=reach_sparc_test_windows_1,
#     subject='07/23/AK',
#     hand='right'
# )

# # Plot scatter plot of LDLJ vs. SPARC values for a specific subject and hand
# utils3.plot_ldlj_sparc_scatter_by_trial(
#     reach_TW_metrics=reach_TW_metrics,
#     reach_sparc_test_windows_1=reach_sparc_test_windows_1,
#     subject='07/23/AK',
#     hand='right'
# )





