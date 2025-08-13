import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from scipy.stats import ttest_1samp
from scipy.stats import spearmanr
import seaborn as sns

# --- SPARCS PLOT ---

# Plot a heatmap of SPARC values for a specific subject and hand, with outliers annotated.
def plot_sparc_heatmap_with_outliers(reach_sparc_test_windows, subject, hand):

    # Extract SPARC values for all trials in the specified hand
    sparc_data = reach_sparc_test_windows[subject][hand]
    sparc_matrix = np.array([values for values in sparc_data.values()])

    # Calculate outlier thresholds
    lower_thresh = np.percentile(sparc_matrix, 1)
    upper_thresh = np.percentile(sparc_matrix, 99)

    # Plot heatmap with color scaling clipped to 1st–99th percentile
    plt.figure(figsize=(10, 8))
    im = plt.imshow(sparc_matrix, aspect='auto', cmap='viridis', interpolation='nearest',
                    vmin=lower_thresh, vmax=upper_thresh)
    plt.colorbar(label='SPARC Values')
    plt.xlabel('Reach')
    plt.xticks(ticks=range(sparc_matrix.shape[1]), labels=range(1, sparc_matrix.shape[1] + 1))
    plt.ylabel('Trials')
    plt.title(f'Heatmap of SPARC Values for {hand.capitalize()} Hand ({subject})')

    # Annotate outliers
    for i in range(sparc_matrix.shape[0]):        # For each trial (row)
        for j in range(sparc_matrix.shape[1]):    # For each time point (column)
            value = sparc_matrix[i, j]
            if value < lower_thresh or value > upper_thresh:
                plt.text(j, i, f'{value:.1f}', ha='center', va='center', color='red', fontsize=6)

    plt.tight_layout()
    plt.show()

# Plot a heatmap of ranked SPARC values for a specific subject and hand.
def plot_ranked_sparc_heatmap(reach_sparc_test_windows, subject, hand):
    import matplotlib.pyplot as plt

    # Extract SPARC values for all trials in the specified hand
    hand_sparc = reach_sparc_test_windows[subject][hand]

    # Combine SPARC values into a 2D array for heatmap
    sparc_matrix = np.array([values for values in hand_sparc.values()])

    # Rank normalization across each row
    sparc_matrix_ranked = np.argsort(np.argsort(sparc_matrix, axis=1), axis=1) + 1

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(sparc_matrix_ranked, aspect='auto', cmap='viridis', interpolation='nearest')

    # Add colorbar, labels, and title
    plt.colorbar(label='Ranked SPARC Values')
    plt.xlabel('Reach')
    plt.xticks(ticks=range(sparc_matrix.shape[1]), labels=range(1, sparc_matrix.shape[1] + 1))
    plt.ylabel('Trials')
    plt.title(f'Heatmap of Ranked SPARC Values for {hand.capitalize()} Hand ({subject})')

    # Show the heatmap
    plt.tight_layout()
    plt.show()

# # Plot a heatmap of average ranked SPARC values across trials for the right hand.
# def plot_average_sparc_ranking_heatmap(right_hand_sparc):
#     import matplotlib.pyplot as plt

#     # Combine SPARC values into a 2D array for heatmap
#     sparc_matrix = np.array([values for values in right_hand_sparc.values()])

#     # Rank normalization across each row
#     sparc_matrix_ranked = np.argsort(np.argsort(sparc_matrix, axis=1), axis=1) + 1

#     # Calculate the average ranking across trials
#     average_ranking = sparc_matrix_ranked.mean(axis=0)

#     # Create a heatmap for the average ranking
#     plt.figure(figsize=(10, 6))
#     plt.imshow(average_ranking[np.newaxis, :], aspect='auto', cmap='viridis', interpolation='nearest')

#     # Add colorbar, labels, and title
#     plt.colorbar(label='Average Ranked SPARC Values')
#     plt.xlabel('Reach')
#     plt.yticks([])  # Remove y-axis ticks since it's a single row
#     plt.xticks(ticks=range(sparc_matrix.shape[1]), labels=range(1, sparc_matrix.shape[1] + 1))
#     plt.title('Heatmap of Average Ranked SPARC Values Across Trials')

#     # Show the heatmap
#     plt.tight_layout()
#     plt.show()

# # Plot a heatmap of average ranked SPARC values across trials for all subjects and a specific hand.
# def plot_average_sparc_ranking_heatmap_for_all_dates(reach_sparc_test_windows, hand):
#     import matplotlib.pyplot as plt
#     all_subjects = list(reach_sparc_test_windows.keys())
#     all_average_rankings = []

#     for subject in all_subjects:
#         if hand not in reach_sparc_test_windows[subject]:
#             continue
#         sparc_matrix = np.array([values for values in reach_sparc_test_windows[subject][hand].values()])
#         sparc_matrix_ranked = np.argsort(np.argsort(sparc_matrix, axis=1), axis=1) + 1
#         all_average_rankings.append(sparc_matrix_ranked.mean(axis=0))

#     plt.figure(figsize=(12, 0.5 * len(all_average_rankings)))
#     plt.imshow(np.array(all_average_rankings), aspect='auto', cmap='viridis', interpolation='nearest')
#     cbar = plt.colorbar(label='Average Ranked SPARC Values')
#     plt.xlabel('Reach')
#     plt.ylabel('Subjects')
#     plt.yticks(ticks=range(len(all_subjects)), labels=all_subjects)
#     plt.xticks(ticks=range(sparc_matrix.shape[1]), labels=range(1, sparc_matrix.shape[1] + 1))
#     plt.title(f'Heatmap of Average Ranked SPARC Values for Hand: {hand}')
#     plt.tight_layout()
#     plt.show()

# Plot a heatmap of average ranked SPARC values across trials across all subjects for a specific hand.
def plot_average_sparc_ranking_heatmap_across_all_dates(reach_sparc_test_windows, hand):
    import matplotlib.pyplot as plt
    all_subjects = list(reach_sparc_test_windows.keys())
    all_average_rankings = []

    for subject in all_subjects:
        if hand not in reach_sparc_test_windows[subject]:
            continue
        sparc_matrix = np.array([values for values in reach_sparc_test_windows[subject][hand].values()])
        sparc_matrix_ranked = np.argsort(np.argsort(sparc_matrix, axis=1), axis=1) + 1
        all_average_rankings.append(sparc_matrix_ranked.mean(axis=0))

    # Plot heatmap of average rankings for all subjects
    plt.figure(figsize=(12, 0.5 * len(all_average_rankings) + 2))
    plt.subplot(2, 1, 1)
    plt.imshow(np.array(all_average_rankings), aspect='auto', cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar(label='Average Ranked SPARC Values')
    plt.xlabel('Reach')
    plt.ylabel('Subjects')
    plt.yticks(ticks=range(len(all_subjects)), labels=all_subjects)
    plt.xticks(ticks=range(sparc_matrix.shape[1]), labels=range(1, sparc_matrix.shape[1] + 1))
    plt.title(f'Heatmap of Average Ranked SPARC Values for Hand: {hand}')

    # Plot average across subjects as a heatmap
    plt.subplot(2, 1, 2)
    overall_average = np.mean(all_average_rankings, axis=0)
    im = plt.imshow(overall_average[np.newaxis, :], aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Average Ranked SPARC Values')
    plt.xlabel('Reach')
    plt.yticks([])  # Remove y-axis ticks since it's a single row
    plt.title('Heatmap of Average Ranked SPARC Values Across Subjects')

    # Annotate values in subplot 2
    for i, value in enumerate(overall_average):
        plt.text(i, 0, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.show()

# Plot a heatmap of actual average SPARC values across trials for all subjects and a specific hand,
# with an additional subplot for the average across all subjects.
def plot_average_sparc_value_heatmap_with_subject_average(reach_sparc_test_windows, hand):
    import matplotlib.pyplot as plt
    all_subjects = list(reach_sparc_test_windows.keys())
    all_average_values = []

    for subject in all_subjects:
        if hand not in reach_sparc_test_windows[subject]:
            continue
        sparc_matrix = np.array([values for values in reach_sparc_test_windows[subject][hand].values()])
        all_average_values.append(sparc_matrix.mean(axis=0))

    all_average_values = np.array(all_average_values)
    overall_average = all_average_values.mean(axis=0)

    plt.figure(figsize=(12, 0.5 * len(all_average_values) + 2))

    # Heatmap for all subjects
    plt.subplot(2, 1, 1)
    plt.imshow(all_average_values, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Actual Average SPARC Values')
    plt.xlabel('Reach')
    plt.ylabel('Subjects')
    plt.yticks(ticks=range(len(all_subjects)), labels=all_subjects)
    plt.xticks(ticks=range(all_average_values.shape[1]), labels=range(1, all_average_values.shape[1] + 1))
    plt.title(f'Heatmap of Actual Average SPARC Values for Hand: {hand}')

    # Heatmap for overall average across subjects
    plt.subplot(2, 1, 2)
    im = plt.imshow(overall_average[np.newaxis, :], aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Overall Average SPARC Values')
    plt.xlabel('Reach')
    plt.yticks([])  # Remove y-axis ticks since it's a single row
    plt.xticks(ticks=range(overall_average.shape[0]), labels=range(1, overall_average.shape[0] + 1))
    plt.title('Overall Average SPARC Values Across Subjects')

    # Annotate values in subplot 2
    for i, value in enumerate(overall_average):
        plt.text(i, 0, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.show()


# Plot a heatmap of overall average SPARC values for a specific hand, rearranged into a 4x4 grid.
def plot_average_sparc_value_heatmap_with_subject_average_4x4(reach_sparc_test_windows, hand):
    all_subjects = list(reach_sparc_test_windows.keys())
    all_average_values = []

    for subject in all_subjects:
        if hand not in reach_sparc_test_windows[subject]:
            continue
        sparc_matrix = np.array([values for values in reach_sparc_test_windows[subject][hand].values()])
        all_average_values.append(sparc_matrix.mean(axis=0))

    all_average_values = np.array(all_average_values)
    overall_average = all_average_values.mean(axis=0)

    # Rearrange the overall average into the specified order
    if hand.lower() == 'right':
        rearranged_indices = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    elif hand.lower() == 'left':
        rearranged_indices = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    else:
        raise ValueError("Invalid hand specified. Use 'right' or 'left'.")

    rearranged_average = overall_average[rearranged_indices]

    # Reshape the rearranged average into a 4x4 grid
    grid_size = 4
    rearranged_average_reshaped = rearranged_average.reshape(grid_size, grid_size)

    # Create a heatmap for the reshaped SPARC values
    plt.figure(figsize=(8, 8))
    plt.title(f"Overall Average SPARC Values for Hand: {hand.capitalize()}", fontsize=16, fontweight='bold')
    im = plt.imshow(rearranged_average_reshaped, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.xticks(range(grid_size), range(1, grid_size + 1))
    plt.yticks(range(grid_size), range(1, grid_size + 1))
    plt.xlabel("Reach Index (Columns)")
    plt.ylabel("Reach Index (Rows)")
    plt.colorbar(im, orientation='vertical', label='SPARC Values')

    # Annotate the heatmap with SPARC values
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f'{rearranged_average_reshaped[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

    # Display the subject names in the plot title
    subject_names = ", ".join(all_subjects)
    plt.suptitle(f"Subjects: {subject_names}", fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()

# Plot a heatmap of overall average LDLJ values for a specific hand, rearranged into a 4x4 grid.
def plot_average_ldlj_value_heatmap_with_subject_average_4x4(reach_TW_metrics, hand):
    all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
    all_average_values = []

    for subject in all_subjects:
        if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
            continue
        ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
        all_average_values.append(ldlj_matrix.mean(axis=0))

    all_average_values = np.array(all_average_values)
    overall_average = all_average_values.mean(axis=0)

    # Rearrange the overall average into the specified order
    if hand.lower() == 'right':
        rearranged_indices = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    elif hand.lower() == 'left':
        rearranged_indices = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    else:
        raise ValueError("Invalid hand specified. Use 'right' or 'left'.")

    rearranged_average = overall_average[rearranged_indices]

    # Reshape the rearranged average into a 4x4 grid
    grid_size = 4
    rearranged_average_reshaped = rearranged_average.reshape(grid_size, grid_size)

    # Create a heatmap for the reshaped LDLJ values
    plt.figure(figsize=(8, 8))
    plt.title(f"Overall Average LDLJ Values for Hand: {hand.capitalize()}", fontsize=16, fontweight='bold')
    im = plt.imshow(rearranged_average_reshaped, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.xticks(range(grid_size), range(1, grid_size + 1))
    plt.yticks(range(grid_size), range(1, grid_size + 1))
    plt.xlabel("Reach Index (Columns)")
    plt.ylabel("Reach Index (Rows)")
    plt.colorbar(im, orientation='vertical', label='LDLJ Values')

    # Annotate the heatmap with LDLJ values
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f'{rearranged_average_reshaped[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

    # Display the subject names in the plot title
    subject_names = ", ".join(all_subjects)
    plt.suptitle(f"Subjects: {subject_names}", fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()

# Plot a violin plot of SPARC values for all participants, each participant in one color, excluding outliers.
def plot_sparc_violin_all_participants_no_outliers(reach_sparc_test_windows, hand):
    # Prepare data for all participants
    all_data = []
    all_labels = []
    all_colors = []
    participants = list(reach_sparc_test_windows.keys())
    color_palette = sns.color_palette("husl", len(participants))

    for idx, participant in enumerate(participants):
        sparc_data = reach_sparc_test_windows[participant][hand]
        sparc_matrix = np.array([values for values in sparc_data.values()])
        trials, reaches = sparc_matrix.shape

        # Remove outliers using z-score
        sparc_matrix = sparc_matrix[(np.abs(zscore(sparc_matrix, axis=None)) < 3).all(axis=1)]

        # Append data and labels
        for reach_idx in range(reaches):
            reach_values = sparc_matrix[:, reach_idx]
            all_data.extend(reach_values)
            all_labels.extend([reach_idx + 1] * len(reach_values))
            all_colors.extend([color_palette[idx]] * len(reach_values))

    # Create DataFrame for plotting
    data = pd.DataFrame({
        'SPARC Values': all_data,
        'Reach': all_labels,
        'Participant': all_colors
    })

    # Plot violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x='Reach', 
        y='SPARC Values', 
        data=data, 
        inner=None, 
        scale='width', 
        palette='light:#d3d3d3'  # Light grey fill for violins
    )

    # Overlay value dots for each participant
    for idx, participant in enumerate(participants):
        sparc_data = reach_sparc_test_windows[participant][hand]
        sparc_matrix = np.array([values for values in sparc_data.values()])
        trials, reaches = sparc_matrix.shape

        # Remove outliers using z-score
        sparc_matrix = sparc_matrix[(np.abs(zscore(sparc_matrix, axis=None)) < 3).all(axis=1)]

        for reach_idx in range(reaches):
            reach_values = sparc_matrix[:, reach_idx]
            plt.scatter(
                [reach_idx] * len(reach_values),
                reach_values,
                c=[color_palette[idx]] * len(reach_values),
                alpha=0.6,
                label=participant if reach_idx == 0 else ""
            )

    plt.xlabel('Reach')
    plt.ylabel('SPARC Values')
    plt.title(f'Violin Plot of SPARC Values for {hand.capitalize()} Hand (All Participants, No Outliers)')
    plt.legend(title='Participants', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# LDLJ
# Plot a heatmap of actual average LDLJ values across trials for all subjects and a specific hand,
# with an additional subplot for the average across all subjects.
def plot_average_ldlj_value_heatmap_with_subject_average(reach_TW_metrics, hand):
    import matplotlib.pyplot as plt
    all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
    all_average_values = []

    for subject in all_subjects:
        if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
            continue
        ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
        all_average_values.append(ldlj_matrix.mean(axis=0))

    all_average_values = np.array(all_average_values)
    overall_average = all_average_values.mean(axis=0)

    plt.figure(figsize=(12, 0.5 * len(all_average_values) + 2))

    # Heatmap for all subjects
    plt.subplot(2, 1, 1)
    plt.imshow(all_average_values, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Actual Average LDLJ Values')
    plt.xlabel('Reach')
    plt.ylabel('Subjects')
    plt.yticks(ticks=range(len(all_subjects)), labels=all_subjects)
    plt.xticks(ticks=range(all_average_values.shape[1]), labels=range(1, all_average_values.shape[1] + 1))
    plt.title(f'Heatmap of Actual Average LDLJ Values for Hand: {hand}')

    # Heatmap for overall average across subjects
    plt.subplot(2, 1, 2)
    im = plt.imshow(overall_average[np.newaxis, :], aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Overall Average LDLJ Values')
    plt.xlabel('Reach')
    plt.yticks([])  # Remove y-axis ticks since it's a single row
    plt.xticks(ticks=range(overall_average.shape[0]), labels=range(1, overall_average.shape[0] + 1))
    plt.title('Overall Average LDLJ Values Across Subjects')

    # Annotate values in subplot 2
    for i, value in enumerate(overall_average):
        plt.text(i, 0, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.show()

# Plot a heatmap of average ranked LDLJ values across trials across all subjects for a specific hand.
def plot_average_ldlj_ranking_heatmap_across_all_dates(reach_TW_metrics, hand):
    import matplotlib.pyplot as plt
    all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
    all_average_rankings = []

    for subject in all_subjects:
        if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
            continue
        ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
        ldlj_matrix_ranked = np.argsort(np.argsort(ldlj_matrix, axis=1), axis=1) + 1
        all_average_rankings.append(ldlj_matrix_ranked.mean(axis=0))

    # Plot heatmap of average rankings for all subjects
    plt.figure(figsize=(12, 0.5 * len(all_average_rankings) + 2))
    plt.subplot(2, 1, 1)
    plt.imshow(np.array(all_average_rankings), aspect='auto', cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar(label='Average Ranked LDLJ Values')
    plt.xlabel('Reach')
    plt.ylabel('Subjects')
    plt.yticks(ticks=range(len(all_subjects)), labels=all_subjects)
    plt.xticks(ticks=range(ldlj_matrix.shape[1]), labels=range(1, ldlj_matrix.shape[1] + 1))
    plt.title(f'Heatmap of Average Ranked LDLJ Values for Hand: {hand}')

    # Plot average across subjects as a heatmap
    plt.subplot(2, 1, 2)
    overall_average = np.mean(all_average_rankings, axis=0)
    im = plt.imshow(overall_average[np.newaxis, :], aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Average Ranked LDLJ Values')
    plt.xlabel('Reach')
    plt.yticks([])  # Remove y-axis ticks since it's a single row
    plt.title('Heatmap of Average Ranked LDLJ Values Across Subjects')

    # Annotate values in subplot 2
    for i, value in enumerate(overall_average):
        plt.text(i, 0, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.show()


# --- SPARC VS LDLJ PLOT ---

# Visualize the correlation between LDLJ and SPARC values using a heatmap of ranked values.
def visualize_ldlj_sparc_correlation(ldlj_values, sparc_values_3):
    # ldlj_values = reach_TW_metrics['reach_LDLJ']['07/23/AK']['right']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/23/AK/AK_tBBT01.csv']
    # sparc_values_3 = reach_sparc_test_windows_1['07/23/AK']['right']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/23/AK/AK_tBBT01.csv']

    # Rank normalization for LDLJ and SPARC values
    ldlj_ranked = np.argsort(np.argsort(ldlj_values)) + 1
    sparc_ranked = np.argsort(np.argsort(sparc_values_3)) + 1

    # Perform Spearman correlation on ranked values
    correlation, p_value = spearmanr(ldlj_ranked, sparc_ranked)
    print(f"Spearman correlation on ranked values: correlation = {correlation}, p-value = {p_value}")

    # Combine ranked LDLJ and SPARC values into a 2D array for heatmap
    comparison_matrix_ranked = np.array([ldlj_ranked, sparc_ranked])

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(comparison_matrix_ranked, aspect='auto', cmap='viridis', interpolation='nearest')

    # Add colorbar, labels, and title
    plt.colorbar(label='Ranked Values')
    plt.yticks(ticks=[0, 1], labels=['LDLJ Ranked Values', 'SPARC Ranked Values (Time Window 3)'])
    plt.xlabel('Reach')
    plt.xticks(ticks=np.arange(len(ldlj_ranked)), labels=np.arange(1, len(ldlj_ranked) + 1))
    plt.title('Heatmap of Ranked LDLJ and SPARC Values')
    # Add text box annotation
    plt.text(
        x=0.5, y=-0.2, 
        s="SPARC and LDLJ: lowest value (Rank 1) = least smooth, highest value (Rank 16) = most smooth", 
        fontsize=10, ha='center', va='center', transform=plt.gca().transAxes
    )

    # Show the heatmap
    plt.tight_layout()
    plt.show()

# Plot scatter plots of LDLJ vs. SPARC values for each trial of a specific subject and hand.
def plot_ldlj_sparc_scatter_by_trial(reach_TW_metrics, reach_sparc_test_windows_1, subject, hand):
    trials = reach_TW_metrics['reach_LDLJ'][subject][hand]
    num_trials = len(trials)
    cols = 5
    rows = int(np.ceil(num_trials / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, (trial, ldlj_values) in enumerate(trials.items()):
        sparc_values = reach_sparc_test_windows_1[subject][hand][trial]

        # Generate a color gradient from light green to dark green
        colors = plt.cm.Greens(np.linspace(0.2, 1, len(ldlj_values)))

        ax = axes[i]
        scatter = ax.scatter(ldlj_values, sparc_values, color=colors, alpha=1)
        ax.set_title(f'Trial {i + 1}')
        ax.set_xlabel('LDLJ')
        ax.set_ylabel('SPARC')

        # Spearman correlation
        corr, p = spearmanr(ldlj_values, sparc_values)
        ax.text(0.05, 0.95, f"corr = {corr:.2f}\np = {p:.2e}", transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"LDLJ vs SPARC Scatter Plots ({subject}, {hand})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Plot heatmaps of ranked LDLJ and SPARC values for each trial of a specific subject and hand.
def plot_ldlj_sparc_correlation_by_trial(reach_TW_metrics, reach_sparc_test_windows_1, subject, hand):
    """
    Plot heatmaps of ranked LDLJ and SPARC values for each trial of a specific subject and hand.

    Parameters:
        reach_TW_metrics (dict): Dictionary containing LDLJ metrics.
        reach_sparc_test_windows_1 (dict): Dictionary containing SPARC metrics.
        subject (str): Subject identifier (e.g., '07/23/AK').
        hand (str): Hand identifier ('right' or 'left').
    """
    # Iterate through all trials in the specified hand and create a grid of subplots
    trials = reach_TW_metrics['reach_LDLJ'][subject][hand]
    num_trials = len(trials)

    fig, axes = plt.subplots(num_trials, 1, figsize=(10, 20))
    axes = axes.flatten()  # Flatten axes for easier indexing

    for i, (trial, ldlj_values) in enumerate(trials.items()):
        # Extract SPARC values for the same trial
        sparc_values_3 = reach_sparc_test_windows_1[subject][hand][trial]

        # Rank normalization for LDLJ and SPARC values
        ldlj_ranked = np.argsort(np.argsort(ldlj_values)) + 1
        sparc_ranked = np.argsort(np.argsort(sparc_values_3)) + 1

        # Perform Spearman correlation on ranked values
        correlation, p_value = spearmanr(ldlj_ranked, sparc_ranked)

        # Combine ranked LDLJ and SPARC values into a 2D array for heatmap
        comparison_matrix_ranked = np.array([ldlj_ranked, sparc_ranked])

        # Plot heatmap for the current trial
        ax = axes[i]
        im = ax.imshow(comparison_matrix_ranked, aspect='auto', cmap='viridis', interpolation='nearest')

        # Add labels and title
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['LDLJ', 'SPARC'])
        if i == num_trials - 1:  # Only show x-axis labels for the last subplot
            ax.set_xlabel('Reach')
            ax.set_xticks(ticks=np.arange(len(ldlj_ranked)))
            ax.set_xticklabels(labels=np.arange(1, len(ldlj_ranked) + 1))
        else:
            ax.set_xticks([])
        ax.text(
            x=-0.1, y=0.5, 
            s=f'Trial: {i + 1}\nCorr: {correlation:.2f}\np: {p_value:.2e}', 
            fontsize=8, ha='right', va='center', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

    # Hide any unused subplots
    for j in range(num_trials, len(axes)):
        axes[j].axis('off')

    # Add a single colorbar for the entire figure
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1, label='Ranked Values', location='top')

    # Adjust the position of the colorbar to avoid overlapping with subplots
    cbar.ax.set_position([0.15, 0.92, 0.7, 0.02])  # [left, bottom, width, height]

    # Adjust layout and show the plot
    fig.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust rect to leave space for the colorbar
    plt.show()

# --- PLOT EACH SEGMENT SPEED AS SUBPLOT WITH LDLJ AND SPARC VALUES ---
def plot_segments_with_ldlj_and_sparc(date, hand, trial, test_windows, results, reach_TW_metrics, reach_sparc, save_path):
    """
    Plot speed segments for a specific trial with LDLJ and SPARC values as titles.

    Parameters:
        date (str): The date of the trial.
        hand (str): The hand ('right' or 'left').
        trial (str): The trial file path.
        test_windows (dict): The time window segments.
        results (dict): The processed results data.
        reach_TW_metrics (dict): The time window metrics containing LDLJ values.
        reach_sparc (dict): The SPARC values for each segment.
        save_path (str): The directory to save the plots.
    """
    # Get the data for the selected trial
    segments = test_windows[date][hand][trial]
    marker = "RFIN" if hand == "right" else "LFIN"
    speed = results[date][hand][1][trial]['traj_space'][marker][1]
    ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]
    sparc_values = reach_sparc[date][hand][trial]

    # Plot the speed for each segment
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}, Trial: {os.path.basename(trial)}", fontsize=20, fontweight='bold')
    axes = axes.flatten()

    for i, (start, end) in enumerate(segments):
        if i >= 16:  # Limit to 16 subplots
            break
        ax = axes[i]
        ax.plot(speed[start:end], color='blue', label='Speed (m/s)')
        ax.set_title(
            f"LDLJ: {ldlj_values[i]:.2f}, SPARC: {sparc_values[i]:.2f}" if ldlj_values[i] is not None and sparc_values[i] is not None else "LDLJ/SPARC: None"
        )
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_ylim(0, 3000)  # Set y-axis range from 0 to 3000
        ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, 16):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_{os.path.basename(trial)}_speed_segments.png")
    plt.savefig(save_file)
    plt.close(fig)


def plot_all_segments_with_ldlj_and_sparc(Figure_folder, test_windows, results, reach_TW_metrics, reach_sparc):
    """
    Plot all segments with LDLJ and SPARC values for each trial and save the plots.

    Parameters:
        Figure_folder (str): The base folder to save the plots.
        test_windows (dict): The time window segments.
        results (dict): The processed results data.
        reach_TW_metrics (dict): The time window metrics containing LDLJ values.
        reach_sparc (dict): The SPARC values for each segment.
    """
    for date in reach_TW_metrics['reach_LDLJ']:
        for hand in ['right', 'left']:
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]:
                # Create a plot save path for the trial
                plot_save_path = os.path.join(Figure_folder, date.replace("/", "_"))
                os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

                # Call the utility function to plot segments with LDLJ and SPARC
                plot_segments_with_ldlj_and_sparc(
                    date=date,
                    hand=hand,
                    trial=trial,
                    test_windows=test_windows,
                    results=results,
                    reach_TW_metrics=reach_TW_metrics,
                    reach_sparc=reach_sparc,
                    save_path=plot_save_path
                )

# --- PLOT JERK SEGMENTS ---
def plot_jerk_segments(date, hand, trial, test_windows, results, reach_TW_metrics, save_path):
    """
    Plot jerk segments for a specific trial with LDLJ values as titles, overlaying speed data.

    Parameters:
        date (str): The date of the trial.
        hand (str): The hand ('right' or 'left').
        trial (str): The trial file path.
        test_windows (dict): The time window segments.
        results (dict): The processed results data.
        reach_TW_metrics (dict): The time window metrics containing LDLJ values.
    """
    # Get the data for the selected trial
    segments = test_windows[date][hand][trial]
    jerk = results[date][hand][1][trial]['traj_space']['RFIN'][3] if hand == "right" else results[date][hand][1][trial]['traj_space']['LFIN'][3]
    speed = results[date][hand][1][trial]['traj_space']['RFIN'][1] if hand == "right" else results[date][hand][1][trial]['traj_space']['LFIN'][1]
    ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]

    # Calculate global min and max values for jerk and speed across all segments
    min_jerk = min(jerk[start:end].min() for start, end in segments)
    max_jerk = max(jerk[start:end].max() for start, end in segments)
    min_speed = min(speed[start:end].min() for start, end in segments)
    max_speed = max(speed[start:end].max() for start, end in segments)

    # Plot the jerk and overlay speed for each segment
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}, Trial: {os.path.basename(trial)}", fontsize=20, fontweight='bold')  # Add date and hand as big title
    axes = axes.flatten()

    for i, (start, end) in enumerate(segments):
        if i >= 16:  # Limit to 16 subplots
            break
        ax1 = axes[i]
        ax1.plot(jerk[start:end], color='blue', label='Jerk (m/s³)')
        ax1.set_title(f"LDLJ: {ldlj_values[i]:.2f}" if ldlj_values[i] is not None else "LDLJ: None")
        ax1.set_xlabel("Time (samples)")
        ax1.set_ylabel("Jerk (m/s³)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_yticks(np.linspace(min_jerk, max_jerk, 5))
        ax1.set_ylim(min_jerk - 1000, max_jerk + 1000)  # Set consistent y-axis limits for jerk

        # Create a secondary y-axis for speed
        ax2 = ax1.twinx()
        ax2.plot(speed[start:end], color='red', label='Speed (m/s)', alpha=0.7)
        ax2.set_ylabel("Speed (m/s)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_yticks(np.linspace(min_speed, max_speed, 5))
        ax2.set_ylim(min_speed - 50, max_speed + 50)  # Set consistent y-axis limits for speed

    # Hide unused subplots
    for j in range(i + 1, 16):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_{os.path.basename(trial)}_jerk_segments.png")
    plt.savefig(save_file)
    plt.close(fig)

# --- PLOT SEGMENTS ---
def plot_segments(date, hand, trial, test_windows, results, reach_TW_metrics, save_path):
    """
    Plot jerk segments for a specific trial with LDLJ values as titles, overlaying speed, position, and acceleration data.

    Parameters:
        date (str): The date of the trial.
        hand (str): The hand ('right' or 'left').
        trial (str): The trial file path.
        test_windows (dict): The time window segments.
        results (dict): The processed results data.
        reach_TW_metrics (dict): The time window metrics containing LDLJ values.
    """
    # Get the data for the selected trial
    segments = test_windows[date][hand][trial]
    marker = "RFIN" if hand == "right" else "LFIN"
    jerk = results[date][hand][1][trial]['traj_space'][marker][3]
    speed = results[date][hand][1][trial]['traj_space'][marker][1]
    position = results[date][hand][1][trial]['traj_space'][marker][0]
    acceleration = results[date][hand][1][trial]['traj_space'][marker][2]
    ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]

    # Plot the jerk and overlay speed, position, and acceleration for each segment
    fig, axes = plt.subplots(4, 4, figsize=(30, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}, Trial: {os.path.basename(trial)}", fontsize=20, fontweight='bold')  # Add date and hand as big title
    axes = axes.flatten()

    for i, (start, end) in enumerate(segments):
        if i >= 16:  # Limit to 16 subplots
            break
        ax1 = axes[i]
        ax1.plot(jerk[start:end], color='blue', label='Jerk (m/s³)')
        ax1.set_title(f"LDLJ: {ldlj_values[i]:.2f}" if ldlj_values[i] is not None else "LDLJ: None")
        ax1.set_xlabel("Time (samples)")
        ax1.set_ylabel("Jerk (m/s³)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create secondary y-axes for speed, position, and acceleration
        ax2 = ax1.twinx()
        ax2.plot(speed[start:end], color='red', label='Speed (m/s)', alpha=0.7)
        ax2.set_ylabel("Speed (m/s)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
        ax3.plot(position[start:end], color='green', label='Position (m)', alpha=0.7)
        ax3.set_ylabel("Position (m)", color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        ax4 = ax1.twinx()
        ax4.spines["right"].set_position(("outward", 120))  # Offset the fourth axis
        ax4.plot(acceleration[start:end], color='purple', label='Acceleration (m/s²)', alpha=0.7)
        ax4.set_ylabel("Acceleration (m/s²)", color='purple')
        ax4.tick_params(axis='y', labelcolor='purple')

    # Hide unused subplots
    for j in range(i + 1, 16):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_{os.path.basename(trial)}_jerk_segments.png")
    plt.savefig(save_file)
    plt.close(fig)

# --- PLOT REACH LDLJ VALUES OVER TRIALS ---
def plot_reach_ldlj_over_trials(reach_TW_metrics, date, hand, save_path):
    """
    Save scatter plots for LDLJ values across trials for each reach as subplots.

    Parameters:
        reach_TW_metrics (dict): The reach time window metrics containing LDLJ values.
        date (str): The date of the trials.
        hand (str): The hand ('right' or 'left').
        save_path (str): The directory to save the plots.
    """
    # Determine global y-axis limits across all subplots
    all_ldlj_values = [
        ldlj
        for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
        for reach_index, ldlj in enumerate(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
    ]
    y_min, y_max = min(all_ldlj_values), max(all_ldlj_values)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}", fontsize=20, fontweight='bold')  # Add date and hand as big title
    axes = axes.flatten()

    for reach_index in range(16):
        ldlj_values = [
            reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index]
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
            if reach_index < len(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
        ]

        ax = axes[reach_index]
        ax.scatter(range(len(ldlj_values)), ldlj_values, color='blue', label='LDLJ Values')
        ax.set_xlabel('Trial Index')
        ax.set_ylabel('LDLJ Value')
        ax.set_title(f'Reach {reach_index + 1}')
        ax.legend()
        ax.grid(True)

        # Set consistent y-axis limits
        ax.set_ylim(y_min, y_max)

    # Hide unused subplots if any
    for i in range(reach_index + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_reach_ldlj_over_trials.png")
    plt.savefig(save_file)
    plt.close(fig)

# --- PLOT REACH LDLJ VALUES AGAINST REACH DURATIONS ---
def plot_reach_ldlj_vs_duration(reach_TW_metrics, reach_metrics, date, hand, save_path):
    """
    Generates scatter plots of LDLJ values against reach durations for multiple reaches 
    and saves the plots as an image file.

    Parameters:
        reach_TW_metrics (dict): A dictionary containing LDLJ metrics data. 
                                 Expected structure:
                                 reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index].
        reach_metrics (dict): A dictionary containing reach duration metrics data. 
                              Expected structure:
                              reach_metrics['reach_durations'][date][hand][trial][reach_index].
        date (str): The date key used to access the metrics data.
        hand (str): The hand key ('left' or 'right') used to access the metrics data.
        save_path (str): The directory path where the scatter plot image will be saved.

    Returns:
        None: The function saves the scatter plot image to the specified path and does not return anything.

    Notes:
        - The function creates a 4x4 grid of subplots, one for each reach (up to 16 reaches).
        - If there are fewer than 16 reaches, unused subplots are hidden.
        - Each subplot displays a scatter plot of reach durations (x-axis) against LDLJ values (y-axis).
        - The image is saved as "{hand}_LDLJ_scatter_plots.png" in the specified save_path directory.
    """
    # Determine global x and y axis limits across all subplots
    all_ldlj_values = [
        ldlj
        for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
        for reach_index, ldlj in enumerate(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
    ]
    all_reach_durations = [
        duration
        for trial in reach_metrics['reach_durations'][date][hand]
        for reach_index, duration in enumerate(reach_metrics['reach_durations'][date][hand][trial])
    ]
    x_min, x_max = min(all_reach_durations), max(all_reach_durations)
    y_min, y_max = min(all_ldlj_values), max(all_ldlj_values)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}", fontsize=20, fontweight='bold')# Add date as big title
    axes = axes.flatten()

    for reach_index in range(16):
        ldlj_values = [
            reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index]
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
            if reach_index < len(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
        ]

        reach_durations = [
            reach_metrics['reach_durations'][date][hand][trial][reach_index]
            for trial in reach_metrics['reach_durations'][date][hand]
            if reach_index < len(reach_metrics['reach_durations'][date][hand][trial])
        ]
        ax = axes[reach_index]
        ax.scatter(reach_durations, ldlj_values, color='blue', label='LDLJ Values')
        ax.set_xlabel('Reach Duration (s)')
        ax.set_ylabel('LDLJ Value')
        ax.set_title(f'Reach {reach_index + 1}')
        ax.grid(True)

        # Set consistent axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Hide unused subplots if any
    for i in range(reach_index + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_reach_ldlj_vs_duration.png")
    plt.savefig(save_file)
    plt.close(fig)

# --- CALCULATE REACH LDLJ VALUES AGAINST REACH DURATIONS AND CORRELATIONS ---
def calculate_ldlj_vs_duration_correlation(reach_TW_metrics, reach_metrics, date, hand):
    """
    Calculates the correlation between z-scores of LDLJ values and reach durations for multiple reaches.

    Parameters:
        reach_TW_metrics (dict): A dictionary containing LDLJ metrics data. 
                                 Expected structure:
                                 reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index].
        reach_metrics (dict): A dictionary containing reach duration metrics data. 
                              Expected structure:
                              reach_metrics['reach_durations'][date][hand][trial][reach_index].
        date (str): The date key used to access the metrics data.
        hand (str): The hand key ('left' or 'right') used to access the metrics data.

    Returns:
        dict: A dictionary containing correlations for each reach index.

    Notes:
        - The function calculates z-scores for LDLJ values and reach durations.
        - Correlation is calculated for each reach index across trials.
        - If there are fewer than 2 data points for a reach index, correlation is set to None.
    """
    correlations = {}

    for reach_index in range(16):
        ldlj_values = [
            reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index]
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
            if reach_index < len(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
        ]

        reach_durations = [
            reach_metrics['reach_durations'][date][hand][trial][reach_index]
            for trial in reach_metrics['reach_durations'][date][hand]
            if reach_index < len(reach_metrics['reach_durations'][date][hand][trial])
        ]

        if len(ldlj_values) > 1 and len(reach_durations) > 1:
            # Calculate z-scores
            ldlj_z = zscore(ldlj_values)
            durations_z = zscore(reach_durations)

            # Calculate correlation
            correlation = np.corrcoef(ldlj_z, durations_z)[0, 1]
            correlations[reach_index] = correlation
        else:
            correlations[reach_index] = None

    return correlations

# --- PLOT REACH LDLJ DURATION CORRELATIONS AS HISTOGRAM ---
def plot_ldlj_duration_correlations_histogram_by_hand(reach_ldlj_duration_correlations, save_path):
    """
    Plots histograms of LDLJ duration correlations for each date and hand in one figure as subplots.

    Parameters:
        reach_ldlj_duration_correlations (dict): A dictionary containing LDLJ duration correlations.
        save_path (str): The directory to save the histogram plots.

    Returns:
        None: The function saves the histogram plots to the specified path.
    """
    fig, axes = plt.subplots(len(reach_ldlj_duration_correlations), 2, figsize=(15, 5 * len(reach_ldlj_duration_correlations)))
    fig.suptitle("LDLJ Duration Correlations Histograms", fontsize=20, fontweight='bold')

    for i, date in enumerate(reach_ldlj_duration_correlations):
        for j, hand in enumerate(['right', 'left']):
            correlations = [
                corr for corr in reach_ldlj_duration_correlations[date][hand].values() if corr is not None
            ]

            ax = axes[i][j] if len(reach_ldlj_duration_correlations) > 1 else axes[j]
            ax.hist(correlations, bins=10, color='blue', alpha=0.7)
            ax.set_title(f"Date: {date}, Hand: {hand}", fontsize=14)
            ax.set_xlabel("Correlation Coefficient")
            ax.set_ylabel("Frequency")
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "ldlj_duration_correlations_histograms.png")
    plt.savefig(save_file)
    plt.close()

# --- PLOT REACH LDLJ DURATION CORRELATIONS AS OVERLAPPED HISTOGRAM ---
def plot_ldlj_duration_correlations_histogram(reach_ldlj_duration_correlations, save_path):
    """
    Plots a single histogram with overlapped data for LDLJ duration correlations for each date, using different colors.
    Also calculates if the correlations are significantly shifted from 0 and labels the p-value.

    Parameters:
        reach_ldlj_duration_correlations (dict): A dictionary containing LDLJ duration correlations.
        save_path (str): The directory to save the histogram plot.

    Returns:
        None: The function saves the histogram plot to the specified path.
    """
    plt.figure(figsize=(15, 10))
    plt.title("LDLJ Duration Correlations Histogram", fontsize=20, fontweight='bold')
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    for i, date in enumerate(reach_ldlj_duration_correlations):
        correlations = [
            corr for hand in reach_ldlj_duration_correlations[date]
            for corr in reach_ldlj_duration_correlations[date][hand].values() if corr is not None
        ]
        
        # Perform a one-sample t-test to check if correlations are significantly different from 0
        t_stat, p_value = ttest_1samp(correlations, 0)
        
        # Plot histogram
        plt.hist(correlations, bins=10, color=colors[i % len(colors)], alpha=0.7, 
                 label=f"Date: {date} (n={len(correlations)}, p={p_value:.3f})")

    # Add a vertical line at x = 0
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label="x = 0")

    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.grid(True)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "ldlj_duration_correlations_histogram.png")
    plt.savefig(save_file)
    plt.close()

