import utils1 # Importing utils1 for data Pre-processing
import utils2 # Importing utils2 for reach metrics calculation and time window Specific calculation
import utils3 # Importing utils3 for plotting functions
import utils4 # Importing utils4 for image files
import utils5 # Importing utils5 for combining metrics
import utils6 # Importing utils6 for Data Analysis and Visualization
import utils7 # Importing utils7 for Motor Experiences
import utils8 # Importing utils8 for sBBTResult
import utils9 # Importing utils9 for thesis
# -------------------------------------------------------------------------------------------------------------------
import pickle
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

import math
import numpy as np
from scipy.stats import zscore
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
from scipy.stats import chisquare
from scipy.stats import circmean, rayleigh
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import norm

import pandas as pd
import seaborn as sns

import pingouin as pg

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.patches as patches
# -------------------------------------------------------------------------------------------------------------------

Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025"
Box_Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Box/Traj/2025"
Figure_folder = "/Users/yilinwu/Desktop/honours/Thesis/figure"
DataProcess_folder = "/Users/yilinwu/Desktop/honours data/DataProcess"
tBBT_Image_folder = "/Users/yilinwu/Desktop/Yilin-Honours/tBBT_Image/2025/"

prominence_threshold_speed = 400
prominence_threshold_position = 80
# -------------------------------------------------------------------------------------------------------------------
If x axis is hand, x axis label as Non-Dominant / Dominant

overlay sample size
n=32 reaches 
n=16 locations 
n=29 participants

stat test
paired t-test


distance label as Error (mm)
duration label as Duration (s)
if the y axis is corrlation coefficient, y axis label as Correlation
if colcor bar is correlation coefficient, label as Correlation

1. sbbt results
y axis label as sBBT Score (no. of blocks)
y-limits to round numbers and consider having the lowest value be 0.
If x axis is hand, x axis label as Non-Dominant / Dominant
n =29 participants show on the plot
do paired t-test and  to assess whether dominant score is higher and indicate significance on the plot


# 1. sbbt results
# y-axis label → "sBBT Score (no. of blocks)"
# y-axis limits → rounded, minimum forced to 0
# x-tick label → "Non-Dominant / Dominant"
# x-axis label → "Hand"
# n = 29 participants displayed on the plot
# Annotate sample size in the figure right top
# paired t-test performed and annotated on the figure (showing significance if dominant > non-dominant).
stattest result size = 50
figsize=(6, 4)
axis_label_font=14,
tick_label_font=14,
marker_size=50,
alpha=0.7,
bar_width=0.6,
order=("Non-dominant", "Dominant"),
box_colors = ["Non-dominant":"#D3D3D3","Dominant" "#F0F0F0"]
random_jitter=0.04
no tittle
no Grid
# Perform paired t-test (Dominant > Non-dominant) and annotate significance
t_stat, p_val = ttest_rel(sBBTResult["dominant"], sBBTResult["non_dominant"])
y_sig = y_max * 1.05
ax.plot([indices[0], indices[1]], [y_sig, y_sig], color="black", linewidth=1.5)

if p_val < 0.001:
    stars = "***"
elif p_val < 0.01:
    stars = "**"
elif p_val < 0.05:
    stars = "*"
else:
    stars = "ns"
ax.text(np.mean(indices), y_sig + (y_max * 0.02), stars,
        ha="center", va="bottom", fontsize=axis_label_font)



2. tbbt results



def generate_placement_colors(show_plot=True):
    """
    Generate 16 distinct colors (4 categories x 4 shades) for placement locations.
    
    Parameters:
        show_plot (bool): If True, display a bar plot of the colors.
        
    Returns:
        np.ndarray: Array of shape (16, 4) with RGBA colors.
    """
    def generate_shades(base_rgb, n_shades=4):
        factors = np.linspace(0.4, 1.0, n_shades)  # avoid very dark
        shades = np.array([base_rgb * f for f in factors])
        shades = np.clip(shades, 0, 1)
        return np.hstack([shades, np.ones((n_shades, 1))])  # add alpha=1

    # Base colors (RGB)
    base_colors = [
        np.array([0.0, 0.3, 0.8]),  # blue
        np.array([0.0, 0.6, 0.2]),  # green
        np.array([0.8, 0.1, 0.1]),  # red
        np.array([0.9, 0.5, 0.0])   # orange
    ]
    
    # Generate shades and combine
    placement_location_colors = np.vstack([generate_shades(c) for c in base_colors])
    
    if show_plot:
        plt.figure(figsize=(12, 2))
        for i, c in enumerate(placement_location_colors):
            plt.bar(i, 1, color=c)
            plt.text(i, 0.5, str(i+1), ha='center', va='center', 
                     color='white' if np.mean(c[:3]) < 0.5 else 'black', fontsize=12)
        plt.axis('off')
        plt.show()
    
    return placement_location_colors
def create_icon_layout(colors, mirror=False):
    """
    Create compact square icon with annotated numbers using given colors.
    Layout:
      13 14 15 16
       9 10 11 12
       5  6  7  8
       1  2  3  4
    If mirror is True, the layout will be mirrored horizontally.
    """
    layout = np.array([
        [13, 14, 15, 16],
        [9, 10, 11, 12],
        [5, 6, 7, 8],
        [1, 2, 3, 4]
    ])

    if mirror:
        layout = np.fliplr(layout)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Draw squares
    for row in range(4):
        for col in range(4):
            num = layout[row, col]
            color = colors[num-1]
            rect = patches.Rectangle((col, 3-row), 1, 1, linewidth=1,
                                     edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            # Choose text color based on brightness
            text_color = 'white' if np.mean(color[:3]) < 0.5 else 'black'
            ax.text(col+0.5, 3-row+0.5, str(num), ha='center', va='center',
                    fontsize=40, color=text_color, weight="bold")
    
    plt.show()

# Generate colors and draw icon (not mirrored)
placement_location_colors = generate_placement_colors(show_plot=False)
# create_icon_layout(placement_location_colors, mirror=False)
# create_icon_layout(placement_location_colors, mirror=True)


# -------------------------------------------------------------------------------------------------------------------
# --- GET ALL DATES ---
All_dates = sorted(utils1.get_subfolders_with_depth(Traj_folder, depth=3))

# --- LOAD ALL SUBJECTS' tBBT ERROR FROM IMAGE, SAVE AS pickle file---
Block_Distance = utils4.load_selected_subject_errors(All_dates, DataProcess_folder)

# --- LOAD RESULTS FROM PICKLE FILE "processed_results.pkl" ---
results = utils1.load_selected_subject_results(All_dates, DataProcess_folder)
# -------------------------------------------------------------------------------------------------------------------
# Calculate RMS reprojection error statistics
utils4.compute_rms_reprojection_error_stats()
# # -------------------------------------------------------------------------------------------------------------------
# Load sBBTResult from CSV into a DataFrame and compute right and left hand scores
sBBTResult = utils8.load_and_compute_sbbt_result()
# Swap and rename sBBTResult scores for specific subjects
sBBTResult = utils8.swap_and_rename_sbbt_result(sBBTResult)
sBBTResult_stats = utils8.compute_sbbt_result_stats(sBBTResult)
sBBT_combine_stat = utils8.analyze_sbbt_results(sBBTResult)
# -------------------------------------------------------------------------------------------------------------------
# 1. sBBT task performance: Dominant vs Non-Dominant hand
sbbt_plot_config = dict(
    figsize=(5, 4),
    scale_factor=1,
    axis_label_font=14,
    tick_label_font=14,
    title_font=16,
    marker_size=50,
    alpha=0.4,
    bar_width=0.5,
    bar_edge_width=1.5,
    bar_colors={"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"},
    order=("Non-dominant", "Dominant"),
    random_jitter=0.04,
    bar_spacing=0.1,  # spacing fraction between bars
    show_title=False,
    show_grid=False,
    x_ticks=True,
    y_ticks=True,
    annotate_n=True,
    n_loc=(0.95, 1.05),  # "top-right" or "bottom" or (x, y) in axes fraction
    n_unit="participants",  # optional: "blocks", "participants", "cm", "locations", or None
    annotate_sig=True,
    sig_text_offset=-0.05,
    sig_marker_size=40,
    sig_line=True,
    sig_line_width=1.5,
    sig_line_color="black",
    sig_y_loc=90,
    sig_levels=[(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")],
    test_type="greater", # "greater" (if Dominant > Non-dominant), "less", or "two-sided" 
    y_unit="blocks",  # optional: "blocks", "participants", "cm", "locations", or None
    show_whiskers=False,  #  option: show error bar whiskers or not
    show_points=True     #  option: overlay individual data points or not
)

def plot_sbbt_bargraph(sBBTResult, config):
    """
    Plot a bar graph of sBBT scores by hand with full configurability.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import ttest_rel

    # Scaling factor
    sf = config.get("scale_factor", 1.0)

    # Figure & fonts
    figsize = config.get("figsize", (6, 4))
    axis_label_font = config.get("axis_label_font", 14) * sf
    tick_label_font = config.get("tick_label_font", 14) * sf
    title_font = config.get("title_font", 16) * sf

    # Bar & scatter
    marker_size = config.get("marker_size", 50) * sf
    alpha = config.get("alpha", 0.7)
    bar_width = config.get("bar_width", 0.6)
    bar_edge_width = config.get("bar_edge_width", 1.5) * sf
    bar_colors = config.get("bar_colors", {"Non-dominant": "#D3D3D3", "Dominant": "#F0F0F0"})
    order = config.get("order", ("Non-dominant", "Dominant"))
    random_jitter = config.get("random_jitter", 0.04)
    bar_spacing = config.get("bar_spacing", 0.3)  # spacing fraction between bars

    # Axis & grid
    show_title = config.get("show_title", False)
    show_grid = config.get("show_grid", False)
    x_ticks = config.get("x_ticks", True)
    y_ticks = config.get("y_ticks", True)

    # Sample size
    annotate_n = config.get("annotate_n", True)
    n_loc = config.get("n_loc", "top-right")
    n_unit = config.get("n_unit", "participants")

    # Significance
    annotate_sig = config.get("annotate_sig", True)
    sig_text_offset = config.get("sig_text_offset", 0.02)
    sig_marker_size = config.get("sig_marker_size", 14) * sf
    sig_line = config.get("sig_line", True)
    sig_line_width = config.get("sig_line_width", 1.5) * sf
    sig_line_color = config.get("sig_line_color", "black")
    sig_y_loc = config.get("sig_y_loc", "auto")
    sig_levels = config.get("sig_levels", [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")])
    test_type = config.get("test_type", "greater")

    # Y-axis unit
    y_unit = config.get("y_unit", None)
    y_label_base = "sBBT Score"
    y_label = f"{y_label_base} (no. of {y_unit})" if y_unit else y_label_base

    # New options
    show_whiskers = config.get("show_whiskers", True)
    show_points = config.get("show_points", True)

    # Subjects
    n = len(sBBTResult)
    means = [sBBTResult["non_dominant"].mean(), sBBTResult["dominant"].mean()]
    sems = [sBBTResult["non_dominant"].std() / np.sqrt(n),
            sBBTResult["dominant"].std() / np.sqrt(n)]

    # Figure
    fig, ax = plt.subplots(figsize=(figsize[0]*sf, figsize[1]*sf))
    indices = np.arange(len(order)) * (1 + bar_spacing)

    # Bars with or without whiskers
    error_kw = {'elinewidth': bar_edge_width} if show_whiskers else None
    yerr = sems if show_whiskers else None
    ax.bar(indices, means,
           yerr=yerr,
           color=[bar_colors[order[0]], bar_colors[order[1]]],
           width=bar_width,
           capsize=8 if show_whiskers else 0,
           edgecolor='black',
           linewidth=bar_edge_width,
           error_kw=error_kw)

    # Scatter points overlay if enabled
    if show_points:
        for i, key in enumerate(["non_dominant", "dominant"]):
            x_vals = indices[i] + np.random.uniform(-random_jitter, random_jitter, n)
            ax.scatter(x_vals, sBBTResult[key], color='black', s=marker_size, zorder=10, alpha=alpha)

    # Labels
    ax.set_xlabel("Hand", fontsize=axis_label_font)
    ax.set_ylabel(y_label, fontsize=axis_label_font)
    if show_title:
        ax.set_title("sBBT Bar Graph", fontsize=title_font)
    ax.set_xticks(indices)
    ax.set_xticklabels(["Non-dominant", "Dominant"], fontsize=tick_label_font)
    ax.tick_params(axis='x', which='both', bottom=x_ticks)
    ax.tick_params(axis='y', which='both', left=y_ticks, labelsize=tick_label_font)

    # Y-axis limits
    max_score = sBBTResult[["non_dominant", "dominant"]].max().max()
    y_max = int(np.ceil(max_score / 5.0)) * 5
    if y_max < max_score:
        y_max += 5
    ax.set_ylim(0, y_max)

    # Spines & grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(show_grid)

    # Sample size annotation
    if annotate_n:
        if isinstance(n_loc, tuple):
            ax.text(n_loc[0], n_loc[1], f"n = {n} {n_unit}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=tick_label_font)
        elif n_loc == "top-right":
            ax.text(0.95, 0.95, f"n = {n} {n_unit}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=tick_label_font)
        elif n_loc == "bottom":
            ax.text(0.5, -0.15, f"n = {n} {n_unit}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=tick_label_font)

    # Significance annotation
    if annotate_sig:
        t_stat, p_val_two = ttest_rel(sBBTResult["dominant"], sBBTResult["non_dominant"])
        if test_type == "greater":
            p_val = p_val_two / 2 if t_stat > 0 else 1.0
        elif test_type == "less":
            p_val = p_val_two / 2 if t_stat < 0 else 1.0
        else:
            p_val = p_val_two

        # Determine significance stars
        for threshold, symbol in sig_levels:
            if p_val <= threshold:
                stars = symbol
                break

        # Determine y location
        y_sig = y_max * 1.05 if sig_y_loc == "auto" else sig_y_loc

        # Draw line connecting bars if enabled
        if sig_line:
            ax.plot([indices[0], indices[1]], [y_sig, y_sig],
                    color=sig_line_color, linewidth=sig_line_width)

        # Place significance text
        ax.text(np.mean(indices), y_sig + (y_max * sig_text_offset),
                stars, ha="center", va="bottom", fontsize=sig_marker_size)

    plt.tight_layout()
    plt.show()

plot_sbbt_bargraph(sBBTResult, sbbt_plot_config)
# -------------------------------------------------------------------------------------------------------------------
# 2. tBBT task performance: Duration and Distance Trade-off
# tbbt_plot_config = dict(
#     figsize=(5, 4),
#     scale_factor=1,
#     axis_label_font=14,
#     tick_label_font=14,
#     title_font=16,
#     marker_size=50,
#     alpha=0.4,
#     bar_width=0.5,
#     bar_edge_width=1.5,
#     bar_colors={"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"},
#     order=("Non-dominant", "Dominant"),
#     random_jitter=0.04,
#     bar_spacing=0.1,  # spacing fraction between bars
#     show_title=False,
#     show_grid=False,
#     x_ticks=True,
#     y_ticks=True,
#     annotate_n=True,
#     n_loc=(0.95, 1.05),  # "top-right" or "bottom" or (x, y) in axes fraction
#     n_unit="participants",  # optional: "blocks", "participants", "cm", "locations", or None
#     annotate_sig=True,
#     sig_text_offset=-0.05,
#     sig_marker_size=40,
#     sig_line=True,
#     sig_line_width=1.5,
#     sig_line_color="black",
#     sig_y_loc=90,
#     sig_levels=[(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")],
#     test_type="greater", # "greater" (if Dominant > Non-dominant), "less", or "two-sided" 
#     y_unit="blocks",  # optional: "blocks", "participants", "cm", "locations", or None
#     show_whiskers=False,  #  option: show error bar whiskers or not
#     show_points=True     #  option: overlay individual data points or not
# )

plot_config_summary = dict(
    # -----------------------------
    # General Plot Settings
    # -----------------------------
    general=dict(
        figsize=(5, 4),
        scale_factor=1,
        axis_label_font=14,
        tick_label_font=14,
        title_font=16,
        show_title=False,
        show_grid=False,
        x_ticks=False,   # no numeric x-axis ticks
        y_ticks=False,   # no numeric y-axis ticks
        annotate_n=True,
        n_loc=(0.95, 1.05),
        n_unit="participants",
        random_jitter=0.04,
        show_whiskers=False,
        show_points=True,
        marker_size=50,
        alpha=0.4,
        label_offset=0.08  # fraction of axis range to offset start/end labels
    ),

    # -----------------------------
    # Axis Labeling Rules (common style)
    # -----------------------------
    axis_labels=dict(
        duration="Duration (s)",
        distance="Error (mm)",
        correlation="Correlation"
    ),
    axis_colors=dict(
        x={
            "Duration (s)": {"start": "fast", "end": "slow", "colors": ["green", "red"]}
        },
        y={
            "Error (mm)": {"start": "accurate", "end": "inaccurate", "colors": ["green", "red"]}
        }
    ),

    # -----------------------------
    # Plot-Type Specific Options
    # -----------------------------
    scatter=dict(
        show_points=True,
        annotate_corr=True,
        corr_line_y0=True,
        ylim_centered_at_zero=True
    ),
    line=dict(
        show_markers=False,
        show_error_shade=False, 
        linewidth=4
    ),
    heatmap=dict(
        colormap="viridis",
        show_colorbar=True,
        colorbar_label=None,
        center_zero=False
    ),
    box=dict(
        bar_width=0.5,
        bar_edge_width=1.5,
        bar_colors={"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"},
        show_whiskers=False,
        show_points=True,
        annotate_sig=True,
        sig_levels=[(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")],
        test_type="greater",
        sig_y_loc=90,
        sig_line=True,
        sig_line_width=1.5,
        sig_line_color="black",
        sig_marker_size=40,
        sig_text_offset=-0.05
    ),

    # -----------------------------
    # Statistical Options
    # -----------------------------
    stats=dict(
        compare_correlation="fisher_z",
        test_type_options=["greater", "less", "two-sided"]
    ),

    # -----------------------------
    # Optional / Misc Features
    # -----------------------------
    misc=dict(
        bar_spacing=0.1,
        placement_icon=True,
        annotate_sig=True
    )
)

# -------------------------------------------------------------------------------------------------------------------
# Hypothesis: Participants will be mmore accuracy when they take longer to complete placements, demonstrating a speed-accuracy trade-off.


# 2.1 Duration vs Distance within each placemnet location


def plot_speed_accuracy_tradeoff(cfg):
    """
    Plots a speed-accuracy tradeoff line plot using formatting directly from plot_config_summary.
    
    Parameters
    ----------
    x : array-like
        X-axis data (duration)
    y : array-like
        Y-axis data (error)
    cfg : dict
        Plot configuration dictionary (plot_config_summary)
    """
    x = np.linspace(0.5, 2.0, 100)   # Duration (s)
    y = -10 * x + 20                 # Error (mm)

    # Load settings from config
    general = cfg["general"]
    line_cfg = cfg["line"]
    axis_labels = cfg["axis_labels"]
    axis_colors = cfg["axis_colors"]
    label_offset = general.get("label_offset", 0.08)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=general["figsize"])

    # Plot line
    ax.plot(
        x,
        y,
        color='blue',
        linewidth=line_cfg.get("linewidth", 2),
        marker='o' if line_cfg.get("show_markers", False) else None
    )

    # Set axis labels
    ax.set_xlabel(axis_labels.get("duration", "X"), fontsize=general["axis_label_font"])
    ax.set_ylabel(axis_labels.get("distance", "Y"), fontsize=general["axis_label_font"])

    # Remove numeric ticks if configured
    if not general.get("x_ticks", True):
        ax.set_xticks([])
    if not general.get("y_ticks", True):
        ax.set_yticks([])

    # Restore x/y spines
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Axis ranges for offsetting labels
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    # X-axis start/end labels
    x_cfg = axis_colors.get("x", {}).get(axis_labels["duration"], {})
    if x_cfg:
        ax.text(x[0], y.min() - label_offset*y_range, x_cfg["start"],
                color=x_cfg["colors"][0], ha="center", va="top",
                fontsize=general["tick_label_font"])
        ax.text(x[-1], y.min() - label_offset*y_range, x_cfg["end"],
                color=x_cfg["colors"][1], ha="center", va="top",
                fontsize=general["tick_label_font"])

    # Y-axis start/end labels
    y_cfg = axis_colors.get("y", {}).get(axis_labels["distance"], {})
    if y_cfg:
        ax.text(x.min() - label_offset*x_range, y[-1], y_cfg["start"],
                color=y_cfg["colors"][0], ha="right", va="center",
                fontsize=general["tick_label_font"])
        ax.text(x.min() - label_offset*x_range, y[0], y_cfg["end"],
                color=y_cfg["colors"][1], ha="right", va="center",
                fontsize=general["tick_label_font"])

    plt.tight_layout()
    plt.show()

plot_speed_accuracy_tradeoff(plot_config_summary)
