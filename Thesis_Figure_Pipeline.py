import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Load the DataFrame from a pickle file
with open("/Users/yilinwu/Desktop/honours data/DataProcess/df.pkl", "rb") as f:
    df = pickle.load(f)

print("DataFrame loaded with shape:", df.shape)

# -----------------------
# Global Styles & Colors
# -----------------------
variable_colors = {
    "ballistic_LDLJ": "#a6cee3",    # light blue
    "ballistic_sparc": "#fdb863",   # light orange
    "corrective_LDLJ": "#1f78b4",   # dark blue
    "corrective_sparc": "#e66101",  # dark orange
    "durations (s)": "#4daf4a",     # green
    "distance (mm)": "#984ea3",     # purple
    "MotorAcuity": "#e7298a"        # magenta/pink
}
LINE_WIDTH = 2
MARKER_SIZE = 6

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

def mm_to_inch(width_mm, height_mm):
    return (width_mm / 25.4, height_mm / 25.4)

def min_mid_max_ticks(ax, axis='both'):
    if axis in ('x', 'both'):
        xmin, xmax = ax.get_xlim()
        ax.set_xticks([xmin, (xmin+xmax)/2, xmax])
    if axis in ('y', 'both'):
        ymin, ymax = ax.get_ylim()
        ax.set_yticks([ymin, (ymin+ymax)/2, ymax])

def format_axis(ax, x_is_categorical=False):
    if x_is_categorical:
        min_mid_max_ticks(ax, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        min_mid_max_ticks(ax, axis='both')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

def smart_legend(fig, axes):
    if isinstance(axes, np.ndarray):
        handles, labels = [], []
        for ax in axes.ravel():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(), by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=False
        )
    else:
        axes.legend(frameon=False, loc="upper right", fontsize=11)

hand_colors = {
    "dominant": "#7570b3",          # purple
    "non_dominant": "#66c2a5"       # teal
}
location_colors = [cm.tab20(i/20) for i in range(16)]
corr_cmap = cm.get_cmap("coolwarm")  # correlation colormap

# Line and marker defaults
MARKER_SHAPE = 'o'
LINE_WIDTH_DEFAULT = 2
MARKER_SIZE_DEFAULT = 6
figsize_mm = (90, 70)



# -----------------------
# 1. Scatter Plot
# -----------------------
def scatter_plot(x, y, xlabel="X", ylabel="Y", color=variable_colors["ballistic_LDLJ"], figsize=figsize_mm):
    fig, ax = plt.subplots(figsize=mm_to_inch(*figsize))
    ax.scatter(x, y, color=color, s=MARKER_SIZE, marker=MARKER_SHAPE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis(ax)
    smart_legend(fig, ax)
    plt.tight_layout()
    plt.show()

# -----------------------
# 2. Histogram
# -----------------------
def histogram(values, xlabel="Value", ylabel="Count", color=variable_colors["MotorAcuity"], bins=10, figsize=figsize_mm):
    fig, ax = plt.subplots(figsize=mm_to_inch(*figsize))
    ax.hist(values, bins=bins, color=color, edgecolor='black')
    # Overlay points at y=0
    ax.scatter(values, np.zeros_like(values), color='black', s=MARKER_SIZE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis(ax)
    smart_legend(fig, ax)
    plt.tight_layout()
    plt.show()

# -----------------------
# 3. Box Plot
# -----------------------
def box_plot(categories, values, xlabel="Category", ylabel="Value", color=variable_colors["durations (s)"], figsize=figsize_mm):
    fig, ax = plt.subplots(figsize=mm_to_inch(*figsize))
    if isinstance(color, list):
        palette = color
    else:
        palette = [color]*len(np.unique(categories))
    sns.boxplot(x=categories, y=values, palette=palette, ax=ax)
    sns.swarmplot(x=categories, y=values, color='black', size=MARKER_SIZE, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis(ax, x_is_categorical=True)
    smart_legend(fig, ax)
    plt.tight_layout()
    plt.show()

# -----------------------
# 4. Violin Plot
# -----------------------
def violin_plot(categories, values, xlabel="Category", ylabel="Value", color=variable_colors["durations (s)"], figsize=figsize_mm):
    fig, ax = plt.subplots(figsize=mm_to_inch(*figsize))
    sns.violinplot(x=categories, y=values, palette=[color]*len(np.unique(categories)), ax=ax)
    sns.stripplot(x=categories, y=values, color='black', size=MARKER_SIZE, ax=ax, jitter=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis(ax, x_is_categorical=True)
    smart_legend(fig, ax)
    plt.tight_layout()
    plt.show()

# -----------------------
# 5. 3D Plot
# -----------------------
def plot_3d(x, y, z, xlabel="X", ylabel="Y", zlabel="Z", color=variable_colors["ballistic_LDLJ"], figsize=figsize_mm):
    fig = plt.figure(figsize=mm_to_inch(*figsize))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color=color, linewidth=LINE_WIDTH)
    ax.scatter(x, y, z, color='black', s=MARKER_SIZE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.tight_layout()
    plt.show()



# Box Plot of sBBT Scores by Hand using the box_plot function
# Aggregate the data so that each subject contributes one value (using the mean if there are multiple entries)
df_unique = df.groupby(['Subject', 'Hand'], as_index=False)['sBBTResult'].mean()

# Ensure a consistent order for the categories
df_unique['Hand'] = pd.Categorical(df_unique['Hand'], categories=["dominant", "non_dominant"], ordered=True)

# Call the box_plot function.
# Here we supply a list of colors for the palette to match the "dominant" and "non_dominant" order.
box_plot(
    categories=df_unique['Hand'],
    values=df_unique['sBBTResult'],
    xlabel="Hand",
    ylabel="sBBT score",
    color=[hand_colors["dominant"], hand_colors["non_dominant"]]
)



