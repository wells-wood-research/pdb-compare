import matplotlib as mpl

mpl.rc("savefig", dpi=300)
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (7, 14)

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np


def boxplot_2d(x, y, ax, color, whis=1.5, ):
    """
    Code from https://stackoverflow.com/questions/53849636/draw-a-double-box-plot-chart-2-axes-box-plot-box-plot-correlation-diagram-in

    Parameters
    ----------
    x
    y
    ax
    whis

    Returns
    -------

    """
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0], ylimits[0]),
        (xlimits[2] - xlimits[0]),
        (ylimits[2] - ylimits[0]),
        ec="k",
        zorder=0,
        color=color,
        alpha=0.5,
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1], xlimits[1]], [ylimits[0], ylimits[2]], color="k", zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0], xlimits[2]], [ylimits[1], ylimits[1]], color="k", zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]], [ylimits[1]], color="k", marker="o")

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2] - xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0] - whis * iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1], ylimits[1]], color="k", zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D([left, left], [ylimits[0], ylimits[2]], color="k", zorder=1)
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2] + whis * iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1], ylimits[1]], color="k", zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D([right, right], [ylimits[0], ylimits[2]], color="k", zorder=1)
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2] - ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0] - whis * iqr])
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [bottom, ylimits[0]], color="k", zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [bottom, bottom], color="k", zorder=1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2] + whis * iqr])
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [top, ylimits[2]], color="k", zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D([xlimits[0], xlimits[2]], [top, top], color="k", zorder=1)
    ax.add_line(whisker_bar)
    ax.grid(False)
    ##outliers
    # mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    # ax.scatter(
    #     x[mask],y[mask],
    #     facecolors='none', edgecolors='k'
    # )


# the figure and axes
fig, (ax, ax2 )= plt.subplots(nrows=2)

models = [
    "rosetta",
    "gx[pc]-distance-12-l10",
    "default",

]
colors = ["#FFC20A", "#0C7BDC",  "#cc79a7",  "#4b4346"]
# colors = [  "#cc79a7",  "#4b4346"]

for i, model_name in enumerate(models):
    df = pd.read_csv(f"results_{model_name}.csv")
    x = df.recall.to_numpy() * 100
    y_nan = df.rmsd.to_numpy()
    y = y_nan[~np.isnan(y_nan)]
    x = x[~np.isnan(y_nan)]
    print(len(x))
    print(len(y))

    # doing the box plot
    boxplot_2d(x, y, ax=ax, whis=1, color=colors[i])
    ax.set_ylabel('RMSD ($\AA$)', fontsize=20)
    ax.set_xlabel('Macro-Recall (%)',  fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax2.scatter(x, y, alpha=0.5, color=colors[i], label=model_name ) #, edgecolors='b')
    ax2.set_ylabel('RMSD ($\AA$)', fontsize=20)
    ax2.set_xlabel('Macro-Recall (%)',  fontsize=20)
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    plt.axhline(2, color='k', linestyle='dashed', linewidth=1)
    plt.text(50*0.9, 2.2,'2 ($\AA$)', fontsize=18)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=4)

plt.savefig("boxplot.png", dpi=300)
plt.show()
