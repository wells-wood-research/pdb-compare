import matplotlib as mpl

mpl.rc("savefig", dpi=300)

nice_fonts = {
    # "text.usetex": True,
    "font.family": "serif",
    "font.serif" : "Times New Roman",
}
mpl.rcParams.update(nice_fonts)
import matplotlib.pyplot as plt
# plt.rc('legend',fontsize=20) # using a size in points

plt.style.use("ggplot")



# plt.rcParams["figure.figsize"] = (10, 5)

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio / 2

    return fig_width_in, fig_height_in

# models = [
#     "rosetta",
#     "evoEF2",
#     "prodconn",
#     "gx[pc]-distance-12-l10",
#     "default",
#     "default_unbalanced",
#     "densecpd"
# ]

models = [
    "evoEF2",
    "prodconn",
    # "prodconn_unbalanced",
    "default",
    "default_unbalanced",
    "gx[pc]-distance-12-l10",
    "rosetta",
    "densecpd",

]

models_better_names = [
    "EvoEF2",
    "ProDCoNN",
    # "prodconn_unbalanced",
    "TIMED",
    "TIMED-Unbalanced",
    "GX[PC]",
    "Rosetta",
    "DenseCPD",

]
class_names = ["Mainly Alpha", "Mainly Beta", "Alpha-Beta", "All"]
pdb_by_class_df = pd.read_csv("af2/af2_set_classes.csv")
pdb_by_class_df.sort_values(by="PDB", inplace=True)
pdb_class_map = pdb_by_class_df['class'].to_numpy()
fig, (ax, ax2, ax3, ax4) = plt.subplots(figsize=set_size(505.89*1.5), ncols=4)

axes = [ax, ax2, ax3, ax4]
for c in range(1,5):
    # the figure and axes
    if c < 4:
        current_map = pdb_class_map == c
    else:
        # Select all
        current_map = pdb_class_map != 0

    data_rmsd = []
    data_recall = []
    data_accuracy = []
    median = 0
    for i, model_name in enumerate(models):
        df = pd.read_csv(f"results_{model_name}.csv")
        df.sort_values(by="PDB", inplace=True)
        x = df.recall.to_numpy() * 100
        z = df.accuracy.to_numpy() * 100
        y_nan = df.rmsd.to_numpy()
        # Remove NaN
        y = y_nan[~np.isnan(y_nan)]
        x = x[~np.isnan(y_nan)]
        z = z[~np.isnan(y_nan)]
        # Select current class:
        # try:
        y = y[current_map]
        x = x[current_map]
        z = z[current_map]
        # print(x)
        # raise ValueError
        # except:
        #     pass
        # print(f"There are {len(x)} structures for class {c} - {class_names[c-1]}")
        print(f"{model_name} has {len(y[y < 2])} for class {class_names[c-1]}")

        data_rmsd.append(y)
        data_recall.append(x)
        data_accuracy.append(x)
        if model_name == "evoEF2":
            median = np.median(y)

        cumulative_rmsd = []
        thresholds = np.arange(0, 10, 2)

        for idx, t in enumerate(thresholds):

            if idx == 0:
                percentage = ((len(y[y < t])) / len(y))
            else:
                percentage = ((len(y[y < t])-cumulative_rmsd[-1]) / len(y))
            cumulative_rmsd.append(percentage)
        if model_name == "default_unbalanced":
            axes[c-1].plot(thresholds, cumulative_rmsd, label=models_better_names[i], linewidth=5)
        else:
            axes[c-1].plot(thresholds, cumulative_rmsd, label=models_better_names[i], linewidth=5)

    # xtickNames = plt.setp(axes[c-1], xticklabels=models)
    # plt.setp(xtickNames, rotation=90, fontsize=5)
    axes[c-1].set_title(f'{class_names[c-1]}') #, fontsize=20)
    # ax.set_xlabel('RMSD ($\AA$)') #, fontsize=20)
    axes[0].set_ylabel('% Structures (%)') #, fontsize=20)
    # axes[0].set_ylabel('Density') #, fontsize=20)
    # axes[c-1].tick_params(axis='x') #, labelsize=18)
    axes[c-1].tick_params(axis='y', labelsize=13)
    axes[c-1].yaxis.set_major_locator(plt.MaxNLocator(3))
    axes[c-1].xaxis.set_major_locator(plt.MaxNLocator(3))
    # axes[c-1].axvline(2, color='k', linestyle='dashed', linewidth=1)
    # axes[c-1].text(50 * 0.9, 2.2, '2 ($\AA$)', fontsize=18)
    # axes[c-1].yaxis.set_major_locator(plt.MaxNLocator(3))

    # axes[c-1].set_xticklabels(tick_labels.astype(int))
    # axes[c-1].axes.xaxis.set_visible(False)
    # axes[c-1].set_yticklabels([])
    # axes[c-1].set_xticklabels([])
    axes[c-1].set_ylim(ymin=0) #, ymax=np.max(data_rmsd))
    axes[c-1].set_xlim(xmin=0, xmax=6) #, ymax=np.max(data_rmsd))

    #
    # sns.boxplot(data=data_recall, ax = ax2, showfliers = SHOW_OUTLIERS)
    # xtickNames = plt.setp(ax2, xticklabels=models)
    # plt.setp(xtickNames, rotation=45, fontsize=18)
    # ax2.set_ylabel('Macro-Recall (%)', fontsize=20)
    # ax2.tick_params(axis='x', labelsize=13)
    # ax2.tick_params(axis='y', labelsize=18)
    # ax2.set_ylim(ymin=0)
    #
    #
    # sns.boxplot(data=data_accuracy, ax = ax3, showfliers = SHOW_OUTLIERS)
    # xtickNames = plt.setp(ax3, xticklabels=models)
    # plt.setp(xtickNames, rotation=45, fontsize=18)
    # ax3.set_xlabel('Models', fontsize=20)
    # ax3.set_ylabel('Accuracy (%)', fontsize=20)
    # ax3.tick_params(axis='x', labelsize=13)
    # ax3.tick_params(axis='y', labelsize=18)
    # ax3.set_ylim(ymin=0) #, ymax=np.max(data_accuracy))


    del data_rmsd; del data_accuracy; del data_recall
fig.text(0.5, 0.01, 'RMSD ($\AA$)', ha='center')
# plt.legend()
legend = plt.legend(ncol = 4, framealpha=1, frameon=False, bbox_to_anchor=(0.5, -0.15))


def export_legend(legend, filename="1jjiflegend_smal.eps", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=300, bbox_inches=bbox)

export_legend(legend)
# plt.legend([],[], frameon=False)
#
# fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical')
plt.tight_layout()
# plt.savefig(f"hist_RMSD.eps", dpi=300)
plt.close()