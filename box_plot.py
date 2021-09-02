import matplotlib as mpl

mpl.rc("savefig", dpi=300)
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (7, 14)

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



models = [
    "rosetta",
    "evoEF2",
    "prodconn",
    "gx[pc]-distance-12-l10",
    "default",
    "default_unbalanced",
]
class_names = ["Mainly Alpha", "Mainly Beta", "Alpha-Beta", "All"]
pdb_by_class_df = pd.read_csv("af2/af2_set_classes.csv")
pdb_by_class_df.sort_values(by="PDB", inplace=True)
pdb_class_map = pdb_by_class_df['class'].to_numpy()

for c in range(1,5):
    # the figure and axes
    if c < 4:
        current_map = pdb_class_map == c
    else:
        # Select all
        current_map = pdb_class_map != 0

    fig, (ax, ax2, ax3) = plt.subplots(nrows=3)
    data_rmsd = []
    data_recall = []
    data_accuracy = []

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
        y = y[current_map]
        x = x[current_map]
        z = z[current_map]
        print(f"There are {len(x)} structures for class {c} - {class_names[c-1]}")
        data_rmsd.append(y)
        data_recall.append(x)
        data_accuracy.append(x)

    SHOW_OUTLIERS = False
    # Do boxplots
    sns.boxplot(data=data_rmsd, ax = ax, showfliers = SHOW_OUTLIERS)
    xtickNames = plt.setp(ax, xticklabels=models)
    plt.setp(xtickNames, rotation=45, fontsize=18)
    ax.set_title(f'{class_names[c-1]}', fontsize=20)
    ax.set_ylabel('RMSD ($\AA$)', fontsize=20)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(ymin=0) #, ymax=np.max(data_rmsd))


    sns.boxplot(data=data_recall, ax = ax2, showfliers = SHOW_OUTLIERS)
    xtickNames = plt.setp(ax2, xticklabels=models)
    plt.setp(xtickNames, rotation=45, fontsize=18)
    ax2.set_ylabel('Macro-Recall (%)', fontsize=20)
    ax2.tick_params(axis='x', labelsize=13)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_ylim(ymin=0)


    sns.boxplot(data=data_accuracy, ax = ax3, showfliers = SHOW_OUTLIERS)
    xtickNames = plt.setp(ax3, xticklabels=models)
    plt.setp(xtickNames, rotation=45, fontsize=18)
    ax3.set_xlabel('Models', fontsize=20)
    ax3.set_ylabel('Accuracy (%)', fontsize=20)
    ax3.tick_params(axis='x', labelsize=13)
    ax3.tick_params(axis='y', labelsize=18)
    ax3.set_ylim(ymin=0) #, ymax=np.max(data_accuracy))

    plt.tight_layout()
    plt.savefig(f"boxplot_{c}.png", dpi=300)
    plt.close()
    del data_rmsd; del data_accuracy; del data_recall
