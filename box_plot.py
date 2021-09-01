import matplotlib as mpl

mpl.rc("savefig", dpi=300)
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (7, 14)

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# the figure and axes
fig, (ax, ax2 )= plt.subplots(nrows=2)

models = [
    "rosetta",
    "gx[pc]-distance-12-l10",
    "default",
    "default_unbalanced",
]

data_rmsd  = []
data_recall  = []

for i, model_name in enumerate(models):
    df = pd.read_csv(f"results_{model_name}.csv")
    x = df.recall.to_numpy() * 100
    y_nan = df.rmsd.to_numpy()
    y = y_nan[~np.isnan(y_nan)]
    x = x[~np.isnan(y_nan)]
    data_rmsd.append(y)
    data_recall.append(x)


# Do boxplots
sns.boxplot(data=data_rmsd, ax = ax)
xtickNames = plt.setp(ax, xticklabels=models)
plt.setp(xtickNames, rotation=45, fontsize=18)
ax.set_xlabel('Model Names', fontsize=20)
ax.set_ylabel('RMSD ($\AA$)', fontsize=20)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=18)


sns.boxplot(data=data_recall, ax = ax2)
xtickNames = plt.setp(ax2, xticklabels=models)
plt.setp(xtickNames, rotation=45, fontsize=18)
ax2.set_xlabel('Model Names', fontsize=20)
ax2.set_ylabel('Macro-Recall (%)', fontsize=20)
ax2.tick_params(axis='x', labelsize=13)
ax2.tick_params(axis='y', labelsize=18)
plt.tight_layout()
plt.show()
