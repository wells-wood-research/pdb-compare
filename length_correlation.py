import matplotlib as mpl

mpl.rc("savefig", dpi=300)
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (28, 7)

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from scipy import stats


models = [
    "rosetta",
    "evoEF2",
    "prodconn",
    "gx[pc]-distance-12-l10",
    "default",
    "default_unbalanced",
    "densecpd"
]
class_names = ["Mainly Alpha", "Mainly Beta", "Alpha-Beta", "All"]
pdb_by_class_df = pd.read_csv("af2/af2_set_classes.csv")
pdb_by_class_df.sort_values(by="PDB", inplace=True)
pdb_class_map = pdb_by_class_df['class'].to_numpy()
fig, (ax, ax2, ax3) = plt.subplots(ncols=3)
print(sns.color_palette())

data_correlation = []
data_pval = []
sequences = []

for i, model_name in enumerate(models):
    df = pd.read_csv(f"results_{model_name}.csv")
    df.sort_values(by="PDB", inplace=True)
    x = df.sequence.to_numpy()
    x = np.array([len(seq) for seq in x])
    if len(sequences) == 0:
        sequences = x
    y_nan = df.rmsd.to_numpy()
    # Remove NaN
    y = y_nan[~np.isnan(y_nan)]
    x = x[~np.isnan(y_nan)]
    corr_coef, p = stats.pearsonr(x, y)
    data_correlation.append(corr_coef)
    data_pval.append(p)

sns.distplot(x=sequences, ax= ax, kde=False)
print(list(range(1, len(models)+1)))
print(len(models))
print(models)
print(data_correlation)
# sns.barplot(x=data_correlation, ax = ax2,)
# ax2.bar(x=list(range(1, len(models)+1)), height=data_correlation)
ax2.bar(x=np.arange(len(data_correlation)), height=data_correlation)
ax2.set_xticks(np.arange(len(data_correlation)))
ax2.set_xticklabels(models, rotation=90)
ax2.set_ylabel('Pearson Correlation Coefficient', fontsize=20)
ax2.tick_params(axis='x', labelsize=13)
ax2.tick_params(axis='y', labelsize=18)

ax3.bar(x=np.arange(len(data_correlation)), height=data_pval)
ax3.set_xticks(np.arange(len(data_correlation)))
ax3.set_xticklabels(models, rotation=90)
ax3.set_ylabel('p-value', fontsize=20)
ax3.tick_params(axis='x', labelsize=13)
ax3.tick_params(axis='y', labelsize=18)


plt.tight_layout()
plt.savefig(f"correlations.eps", dpi=300)
plt.close()