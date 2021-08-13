import pymol
import pandas as pd
import numpy
import urllib
from pathlib import Path
import numpy as np

import matplotlib as mpl
mpl.rc("savefig", dpi=300)
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (10, 10)
model_name1 = "skilled-wind-134-timed_hs1000_l2_dp03_weighted"
model_name2 = "default"
df1 = pd.read_csv(f"results_{model_name1}.csv")
df2 = pd.read_csv(f"results_{model_name2}.csv")
# df1.dropna(inplace=True)
# df2.dropna(inplace=True)

plt.scatter(df1.rmsd, df2.rmsd, alpha=0.5)
# m, b = np.polyfit(df1.rmsd[df1.rmsd > 0],df2.rmsd[df2.rmsd > 0], 1)
# plt.plot(df1.rmsd, m*df1.rmsd + b)

plt.xlim(xmin=0, xmax=25)
plt.ylim(ymin=0, ymax=25)
plt.plot([0, 25], [0, 25], ls='--')
plt.ylabel('TIMED Default RMSD ($\AA$)',  fontsize=20)
plt.xlabel('Skilled-Wind RMSD ($\AA$)',  fontsize=20)
# plt.text(0.5*0.9, 2.2, '2 ($\AA$)', fontsize=18)

plt.savefig(f"{model_name1}_{model_name2}_rmsd_comparison.pdf")
plt.close()