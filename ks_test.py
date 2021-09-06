import numpy as np
import pandas as pd
from scipy import stats



if __name__ == '__main__':

    rosetta = pd.read_csv(f"results_default_unbalanced.csv")
    rosetta_x = rosetta.recall.to_numpy() * 100
    rosetta_y_nan = rosetta.rmsd.to_numpy()
    rosetta_y = rosetta_y_nan[~np.isnan(rosetta_y_nan)]
    rosetta_x = rosetta_x[~np.isnan(rosetta_y_nan)]
    rosetta_2d = np.hstack((rosetta_x, rosetta_y))

    gx = pd.read_csv(f"results_gx[pc]-distance-12-l10.csv")
    gx_x = gx.recall.to_numpy() * 100
    gx_y_nan = gx.rmsd.to_numpy()
    gx_y = gx_y_nan[~np.isnan(gx_y_nan)]
    gx_x = gx_x[~np.isnan(gx_y_nan)]
    gx_2d = np.hstack((gx_x, gx_y))

    timed = pd.read_csv(f"results_default.csv")
    timed_x = timed.recall.to_numpy() * 100
    timed_y_nan = timed.rmsd.to_numpy()
    timed_y = timed_y_nan[~np.isnan(timed_y_nan)]
    timed_x = timed_x[~np.isnan(timed_y_nan)]
    timed_2d = np.hstack((timed_x, timed_y))

    densecpd = pd.read_csv(f"results_densecpd.csv")
    densecpd_x = densecpd.recall.to_numpy() * 100
    densecpd_y_nan = densecpd.rmsd.to_numpy()
    densecpd_y = densecpd_y_nan[~np.isnan(densecpd_y_nan)]
    densecpd_x = densecpd_x[~np.isnan(densecpd_y_nan)]
    densecpd_2d = np.hstack((densecpd_x, densecpd_y))

    print("---UNBALANCED-TIMED---")
    print("ks")
    print(stats.ks_2samp(rosetta_2d, timed_2d,alternative='two-sided'))
    print(stats.ks_2samp(rosetta_y, timed_y,alternative='two-sided'))
    print(stats.ks_2samp(rosetta_x, timed_x,alternative='two-sided'))
    print("Ttest")
    print(stats.ttest_rel(rosetta_2d, timed_2d))
    print(stats.ttest_rel(rosetta_y, timed_y))
    print(stats.ttest_rel(rosetta_x, timed_x))
    print("---UNBALANCED-GX---")
    print("ks")
    print(stats.ks_2samp(rosetta_2d, gx_2d,alternative='two-sided'))
    print(stats.ks_2samp(rosetta_y, gx_y,alternative='two-sided'))
    print(stats.ks_2samp(rosetta_x, gx_x,alternative='two-sided'))
    print("Ttest")
    print(stats.ttest_rel(rosetta_2d, gx_2d))
    print(stats.ttest_rel(rosetta_y, gx_y))
    print(stats.ttest_rel(rosetta_x, gx_x))
    print("---TIMED-GX---")
    print("ks")
    print(stats.ks_2samp(timed_2d, gx_2d,alternative='two-sided'))
    print(stats.ks_2samp(timed_y, gx_y,alternative='two-sided'))
    print(stats.ks_2samp(timed_x, gx_x,alternative='two-sided'))
    print("Ttest")
    print(stats.ttest_rel(gx_2d, timed_2d))
    print(stats.ttest_rel(gx_y, timed_y))
    print(stats.ttest_rel(gx_x, timed_x))
    print("---gx-dense---")
    print("ks")
    print(stats.ks_2samp(timed_y, densecpd_y,alternative='two-sided'))
    print("Ttest")
    print(stats.ttest_rel(gx_y, densecpd_y))
    print(stats.ttest_ind(gx_y, densecpd_y))
    print("---unbalanced-dense---")
    print("ks")
    print(stats.ks_2samp(rosetta_y, densecpd_y,alternative='two-sided'))
    print("Ttest")
    print(stats.ttest_rel(rosetta_y, densecpd_y))
    print(stats.ttest_ind(rosetta_y, densecpd_y))
    print("---timed-dense---")
    print("ks")
    print(stats.ks_2samp(timed_y, densecpd_y,alternative='two-sided'))
    print("Ttest")
    print(stats.ttest_rel(timed_y, densecpd_y))
    print(stats.ttest_ind(timed_y, densecpd_y))
