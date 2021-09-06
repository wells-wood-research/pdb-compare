import matplotlib as mpl

mpl.rc("savefig", dpi=300)
import matplotlib.pyplot as plt

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
    "default",
    "densecpd",

]

models_better_names = [
    "TIMED",
    "DenseCPD",
]
from pathlib import Path
path_to_dense = Path("5zjb/5zjb_densecpd_2.csv")
path_to_timed = Path("5zjb/5zjb_default.csv")

dense_seq = np.genfromtxt(path_to_dense, delimiter=",")
timed_seq = np.genfromtxt(path_to_timed, delimiter=",")

from scipy.stats import entropy
entropy_dense = entropy(dense_seq, axis=1)
entropy_timed = entropy(timed_seq, axis=1)
entropy_msa = [1.0377968161163391,2.2729290821491057,2.1147491324313927,2.4123854955698283,2.27416670789597,2.3019457984405416,1.6675164962003461,2.443077408101581,2.5250952227011867,2.310803268989851,2.3726914549665747,2.199549878734736,2.2739364153518777,2.68456330168803,2.2349514865939466,2.204576718375552,2.5011483124220506,2.205997467328648,0.6749754468353846,2.2833972760335906,2.1504478375397458,1.4517594052022404,1.601300993745962,2.032767652680053,2.0379383900075254,1.782743842549571,1.6029943772003619,0.9184277849930976,1.7716594728726258,1.5678562531603566,1.6307758335027516,2.128506280467958,2.087300492811178,1.9424982257525427,2.225165691233162,2.528603044357194,1.9908269602777784,2.1156913034767713,1.7310302411074623,1.428367252442779,1.6414627336687557,1.3801087571572686,1.3208883431493221,0.6365141682948128,2.330902229794933,2.547830293370813,2.051925051351247,1.5794272443558546,2.15751462999086,2.336768711344823,1.4514626599095037,1.3522093992338284,2.1291412404266676,2.2817747384837612,2.004750391762983,2.3400411174134046,1.5614694247763998,2.162692600653636,1.0211119642926534,1.5574075732504105,1.6886309589181236,1.0856968066493662,1.559743138414276,1.7529907208585989,1.7485659532324227,1.9720632647736307,1.7784552666792037,0.5004024235381879,0.8828275435751076,1.5492346598417437,2.011242441972426,1.6766117345910303,2.2563971867305868,1.9891854889035225,2.330366972046178,1.4195483833026965,1.5067537785671425,1.3242687643049527,1.1762150811013816,2.037818523834492,2.082941950687939,2.30315913798801,1.15374194270109,0.5982695885852573,1.948880880246993,2.0678168498841383,2.1169132283532557,1.8360054189984831,2.0759273734448094,2.42361408434331,2.4007670074477767,2.2712472340136216,2.072547247790487,0.0,1.8891591637540215,2.596501064054183,2.2427743496764103,2.2426429667334915,2.200953960822064,2.14804947272297,1.8363975383686482,1.7789332373471316,2.061306598079271,2.4807851169539257,2.3041710080653237,2.601956798773736,1.7354555193825791,0.0,2.3223358226248685,2.439486441835435,2.4801911878156293,2.218579394020403,1.9770845129875236,2.1688839045597503,1.9479312469542778,1.792191339512752,1.7900357689791353,2.173177677957423,1.4992747143566456,1.6581448397825276,2.2669507874942747,2.3189613014795616,1.8954541066191957,2.196177998573231,2.205226809843608,2.132580813216396,1.9040611632106392,1.873302955064842,1.5509255497174899,1.3732079433050635,1.7996721394887887,0.13906079845474112,1.9418591382817,2.2165327190698996,2.4344766644591633,2.405934182369988,2.536590950802218,2.3703342259824587,1.7120183419242805,2.2519395058022837,2.0455764621533223,0.9921068275686908,0.9911388966130941,0.2509548043576197,1.7198636806607,0.8740052482466639,1.8541345532456779,2.0124395596752374,2.1193697207410684,2.188067132045608,2.178367235227307,2.358773579699079,2.300863909316841,2.653874411597873,2.0548627122420298,2.301708756353272,0.8922997058907158,0.5818786450459387,2.1412564082551246,2.4718149915218928,2.6251483874911585,2.3934307376403137,2.57250537884719,2.0774205453839327,2.475477628140866,1.1285962936622183,1.50213608609538,2.513231602993532,2.4446972856773646,2.4435045555968844,2.2585486024726533,1.9952803187919035,1.806894205856499,2.2730648630312165,2.2618886830542415,2.3029098147715708,1.4692846777207589,0.8808122476172401,1.7994045660411897,1.9179946917372275,1.8095142558492818,1.7955384097258926,2.226119274171139,2.125413741288863,1.4827465533719468,1.9913827130633446,2.1068648551575837,2.117693839301262,1.9027727404413755,1.682201706597402,2.0512714715847533,1.894875624562606,2.567008290670651,2.5015527172153913,0.8903644334095999,2.317247506176903,1.714201275436734,1.4919024590056067,1.7489626018127462,1.312235659519078,2.4025693200624687,2.37358600681322,2.408609731211266,1.9409291009401883,2.1953274157454223,2.1289139892656097,2.243249093416903,2.3371160765857835,0.0,1.0114042647073516,1.9644064443657356,2.3353403981172796,2.095399690557417,0.605797499372304,2.4028863758658887,2.229398289892626,2.3791359023037426,1.294298129968764,1.3143876752866568,0.0,2.2827580015589604,2.090930487357486,2.0638841737390865,2.19893395753223,2.033503271597643,1.5480428938453379,0.0,0.6730116670092565,2.5533570491545716,2.4871406436666215,2.339434352770067,1.9760870455727535]


assert len(dense_seq) == len(entropy_dense) == len(entropy_timed)

fig = plt.figure(figsize=set_size(505.89))
ax = plt.axes()

ax.plot(np.arange(1, len(dense_seq)+1), entropy_dense, label="DenseCPD")
ax.plot(np.arange(1, len(dense_seq)+1), entropy_timed, label="TIMED")
from scipy import stats
print(stats.pearsonr(entropy_dense, entropy_timed))
print(stats.pearsonr(entropy_dense, entropy_msa))
print(stats.pearsonr(entropy_timed, entropy_msa))


ax2=ax.twinx()

ax2.plot(np.arange(1, len(dense_seq)+1), entropy_msa, label="MSA", color="green", linestyle="--")

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

plt.tight_layout()
plt.show()
#
#
#
#     for i, model_name in enumerate(models):
#         df = pd.read_csv(f"results_{model_name}.csv")
#         df.sort_values(by="PDB", inplace=True)
#         x = df.recall.to_numpy() * 100
#         z = df.accuracy.to_numpy() * 100
#         y_nan = df.rmsd.to_numpy()
#         # Remove NaN
#         y = y_nan[~np.isnan(y_nan)]
#         x = x[~np.isnan(y_nan)]
#         z = z[~np.isnan(y_nan)]
#         # Select current class:
#         # try:
#         y = y[current_map]
#         x = x[current_map]
#         z = z[current_map]
#         # print(x)
#         # raise ValueError
#         # except:
#         #     pass
#         # print(f"There are {len(x)} structures for class {c} - {class_names[c-1]}")
#         print(f"{model_name} has {len(y[y < 2])} for class {class_names[c-1]}")
#
#         data_rmsd.append(y)
#         data_recall.append(x)
#         data_accuracy.append(x)
#         if model_name == "evoEF2":
#             median = np.median(y)
#
#         cumulative_rmsd = []
#         thresholds = np.arange(0, 10, 2)
#
#         for idx, t in enumerate(thresholds):
#
#             if idx == 0:
#                 percentage = ((len(y[y < t])) / len(y))
#             else:
#                 percentage = ((len(y[y < t])-cumulative_rmsd[-1]) / len(y))
#             cumulative_rmsd.append(percentage)
#         if model_name == "default_unbalanced":
#             axes[c-1].plot(thresholds, cumulative_rmsd, label=models_better_names[i], linewidth=5,)
#         else:
#             axes[c-1].plot(thresholds, cumulative_rmsd, label=models_better_names[i])
#
#     # xtickNames = plt.setp(axes[c-1], xticklabels=models)
#     # plt.setp(xtickNames, rotation=90, fontsize=5)
#     axes[c-1].set_title(f'{class_names[c-1]}') #, fontsize=20)
#     # ax.set_xlabel('RMSD ($\AA$)') #, fontsize=20)
#     axes[0].set_ylabel('% Structures (%)') #, fontsize=20)
#     # axes[0].set_ylabel('Density') #, fontsize=20)
#     # axes[c-1].tick_params(axis='x') #, labelsize=18)
#     # axes[c-1].tick_params(axis='y') #, labelsize=18)
#     axes[c-1].yaxis.set_major_locator(plt.MaxNLocator(3))
#     axes[c-1].xaxis.set_major_locator(plt.MaxNLocator(3))
#     # axes[c-1].axvline(2, color='k', linestyle='dashed', linewidth=1)
#     # axes[c-1].text(50 * 0.9, 2.2, '2 ($\AA$)', fontsize=18)
#     # axes[c-1].yaxis.set_major_locator(plt.MaxNLocator(3))
#
#     # axes[c-1].set_xticklabels(tick_labels.astype(int))
#     # axes[c-1].axes.xaxis.set_visible(False)
#     # axes[c-1].set_yticklabels([])
#     # axes[c-1].set_xticklabels([])
#     axes[c-1].set_ylim(ymin=0) #, ymax=np.max(data_rmsd))
#     axes[c-1].set_xlim(xmin=0, xmax=6) #, ymax=np.max(data_rmsd))
#
#     #
#     # sns.boxplot(data=data_recall, ax = ax2, showfliers = SHOW_OUTLIERS)
#     # xtickNames = plt.setp(ax2, xticklabels=models)
#     # plt.setp(xtickNames, rotation=45, fontsize=18)
#     # ax2.set_ylabel('Macro-Recall (%)', fontsize=20)
#     # ax2.tick_params(axis='x', labelsize=13)
#     # ax2.tick_params(axis='y', labelsize=18)
#     # ax2.set_ylim(ymin=0)
#     #
#     #
#     # sns.boxplot(data=data_accuracy, ax = ax3, showfliers = SHOW_OUTLIERS)
#     # xtickNames = plt.setp(ax3, xticklabels=models)
#     # plt.setp(xtickNames, rotation=45, fontsize=18)
#     # ax3.set_xlabel('Models', fontsize=20)
#     # ax3.set_ylabel('Accuracy (%)', fontsize=20)
#     # ax3.tick_params(axis='x', labelsize=13)
#     # ax3.tick_params(axis='y', labelsize=18)
#     # ax3.set_ylim(ymin=0) #, ymax=np.max(data_accuracy))
#
#
#     del data_rmsd; del data_accuracy; del data_recall
# fig.text(0.5, 0.01, 'RMSD ($\AA$)', ha='center')
# # plt.legend()
# # legend = plt.legend(ncol = 4, framealpha=1, frameon=True, bbox_to_anchor=(0.5, -0.15))
#
#
# def export_legend(legend, filename="legend.eps", expand=[-5,-5,5,5]):
#     fig  = legend.figure
#     fig.canvas.draw()
#     bbox  = legend.get_window_extent()
#     bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
#     bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(filename, dpi="figure", bbox_inches=bbox)
#
# # export_legend(legend)
# # plt.legend([],[], frameon=False)
# #
# # fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical')
# plt.tight_layout()
# plt.show()
# plt.savefig(f"hist_RMSD.eps", dpi=300)
# plt.close()