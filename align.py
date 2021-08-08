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

pymol.pymol_argv = ['pymol', '-qc']
pymol.finish_launching()
cmd = pymol.cmd


def gdt(model1, model2):
    cmd.delete('all')
    cmd.load(model1)
    cmd.load(model2)
    print(cmd.get_object_list('all'))
    sel1, sel2 = cmd.get_object_list('all')

    # Select only C alphas
    sel1 += ' and name CA'
    sel2 += ' and name CA'
    gdts = []  # Store GDT at the different thresholds
    cutoffs = [1., 2., 4., 8.]
    cmd.align(sel1, sel2, cycles=0, transform=0, object='aln')
    mapping = cmd.get_raw_alignment(
        'aln')  # e.g. [[('prot1', 2530), ('prot2', 2540)], ...]
    distances = []
    for mapping_ in mapping:
        atom1 = '%s and id %d' % (mapping_[0][0], mapping_[0][1])
        atom2 = '%s and id %d' % (mapping_[1][0], mapping_[1][1])
        dist = cmd.get_distance(atom1, atom2)
        cmd.alter(atom1, 'b = %.4f' % dist)
        distances.append(dist)
    distances = numpy.asarray(distances)
    # gdts = []
    for cutoff in cutoffs:
        gdts.append((distances <= cutoff).sum() / float(len(distances)))
    out = numpy.asarray(zip(cutoffs, gdts)).flatten()
    # print "GDT_%d: %.4f; GDT_%d: %.4f; GDT_%d: %.4f; GDT_%d: %.4f;"%tuple(out)
    print(gdts)
    GDT = numpy.mean(gdts)
    # print "GDT: %.4f"%GDT
    # cmd.spectrum('b', 'green_yellow_red', sel1)
    RMSD = cmd.align(sel1, sel2)[0]

    return GDT, RMSD


def fetch_pdb(
        pdb_code: str,
        output_folder: Path,
        pdb_request_url: str = "https://files.rcsb.org/download/",
        is_pdb: bool = True,
) -> None:
    """
    Downloads a specific pdb file into a specific folder.
    Parameters
    ----------
    pdb_code : str
        Code of the PDB file to be downloaded.
    output_folder : Path
        Output path to save the PDB file.
    pdb_request_url : str
        Base URL to download the PDB files.
    is_pdb:bool=False
        If True, get .pdb, else get biological assembly.
    """
    if is_pdb:
        pdb_code_with_extension = f"{pdb_code[:4]}.pdb.gz"
    else:
        pdb_code_with_extension = f"{pdb_code[:4]}.pdb1.gz"
    print(f'{pdb_code_with_extension} is missing and will be downloaded!')
    urllib.request.urlretrieve(pdb_request_url + pdb_code_with_extension,
                               filename=output_folder / pdb_code_with_extension)

model_name = "skilled-wind-134-timed_hs1000_l2_dp03_weighted"
df = pd.read_csv(f"performance/{model_name}_performance.csv")
codes = df.PDB.values + df.chain.values
rmsd_scores = []
gdt_scores = []

# for x in codes:
#     if Path(f"{x}.pdb.gz").exists():
#         continue
#     else:
#         fetch_pdb(x[:-1],Path(''))

#check all files exist
pdb_paths = []
for x in codes:
    if not Path(f"af2/{model_name}/{x}_unrelaxed_model_1.pdb").exists():
        print(x)


for x in codes:
    try:
        g,r = gdt(f"{x[:-1]}.pdb.gz", f"af2/{model_name}/{x}_unrelaxed_model_1.pdb")
        rmsd_scores.append(r)
        gdt_scores.append(g)
    except:
        rmsd_scores.append(np.nan)
        gdt_scores.append(np.nan)

df['rmsd']=rmsd_scores
df['gdt']=gdt_scores
df.to_csv(f'results_{model_name}.csv')

import matplotlib.pyplot as plt


df.dropna()
plt.scatter(df.accuracy, df.rmsd, alpha=0.8)
plt.axhline(2, color='k', linestyle='dashed', linewidth=1)
plt.xlim(xmin=0, xmax=0.55)
plt.ylim(ymin=0, ymax=45)
plt.ylabel('RMSD ($\AA$)',  fontsize=20)
plt.xlabel('Accuracy (%)',  fontsize=20)
plt.text(0.5*0.9, 2.2, '2 ($\AA$)', fontsize=18)

plt.savefig(f"{model_name}_accuracy.eps")
plt.close()

plt.scatter(df.recall, df.rmsd, alpha=0.8, color="yellowgreen")
plt.ylabel('RMSD ($\AA$)',  fontsize=20)
plt.xlabel('Macro-Recall (%)',  fontsize=20)
plt.axhline(2, color='k', linestyle='dashed', linewidth=1)
plt.text(0.5*0.9, 2.2,'2 ($\AA$)', fontsize=18)
plt.xlim(xmin=0, xmax=0.55)
plt.ylim(ymin=0, ymax=45)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(f"{model_name}_recall.eps")
plt.close()

df.rmsd.hist(bins=20, color="blueviolet",)
plt.ylabel('Number of structures',  fontsize=20)
plt.xlabel('RMSD ($\AA$)',  fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(xmin=0)

plt.savefig(f'{model_name}_hist.eps')
