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

model_name = "dashing-water-52-weighted_p1"
# For loop does not work - ffs pymol
pdb_structures = ["3e4g", "2j8k", "2v3i"]
pdb_name = pdb_structures[2]
original = f"{pdb_name}.pdb.gz"
predicted = f"af2/{model_name}/{pdb_name}A_unrelaxed_model_1.pdb"
pymol.pymol_argv = ['pymol', '-qc']
pymol.finish_launching()
cmd = pymol.cmd
cmd.delete('all')
cmd.load(original)
cmd.load(predicted)
print(cmd.get_object_list('all'))
sel1, sel2 = cmd.get_object_list('all')
# Color white (original) cyan (prediction)
cmd.color("magenta", sel1)
cmd.color("cyan", sel2)
# Align structures
cmd.align(sel1, sel2)
# Hide waters
cmd.select(name="h2o", selection="solvent")
cmd.hide(selection="h2o")
# Hide 3D shadows
cmd.set("ray_shadows", value="off")
cmd.reset()
# Export HD
cmd.ray(2400, 2400)
cmd.png(f"af2/{model_name}_{pdb_name}.png", width=2400, height=2400, dpi=300, ray=1)
