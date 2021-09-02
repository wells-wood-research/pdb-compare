import pandas as pd
import numpy as np
from pathlib import Path
import json
from ampal import amino_acids

res_encoder = dict(zip(amino_acids.standard_amino_acids.keys(), range(len(amino_acids.standard_amino_acids.keys()))))

densecpd_files = Path("performance/dense_cpd/").glob('**/*')
densecpd_predictions = {}
densecpd_encoded_predictions = {}

for densecpd_file in densecpd_files:
    pdb_code = densecpd_file.name.split(".pred.txt")[0]
    current_file = pd.read_csv(densecpd_file, delimiter=" ", header=None, skiprows=1)
    current_file.dropna(inplace=True, axis=1)
    # Select chain A
    current_file = current_file[current_file[2] == "A"]
    residue_sequence = current_file[4].to_numpy()
    residue_sequence = "".join(residue_sequence)
    # Encode sequence to get accuracy and macro-recall from benchmark
    current_sequence = []
    for res in residue_sequence:
        idx = res_encoder[res]
        current_encoding = np.zeros(20,)
        current_encoding[idx] = 1
        current_sequence.append(current_encoding)
    current_sequence = np.array(current_sequence)

    densecpd_encoded_predictions[pdb_code] = current_sequence
    densecpd_predictions[pdb_code] = residue_sequence



with open("monomers_af_densecpd.json", "w") as outfile:
    json.dump(densecpd_predictions, outfile)

all_predictions = []
all_count = 0

with open("densecpd.txt", "w") as datasetmap:
    datasetmap.write("ignore_uncommon False\n")
    datasetmap.write("include_pdbs\n")
    datasetmap.write("##########\n")
    for pdb_key, pdb_seq in densecpd_predictions.items():
        datasetmap.write(f"{pdb_key}A {len(pdb_seq)}\n")
        all_predictions += densecpd_encoded_predictions[pdb_key].tolist()
        all_count += len(pdb_seq)
all_predictions = np.array(all_predictions)
assert len(all_predictions) == all_count
# print(all_predictions.shape)
# all_predictions.reshape((20,))
# print(all_predictions)
np.savetxt( "densecpd.csv", all_predictions, delimiter=",")