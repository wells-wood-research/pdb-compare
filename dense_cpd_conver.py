import pandas as pd
import numpy as np
from pathlib import Path
import json
from ampal import amino_acids

res_encoder = dict(zip(amino_acids.standard_amino_acids.keys(), range(len(amino_acids.standard_amino_acids.keys()))))
benchmark_structures_path = Path("../sequence-recovery-benchmark/dataset_visualization/crystal_structure_benchmark.txt")
benchmark_structures = pd.read_csv(benchmark_structures_path, delimiter="\n", header=None).to_numpy().flatten().tolist()
benchmark_dict = {}

for b in benchmark_structures:
    pdb = b[:4]
    chain = b[-1]
    benchmark_dict[pdb] = chain

densecpd_files = Path("performance/dense_cpd/").glob('**/*')
densecpd_predictions = {}
densecpd_encoded_predictions = {}

for densecpd_file in densecpd_files:
    if ".pred" in densecpd_file.name:
        pdb_code = densecpd_file.name.split(".pred")[0]
        current_file = pd.read_csv(densecpd_file, delimiter=" ", header=None)
        current_file.dropna(inplace=True, axis=1)
        current_chain = benchmark_dict[pdb_code]
        # Select chain
        current_file = current_file[current_file[2] == current_chain]
        # Convert to numeric to avoid stupid sorting
        current_file[current_file.columns[1]] = pd.to_numeric(current_file[current_file.columns[1]])
        current_file.sort_values(by=current_file.columns[1], inplace=True)
        residue_sequence = current_file[4].to_numpy()
        residue_sequence = "".join(residue_sequence)
        # Encode sequence to get accuracy and macro-recall from benchmark
        current_sequence = []
        for res in residue_sequence:
            idx = res_encoder[res]
            current_encoding = np.zeros(20, dtype=float)
            current_encoding[idx] = 1.0
            current_sequence.append(current_encoding)
        current_sequence = np.array(current_sequence)

        densecpd_encoded_predictions[pdb_code] = current_sequence
        densecpd_predictions[pdb_code] = residue_sequence


# with open("monomers_af_densecpd.json", "w") as outfile:
#     json.dump(densecpd_predictions, outfile)
af2_pred = {}
af2_remaining = ["3e3vA","5dicA","1dvoA","3cxbA","3rf0A","2ra1A","1a41A","3giaA","3dadA","2of3A","4y5jA","4ozwA","1uzkA","3klkA","1g3pA","2dyiA","3zbdA","3kyfA","1o7iA","1x8qA","4i1kA","2bhuA","3q1nA","1jovA","2v3iA","4efpA","4le7A","3kstA","3o4pA","2w18A","1l0sA","2j8kA","1hxrA","3maoA","4a6qA","5b1rA","3oajA","4m4dA","6baqA","3e8tA","2v3gA","5c12A","3kluA","3gohA","3dkrA","2imhA","1hq0A","1ds1A","3zh4A","3swgA","5bufA","3hvmA","1h70A","4fs7A","3e4gA","4fcgA","4wp6A","4ecoA","3essA"]

with open("monomers_af_densecpd.json", "w") as outfile:
    for structure in af2_remaining:
        af2_pred[structure] = densecpd_predictions[structure[:4]]
    json.dump(af2_pred, outfile)

all_predictions = []
all_count = 0

with open("densecpd.txt", "w") as datasetmap:
    datasetmap.write("ignore_uncommon False\n")
    datasetmap.write("include_pdbs\n")
    datasetmap.write("##########\n")
    for pdb_key, pdb_seq in densecpd_predictions.items():
        current_chain = benchmark_dict[pdb_key]

        datasetmap.write(f"{pdb_key}{current_chain} {len(pdb_seq)}\n")
        all_predictions += densecpd_encoded_predictions[pdb_key].tolist()
        all_count += len(pdb_seq)
all_predictions = np.array(all_predictions)
assert len(all_predictions) == all_count
# print(all_predictions.shape)
# all_predictions.reshape((20,))
# print(all_predictions)
np.savetxt( "densecpd.csv", all_predictions, delimiter=",")