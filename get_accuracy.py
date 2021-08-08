from benchmark import get_cath
from pathlib import Path

PATH_TO_PDB = Path('/scratch/datasets/pdb/')
cath_location = "cath-domain-description-file.txt"
cath_df = get_cath.read_data(cath_location)
# select only monomers
new_df = get_cath.filter_with_user_list(cath_df,
                                        '/home/s1706179/project/sequence-recovery-benchmark/single_chain.txt')
new_df = get_cath.append_sequence(new_df, PATH_TO_PDB)
# choose your model
path_to_file = Path(
    "/home/s1706179/project/sequence-recovery-benchmark/publication_data/default.csv")
with open(path_to_file.with_suffix('.txt')) as datasetmap:
    predictions = get_cath.load_prediction_matrix(new_df,
                                                  path_to_file.with_suffix(
                                                      '.txt'), path_to_file)
accuracy, recall = get_cath.score_each(new_df, predictions, by_fragment=False)
new_df['accuracy'] = accuracy
new_df