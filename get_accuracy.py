from benchmark import get_cath
from pathlib import Path
import urllib



PATH_TO_PDB = Path('/scratch/datasets/pdb/')
model_names = ['gx[c4]-distance-6', 'gx[p20]-distance-12-l5', 'gx[pc]-distance-12-l10_']
for model_name in model_names:
	#model_name = 'skilled-wind-134-timed_hs1000_l2_dp03_weighted'
	path_to_file = Path(f"../../converting_predictions_for_AF2/{model_name}.csv")
	cath_location = "../../sequence-recovery-benchmark/cath-domain-description-file.txt"
	cath_df = get_cath.read_data(cath_location)
	# select only monomers
	new_df = get_cath.filter_with_user_list(cath_df,'single_chain.txt')
	new_df = get_cath.append_sequence(new_df, PATH_TO_PDB)

	# choose your model
	with open(path_to_file.with_suffix('.txt')) as datasetmap:
	    predictions = get_cath.load_prediction_matrix(new_df,
	                                                  path_to_file.with_suffix(
	                                                      '.txt'), path_to_file)
	accuracy, recall = get_cath.score_each(new_df, predictions, by_fragment=False)
	new_df['accuracy'] = accuracy
	new_df['recall'] = recall
	print(new_df)
	new_df.to_csv(f"/scratch/accuracy_for_af2/performance/{model_name}_performance.csv")