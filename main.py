from benchmark import get_cath
from pathlib import Path
import urllib


def fetch_pdb(
    pdb_code: str,
    output_folder:Path,
    pdb_request_url: str = "https://files.rcsb.org/download/" ,
    is_pdb:bool=True,
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
    urllib.request.urlretrieve(pdb_request_url + pdb_code_with_extension,filename=output_folder / pdb_code_with_extension)

PATH_TO_PDB = Path('/scratch/datasets/pdb/')
path_to_file = Path("/default.csv")
cath_location = "cath-domain-description-file.txt"
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
new_df.to_csv(path_to_file.with_suffix('_accuracy.csv'))