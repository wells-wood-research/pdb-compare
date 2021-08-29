from benchmark import get_cath
from pathlib import Path
import numpy as np
from numpy import genfromtxt

path_to_csv = Path("apobec.csv")
my_data = genfromtxt(path_to_csv, delimiter=',')


print(get_cath.most_likely_sequence(my_data))