# pdb-compare

## 0. Getting started

1. Create a conda environment with Python 3.7 and activate it

```shell
conda create --name pdbcompare python=3.7
```

```shell
conda activate pdbcompare
```

2. Install poetry (python package manager)

```shell
conda install poetry 
```

3. Install pymol

```shell
conda install -c schrodinger -c conda-forge pymol-bundle
```

4. Install dependencies

```shell
poetry install
```