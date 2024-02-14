## Aristieta, Parker, Gao, Gittis, and Rubin, 2024
This repo contains all the data and figures for the paper by Aristieta, Parker, Gao, Gittis and Rubin, 2024, submitted.

A Python virtual environment is advised and is assumed in all instructions in this README.

For any questions or issues please contact the owner of this repository.

## Running STReaC on example data set for figures.
Data is stored in `data` directory. Steps to reproduce all figures are as follows:

1. Be sure that the following directories have been created: `figures` and `data`. The directory `data` should already exist since that holds the pre-processed data.
1. Copy `run_classification.py` to STReaC directory (see repository link below): `$ cp scripts/run_classification.py /path/to/streac`
2. Activate virtual environment in STReaC directory and run script: `$ python run_classification.py`. Be sure to replace the variable `data_direc` with the approrpiate string pointing to `/path/to/Aristieta_Parker_Gao_Gittis_Rubin_2024_D1_GPe/data`.
3. Wait patiently, once completed output will be in `./data/`.
4. All figures can then be run from calling `$ python run.py --figures 1 2 3 5 7` or any of the integer figure numbers.
1. Statistics can be run on data by `python run.py --stats` and output will be in `statistics.txt`.
5. You can be reminded about what commands can be called on `run.py` by calling `$ python run.py -h`.

STReaC toolbox repository: [https://github.com/jparker25/streac](https://github.com/jparker25/streac) 

## Python Virtual Environment
Python version 3.12.1 was used to implement this repository.

Please see [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html) for instructions on how to create a virtual environment on your machine.

Then, read in required Python modules via `$ pip install -r requirements.txt`