# GAT DL Fork

# Setup

```shell
git clone git@github.com:Nielsmitie/GAT.g
# or 
git clone https://github.com/Nielsmitie/GAT.git
```
If on windows:
Install python 3 and pip. Added both of them to your path. Pipenv should get installed as well.
Otherwise open powershell as admin and type: pip install pipenv

Move into the repository:
```shell
# install all required dependencies
pipenv install
# run the provided example
pipenv run python -m execute_cora
```

# Data

Data downloaded from this repo:
https://github.com/danielegrattarola/keras-gats

The small datasets are already included in the repository.

## PPI dataset
Go to
http://snap.stanford.edu/graphsage/
and select Protein-Protein Interactions -> preprocessed. Download and unpack it.
Move it into the p2p_dataset directory.
