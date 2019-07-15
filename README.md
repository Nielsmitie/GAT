# GAT DL Fork
Forked from https://github.com/PetarV-/GAT

## Setup

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

# for GPU execution install
pipenv install tensorflow-gpu
```

## Data

Data downloaded from this repo:
https://github.com/danielegrattarola/keras-gats

The small datasets are already included in the repository.

### PPI dataset
Go to
http://snap.stanford.edu/graphsage/
and select Protein-Protein Interactions -> preprocessed. Download and unpack it.
Move it into the p2p_dataset directory.

## Run

To run the experiments from the paper:
```shell
# run the provided example
pipenv run python -m execute_cora
pipenv run python -m execute_citeseer
pipenv run python -m execute_pupmed
pipenv run python -m execute_ppi
```

Note: If the code should be run on GPU it might be necessary to uncomment the first few
lines in the execute_.run_gat function.

If you don't use pipenv just use pure python instead.

Modifications to the hyperparameters can be made in the respective execute_...
script.

To evaluate the run call:
```shell
pipenv run python -m evaluate_
```

## Repository structure

### data

Contains the tracked data files for the cora, citeseer and pumed experiments.

### p2p_dataset

An untracked folder where the Protein-Protein-Interaction (ppi) dataset is stored.

### pretrained

Contains the checkpoint files of the last trained models for all experiments.

Also has all the configurations saved in .csv files with the respective results from
the experiment.

Lastly, the directory has {}out.txt files where all output, mainly the training 
and validation loss and accuracy during a run is tracked.

### utils

#### process

Has the necessary functions to load and preprocess the datasets cora, citeseer and pubmed.

Also it contains the function adj_to_bias that is used in execute_.py

#### process_ppi

Contains the necessary functions to load and preprocess the ppi dataset.

#### layers

Contains the attention layers and sparse attention layers.


### models

#### base_gattn

Has the functions for training, evaluation and loss calculation.

#### gat

Inherits from BaseGAttn and add the inference functions.
There the layers are defined. It uses the function layers.attn_head.

#### sp_gat

Also inherits from BaseGAttn. But here the inference
is done with layers.sp_attn_head

### root

The execute functions defines the basic setup for all experiments. It calls the
load method. Then preprocesses the data and runs the test, validation and test
step for a number of epochs. All results will be printed and directly output into
a text file.

The final scores are documented in the {}log.csv files.

The execute_... files just define the parameters and then call the described function
in execute_.py.

execute_ppi reimplements the function from execute_, because the training, validation and
test step use different parts of the data.
