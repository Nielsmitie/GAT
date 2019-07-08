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

## PPI dataset

Couldn't find the right dataset.
Install the dgl library and ran the commands:
```python
from dgl.data import ppi

ppi.PPIDataset('train')

```
```shell
Downloading /home/nuels/.dgl/ppi.zip from https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip...
Extracting file to /home/nuels/.dgl/ppi
Loading G...
<dgl.data.ppi.PPIDataset object at 0x7f3324112c88>
```

Get the data from the said directory.