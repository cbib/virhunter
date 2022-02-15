# VirHunter

**VirHunter** is a deep learning method that uses Convolutional Neural Networks (CNNs) and a Random Forest Classifier to identify viruses in sequening datasets. More precisely, VirHunter classifies previously assembled contigs as viral, host and bacterial (contamination). 

## System Requirements
VirHunter installation requires a Unix environment with [python 3.8](http://www.python.org/). 
It was tested on Linux and macOS operating systems.

In order to run VirHunter your installation should include conda. 
If you are installing it for the first time, we suggest you to use 
a lightweight [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Otherwise, you can use pip for the dependencies' installation.
         
## Installation 

The full installation process should take less than 15 minutes on a standard computer.

Then clone the repository from [github](https://github.com/cbib/virhunter)

```shell
git clone https://github.com/cbib/virhunter.git
```

Go to the VirHunter root folder

```shell
cd virhunter/
```


## Installing dependencies with Conda

First, you have to create the environment from the `envs/environment.yml` file. 
The installation may take around 500 Mb of drive space. 

```shell
conda env create -f envs/environment.yml
```

Second, activate the environment:

```shell
conda activate virhunter
```

## Installing dependencies with pip

If you don't have Conda installed in your system, you can install python dependencies via pip program:

```shell
pip install -r envs/requirements.txt
```

Then if you have macOS you will need to install `wget` library to run some scripts (Conda installation already has it). You can do this with `brew` package manager.

```shell
brew install wget
```

## Testing installation of the VirHunter

You can test that VirHunter was successfully installed on the toy dataset we provide. 
IMPORTANT: the toy dataset is intended only to test the correct work of VirHunter. 
The trained modules should not be used for prediction on your datasets!

First, you have to download the toy dataset
```shell
bash scripts/download_toy_dataset.sh
```
Then launch the script for testing training and prediction python scripts of VirHunter
```shell
bash scripts/test_installation_toy_dataset.sh
```
## Using VirHunter for prediction

VirHunter takes as input a fasta file with contigs and outputs a prediction for each contig to be viral, host (plant) or bacterial.

Before running VirHunter you have to fill in the `config.yaml` file. For the prediction you need to fill in only the `predict` part (see below).

To run VirHunter you can use the already pre-trained models. Provided are fully trained models for 3 host species  (peach, grapevine, sugar beet) and 
for fragment sizes 500 and 1000. Weights for these models can be downloaded by running the `download_weights.sh` script:
```shell
bash scripts/download_weights.sh
```
Once the weights are downloaded, if you want for example to use the weights of the model trained on peach 1000bp fragments, 
you should add in the `configs/config.yaml` file the path to `weights/peach/1000`.

The command to run predictions is then:

```shell
python virhunter/predict.py configs/config.yaml
```

## Training your own model

You can train your own model, for example for a specific host species. Before training, you need to collect sequence 
data for training for three reference datasets: _viruses_, _bacteria_ and _host_. 
Examples are provided by running `scripts/download_toy_dataset.sh` that will download `viruses.fasta`, 
`host.fasta` and `bacteria.fasta` files (real reference datasets should correspond 
e.g. to the whole genome of the host, all bacteria and all viruses from the NCBI).


Training requires execution of the following steps:
- prepare the training dataset for the neural network module from fasta files with `prepare_ds_nn.py`. 
This step splits the reference datasets into fragments of fixed size (specified in the `config.yaml` file, see below)
- prepare the training dataset for Random Forest classifier module with `prepare_ds_rf.py`
- train the neural network module with `train_nn.py`
- train the Random Forest module with `train_rf.py`

To execute these steps you must first fill in the `config.yaml` and then launch the scripts consecutively providing them 
with the config file like this:
```shell
python virhunter/prepare_ds_nn.py configs/config.yaml
```

## VirHunter config.yaml description

`predict`:
- `ds_path`: path to the input file with contigs in fasta format
- `nn_weights_path`: folder containing weights for a neural network module
- `rf_weights_path`: folder containing weights for a random forest module
- `out_path`: where to save predictions
- `fragment_length`: 500 or 1000
- `n_cpus`: number of cpus you want to use

`prepare_ds_nn`:
- `path_virus`: path to fasta file with viral sequences
- `path_plant`: path to fasta file with plant sequences 
- `path_bact`: path to fasta file with bacterial sequences 
- `out_path`: where to save training dataset for neural networks (in hdf5 format)
- `fragment_length`: 500 or 1000 
- `n_cpus`: number of cpus you want to use
- `random_seed`: random seed for reshuffling dataset

`prepare_ds_rf`:
- `path_virus`: path to fasta file with viral sequences
- `path_plant`: path to fasta file with plant sequences 
- `path_bact`: path to fasta file with bacterial sequences 
- `out_path`: where to save training dataset for RF
- `fragment_length`: 500 or 1000 
- `n_cpus`: number of cpus you want to use
- `random_seed`: random seed for reshuffling dataset

`train_nn`:
- `ds_path`: path to the dataset prepared with `prepare_ds_nn` 
- `out_path`: where to save weights of the trained neural networks
- `epochs`: Number of epochs for each neural network to train
- `fragment_length`: 500 or 1000
- `random_seed`: random seed for reshuffling of the training dataset

`train_rf`:
- `nn_weights_path`: path to weights of trained neural networks prepared with `train_nn`
- `ds_rf_path`: path to training dataset for RF prepared with `prepare_ds_rf`
- `out_path`: where to save weights of the trained RF 
- `fragment_length`: 500 or 1000 
- `n_cpus`: number of cpus you want to use 
- `random_seed`: random seed for reshuffling of the training dataset

## VirHunter on GPU

If you plan to train VirHunter on GPU, please use `environment_gpu.yml` or `requirements_gpu.ytxt` for dependencies installation.
Additionally, if you plan to train VirHunter on cluster with multiple GPUs, you will need to uncomment line with
`CUDA_VISIBLE_DEVICES` variable and replace `""` with `"N"` in header of `train_nn.py`, where N is the number of GPU you want to use.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "N"
```
