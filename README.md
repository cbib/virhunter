# VirHunter

**VirHunter** is a deep learning method that uses Convolutional Neural Networks (CNNs) and a Random Forest Classifier to identify viruses in sequening datasets. More precisely, VirHunter classifies previously assembled contigs as viral, host and bacterial (contamination). 

## System Requirements
VirHunter installation requires a Unix environment with [python 3.8](http://www.python.org/). 
It was tested on Linux and macOS operating systems. 
For now, VirHunter is still not fully compatible with M1 chip Macbooks.

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
bash scripts/download_test_installation.sh
```
Then launch the script for testing training and prediction python scripts of VirHunter
```shell
bash scripts/test_installation.sh
```
## Using VirHunter for prediction

VirHunter takes as input a fasta file with contigs and outputs a prediction for each contig to be viral, host (plant) or bacterial.

For given contigs VirHunter produces a tab delimited csv file with prediction. `id` stores the fasta header of a contig,
`length` describes the length of the contig. Columns `virus`, `plant` and `bacteria` store the number of fragments of the contig
that received corresponding prediction by the RF classifier. Finally, column `decision` tell you about the final decision for a given contig.
You should refer to it, when filtering viral contigs. 

To do predictions VirHunter needs to be fully trained for fragment sizes 500 and 1000. VirHunter will discard from prediction
contigs shorter than 500 bp. VirHunter trained on 500 fragment size will be used for contigs with `750 < length < 1500`. The VirHunter
trained on fragment size 1500 will be used for contigs longer than 1500 bp.

Before running VirHunter you have to fill in the `predict_config.yaml` file.

To run VirHunter you can use the already pre-trained models. Provided are fully trained models for 3 host species  (peach, grapevine, sugar beet) and 
for fragment sizes 500 and 1000. Weights for these models can be downloaded by running the `download_weights.sh` script:
```shell
bash scripts/download_weights.sh
```
Once the weights are downloaded, if you want for example to use the weights of the model trained on peach, 
you should add in the `configs/predict_config.yaml` paths  `weights/peach/1000` and `weights/peach/500`.

The command to run predictions is then:

```shell
python virhunter/predict.py configs/predict_config.yaml
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

To execute these steps you must first fill in the `config.yaml`. This file already contains information on all expected inputs .
Once `config.yaml` is filled you can launch the scripts consecutively providing them with the config file like this:
```shell
python virhunter/prepare_ds_nn.py configs/train_config.yaml
```

## VirHunter on GPU

If you plan to train VirHunter on GPU, please use `environment_gpu.yml` or `requirements_gpu.txt` for dependencies installation.
Those recipes were tested only on the Linux cluster with multiple GPUs.
If you plan to train VirHunter on cluster with multiple GPUs, you will need to uncomment line with
`CUDA_VISIBLE_DEVICES` variable and replace `""` with `"N"` in header of `train_nn.py`, where N is the number of GPU you want to use.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "N"
```
