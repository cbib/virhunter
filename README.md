# VirHunter

**VirHunter** is a deep learning method that uses Convolutional Neural Networks (CNNs) and a Random Forest Classifier to identify viruses in sequening datasets. More precisely, VirHunter classifies previously assembled contigs as viral, host and bacterial (contamination). 

## System Requirements
VirHunter installation requires a Unix environment with [python 3.8](http://www.python.org/). 
It was tested on Linux and macOS operating systems. 
For now, VirHunter is still not fully compatible with M1 chip MacBook.

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

## Testing your installation of VirHunter

You can test that VirHunter was successfully installed on the toy dataset we provide. 
IMPORTANT: the toy dataset is intended only to test that VirHunter has been well installed and all the scripts can be executed. 
These modules should not be used for prediction on your owd datasets!

First, you have to download the toy dataset
```shell
bash scripts/download_test_installation.sh
```
Then run the bash script that calls the testing, training and prediction python scripts of VirHunter
```shell
bash scripts/test_installation.sh
```

## Using VirHunter for prediction

To run VirHunter you can use the already pre-trained models or train VirHunter yourself (described in the next section).
Pret-rained model weights are available for the following host plants: 
- grapevine
- tomato
- sugar beet
- peach
- rice
- carrot
- lettuce


To use the model pretrained on one of the plants listed you will need first to download weights using `download_weights.sh` script.
```shell
bash scripts/download_weights.sh 
```
Then you will need to fill `predict_config.yaml file`. If for example, you want to use the weights of the pretrained model for peach, 
you should add in the `configs/predict_config.yaml` paths  `weights/peach/1000` and `weights/peach/500`.


The command to run predictions is then:

```shell
python virhunter/predict.py configs/predict_config.yaml
```

VirHunter models can also be trained on your own data, for example using the same plant as the host of your RNAseq dataset. In this case 2 models have to be trained for fragment sizes 500 and 1000. See below how to train your own models.

Given input contigs, VirHunter produces two comma delimited csv files with prediction:

1. The first file ends with `_contig_fragments.csv`
It is an intermediate result containing predictions of three CNN networks (probabilities of belonging to each of the virus/plant/bacteria class) and of the RF classifier for each fragment of each input contig. 

2. The second file ends with `_predicted_contigs.csv`. 
This file contains final predictions for contigs. In this file, field `id` stores the fasta header of each contig,
`length` describes the length of the contig. Columns `# viral fragments`, `# plant fragments` and `# bacterial fragments` 
store the number of fragments of the contig that received corresponding class prediction by the RF classifier. 
Finally, column `decision` tells you about the final decision for a given contig by the VirHunter.

VirHunter will discard from prediction contigs shorter than 500 bp. VirHunter trained on 500 fragment size will be used for contigs with `750 < length < 1500`. The VirHunter trained on fragment size 1500 will be used for contigs longer than 1500 bp.


## Training your own model

You can train your own model, for example for a specific host species. Before training, you need to collect sequence 
data for training for three reference datasets: _viruses_, _bacteria_ and _host_. 
Examples are provided by running `scripts/download_test_installation.sh` that will download `viruses.fasta`, 
`host.fasta` and `bacteria.fasta` files (real reference datasets should correspond 
e.g. to the whole genome of the host, all bacteria and all viruses from the NCBI).

Training requires execution of the following steps:
- prepare the training dataset for the neural network module from fasta files with `prepare_ds_nn.py`. 
This step splits the reference datasets into fragments of fixed size (specified in the `config.yaml` file, see below)
- prepare the training dataset for Random Forest classifier module with `prepare_ds_rf.py`
- train the neural network module with `train_nn.py`
- train the Random Forest module with `train_rf.py`

The successful training of VirHunter produces weights for the three neural networks from the first module and weights for the 
trained Random Forest classifier. They can be subsequently used for prediction.

To execute the steps of the training you must first fill in the `train_config.yaml`. This file already contains information on all expected inputs.
Once `train_config.yaml` is filled you can launch the scripts consecutively providing them with the config file like this:
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
new branch
