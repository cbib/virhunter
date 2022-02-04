# VirHunter

**VirHunter** is a deep learning method that uses Convolutional Neural Networks (CNNs) and a Random Forest Classifier to identify viruses in sequening datasets. More precisely, VirHunter classifies previously assembled contigs as viral, host and bacterial (contamination). 

## System Requirements
VirHunter installation requires an Unix environment with [python 3.8](http://www.python.org/). 
It was tested under Linux environment.

In order to run VirHunter your installation should include conda. 
If you are installing it for the first time, we suggest you to use 
a lightweight [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Otherwise, you can use pip for the dependencies' installation.
         
## Installation 

The full installation process should take less than 15 minutes on a standard computer.

Clone the repository from [github](https://github.com/cbib/virhunter)


`git clone https://github.com/cbib/virhunter.git`

Go to the VirHunter root folder

`cd virhunter/`


## Installing dependencies with Conda

Firstly, you have to create the environment from the `envs/environment.yml` file. 
The installation may take around 500 Mb of drive space and take about 15 minutes. 

`conda env create -f envs/environment.yml`

Then activate the environment:

`conda activate virhunter`

## Installing dependencies with pip

If you don't have Conda installed in your system, you can install python dependencies via pip program:

`pip install -r envs/environment.txt`

## Using VirHunter for prediction

VirHunter takes as input a fasta file with assembled contigs and outputs a prediction for each contig to be viral, host (plant) or bacterial.

Befor running VirHunter you have to fill in the config.yaml. You need only to fill the predict part with the following information:
- `ds_path` - path to the input file with contigs in fasta format
- `weights_path` - folder containing weights for a trained model  
- `out_path` - path to the ouput folder to save results
- `fragment_length` - 500 or 1000
- `n_cpus` - number of cpus you want to use

To run VirHunter you can use the already pre-trained models. Provided are fully trained models for 3 host species  (peach, grapevine, sugar beet) and 
for fragment sizes 500 and 1000. Weights for these models are available for download with script `download_weights.sh`.

`bash scripts/download_weights.sh`

Once the weights are downloaded, if you want for example to use the weights of the model trained on peach 1000bp fragments, you should add in the `config.yaml` file the path to `$DIR/weights/peach/1000` where `$DIR` is the location where you have downloaded the weights.

The command to run predictions is then:

`python virhunter/predict.py configs/config.yaml`

## Training your own model

You can train your own model, for example for a specific host species. Training requires to
- prepare the training dataset for the neural network module from fasta file
- prepare the training dataset for Random Forest classifier module
- train both modules 

We provide a toy dataset to illustrate the training process downloadable with the line:

`bash scripts/download_toy_dataset.sh`

The `configs/toy_config.yaml` file is provided to work with this toy dataset. Running the following 4 commands will prepare the datasets and train the models:

`python virhunter/prepare_ds_nn.py configs/toy_config.yaml`

`python virhunter/prepare_ds_nn.py configs/toy_config.yaml`

`python virhunter/train_nn.py configs/toy_config.yaml`

`python virhunter/train_rf.py configs/toy_config.yaml`

`python virhunter/predict.py configs/toy_config.yaml`

## VirHunter on GPU

If you plan to train VirHunter on GPU, please use `virhunter_gpu.yml` for dependencies installation.
Additionally, if you plan to train VirHunter on cluster with multiple GPUs, you may want to add these lines to `train_nn.py`,
where N is the number of GPU you want to use.

```
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "N"
```
