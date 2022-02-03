# VirHunter

**VirHunter** is a deep learning method that uses Convolutional Neural Networks (CNNs) 
and classifies previously assembled contigs to identify potential viral, host and 
bacterial (contamination) sequences in RNAseq sequencing samples. 

## System Requirements
VirHunter installation requires an Unix environment with [python 3.8](http://www.python.org/). 
It was tested under Linux environment.

In order to run VirHunter your installation should include conda. 
If you are installing it for the first time, we suggest you to use 
a lightweight [miniconda](https://docs.conda.io/en/latest/miniconda.html).
         
## Installation 

The full installation process should take less than 15 minutes on a standard computer.

Clone the repository from [github](https://github.com/cbib/virhunter)


`git clone https://github.com/cbib/virhunter.git`

Go to the VirHunter root folder

`cd virhunter/`

You will install dependencies with Conda.
Firstly, you have to create the environment from the `virhunter.yml` file. 
The installation may take around 500 Mb of drive space and take about 15 minutes. 

`conda env create -f virhunter.yml`


Then activate the environment:

`conda activate virhunter`

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

`bash download_weights.sh`

Once the weights are downloaded, if you want for example to use the weights of the model trained on peach 1000bp fragments, you should add in the `config.yaml` file the path to `$DIR/weights/peach/1000` where `$DIR` is the location where you have downloaded the weights.

The command to run predictions is then:

`python main.py predict config.yaml`

## Training your own model

You can train your own model, for example for a specific host species. Training requires to
- prepare the training dataset for the neural network module from fasta file
- prepare the training dataset for Random Forest classifier module
- train both modules 

We provide a toy dataset to illustrate the training process downloadable with the line:

`bash download_toy_dataset.sh`

The `example_config.yaml` file is provided to work with this toy dataset. Running the following 3 commands will prepare the datasets and train the models:

`python main.py prepare_ds toy_config.yaml`

`python main.py sample_test toy_config.yaml`

`python main.py train toy_config.yaml`

## VirHunter on GPU

If you plan to train VirHunter on GPU, please use `virhunter_gpu.yml` for dependencies installation.
Additionally, you have to change `os.environ["CUDA_VISIBLE_DEVICES"] = ""` line in main.py by giving the number of
GPU to use. If your system has only one GPU, likely, it will be `"0"`.


# LICENSE MIT

Copyright (c) 2022 
    Grigorii Sukhorukov (1,2)  (grigorii.sukhorukov@u-bordeaux.fr),
    Macha Nikolski (1,2)    (macha.nikolski@u-bordeaux.fr) 
    
        (1) CBiB - University of Bordeaux,
        146, rue Leo Saignat, 33076 Bordeaux, France

        (2) CNRS, IBGC - University of Bordeaux,
        1, rue Camille Saint-Saens, 33077 Bordeaux, France

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
