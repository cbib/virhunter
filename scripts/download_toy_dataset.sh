#!/bin/bash

wget https://zenodo.org/record/5957275/files/toy_dataset.tar.gz
tar -xvf toy_dataset.tar.gz && mv test toy_dataset
mkdir toy_dataset_out
rm toy_dataset.tar.gz
