#!/bin/bash

wget https://zenodo.org/record/5957275/files/toy_dataset.tar.gz
mkdir -p toy_dataset
tar -xf toy_dataset.tar.gz -C toy_dataset --strip-components=1
mkdir -p toy_dataset_out
rm toy_dataset.tar.gz
