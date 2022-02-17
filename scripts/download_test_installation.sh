#!/bin/bash

wget https://zenodo.org/record/5957275/files/toy_dataset.tar.gz
mkdir -p test_installation
tar -xf toy_dataset.tar.gz -C toy_dataset --strip-components=1
mkdir -p test_installation_out
rm toy_dataset.tar.gz
