#!/bin/bash

wget https://www.dropbox.com/s/ndrxwgoiw8o9pon/toy_dataset.tar.gz
mkdir -p test_installation
tar -xf toy_dataset.tar.gz -C test_installation --strip-components=1
mkdir -p test_installation_out
rm toy_dataset.tar.gz
