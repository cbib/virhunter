#!/bin/bash

wget https://www.dropbox.com/s/efkwusdg5pahdb1/weights_vh.tar.gz
mkdir -p weights
tar -xf weights_vh.tar.gz -C weights --strip-components=1
rm weights_vh.tar.gz
