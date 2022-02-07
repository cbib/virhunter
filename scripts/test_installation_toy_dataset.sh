#!/bin/bash

python virhunter/prepare_ds_nn.py configs/toy_config.yaml
python virhunter/prepare_ds_rf.py configs/toy_config.yaml
python virhunter/train_nn.py configs/toy_config.yaml
python virhunter/train_rf.py configs/toy_config.yaml
python virhunter/predict.py configs/toy_config.yaml