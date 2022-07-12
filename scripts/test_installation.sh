#!/bin/bash
python virhunter/prepare_ds.py configs/test_installation_config.yaml
echo "finished preparation of training dataset for neural network and random forest modules"
python virhunter/train.py configs/test_installation_config.yaml
echo "finished training neural network and random forest modules"
python virhunter/predict.py configs/test_installation_config.yaml
echo "finished prediction of the test file"
echo "If there were no errors, VirHunter works properly!"