#!/bin/bash
python virhunter/prepare_ds_nn.py configs/toy_config.yaml
echo "finished preparation of training dataset for neural network module"
python virhunter/prepare_ds_rf.py configs/toy_config.yaml
echo "finished preparation of training dataset for random forest module"
python virhunter/train_nn.py configs/toy_config.yaml
echo "finished training neural network module"
python virhunter/train_rf.py configs/toy_config.yaml
echo "finished training random forest module"
python virhunter/predict.py configs/toy_config.yaml
echo "finished giving predictions for viral sequences"