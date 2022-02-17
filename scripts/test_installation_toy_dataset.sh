#!/bin/bash
python virhunter/prepare_ds_nn.py configs/test_installation_config.yaml
echo "finished preparation of training dataset for neural network module"
python virhunter/prepare_ds_rf.py configs/test_installation_config.yaml
echo "finished preparation of training dataset for random forest module"
python virhunter/train_nn.py configs/test_installation_config.yaml
echo "finished training neural network module"
python virhunter/train_rf.py configs/test_installation_config.yaml
echo "finished training random forest module"
echo "If there were no errors, VirHunter works properly!"