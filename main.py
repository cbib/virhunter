import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
from virhunter.train import train
from virhunter.predict import predict
from virhunter.prepare_ds import prepare_ds







if __name__ == '__main__':
    fire.Fire()
