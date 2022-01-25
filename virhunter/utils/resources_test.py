import ray

ray.init()
print(ray.available_resources())
print(ray.cluster_resources())


@ray.remote(num_gpus=1)
def test_tf():
    import tensorflow as tf
    return tf.config.list_physical_devices("GPU")


print(ray.get([test_tf.remote() for x in range(10)]))
