from tensorflow.keras import layers, models
import tensorflow as tf


class KMaxPooling(layers.Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    Adopted from https://github.com/cjiang2/VDCNN
    """

    def __init__(self, k=None, sorted=False):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.sorted = sorted

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.k, input_shape[2]

    def get_config(self):
        config = super(KMaxPooling, self).get_config().copy()
        config.update({
            'k': self.k,
            'sorted': self.sorted,
        })
        return config

    def call(self, inputs):
        # Swap last two dimensions since top_k will be applied along the last dimension
        shifted_inputs = tf.transpose(inputs, [0, 2, 1])

        # Extract top_k, returns two tensors [values, indices]
        top_k = tf.math.top_k(shifted_inputs, k=self.k, sorted=self.sorted)[0]

        # return flattened output
        return tf.transpose(top_k, [0, 2, 1])


def launch(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output


def model(length, kernel_size=10, filters=512, dense_ns=512):
    forward_input = layers.Input(shape=(length, 4))
    reverse_input = layers.Input(shape=(length, 4))
    hidden_layers = [
        layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        KMaxPooling(k=2),
        layers.Flatten(),
        layers.Dropout(0.1),
    ]
    forward_output = launch(forward_input, hidden_layers)
    reverse_output = launch(reverse_input, hidden_layers)
    output = layers.Concatenate()([forward_output, reverse_output])
    output = layers.Dense(dense_ns, activation='relu')(output)
    output = layers.Dropout(0.1)(output)
    output = layers.Dense(3, activation='softmax')(output)
    model_ = models.Model(inputs=[forward_input, reverse_input], outputs=output)
    model_.compile(optimizer="adam", loss='categorical_crossentropy', metrics='accuracy')
    return model_
