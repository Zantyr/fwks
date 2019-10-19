import keras
import numpy as np

import keras.backend as K

class CZT(keras.layers.Layer):
    def build(self, input_shape):
        self.z = self.add_weight(shape=[2], initializer=lambda shape: K.variable([np.real(shift), np.imag(shift)]),
                                 name='kernel_z', constraint=max_norm(1.))
        self.w = self.add_weight(shape=[2], initializer=lambda shape: K.variable([1, 0]),
                                 name='kernel_w', constraint=max_norm(1.))
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 257
        return tuple(output_shape)
    
    def call(self, inputs):
        ints = K.reshape(K.arange(257, dtype='complex64'), [257, 1])
        k = K.reshape(K.arange(512, dtype='complex64'), [1, 512])
        z = K.cast(self.z[0], dtype='complex64_ref') + 1j * K.cast(self.z[1], dtype='complex64_ref')
        w = K.cast(self.w[0], dtype='complex64_ref') + 1j * K.cast(self.w[1], dtype='complex64_ref')
        weights = K.dot(z * K.ones([257, 1], dtype='complex64'), K.reshape(w, [1, -1]) ** (-k)) ** (-ints)
        print(z.shape, w.shape, weights.shape)
        czt = K.dot(K.cast(inputs, dtype='complex64_ref'), K.transpose(weights))
        return K.abs(czt)
