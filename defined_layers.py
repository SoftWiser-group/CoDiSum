from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Embedding, Lambda, GRU, Dense
from keras.layers import Conv2D, AveragePooling2D, Bidirectional
from keras.models import Model
from attention import Masked
import numpy as np


class GetPiece(Layer):
    def __init__(self, num, **kwargs):
        self.supports_masking = True
        self.num = num
        super(GetPiece, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(GetPiece, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        output = x[:, self.num, :, :]
        return output

    def compute_mask(self, inputs, mask=None):
        output_mask = mask[:, self.num, :]
        return output_mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3]


class AttentionCopy(Layer):
    def __init__(self, size, **kwargs):
        self.size = size
        super(AttentionCopy, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionCopy, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs is a list, x[0] is encoder word input, x[1] is attention alpha
        in_one_hot = K.one_hot(K.cast(inputs[0], 'int32'), self.size)
        output = K.batch_dot(inputs[1], in_one_hot)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        y = list(input_shape[1])
        y[-1] = self.size
        return tuple(y)


class ComputeAttention(Layer):
    # compute the attention between encoder hidden state and decoder hidden state
    def __init__(self, units, **kwargs):
        super(ComputeAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0

    def build(self, input_shape):
        self.input_dim_en = input_shape[0][-1]
        self.input_en_times = input_shape[0][-2]
        self.input_dim_de = input_shape[1][-1]
        # Create a trainable weight variable for this layer.
        # w1
        self.w_en = self.add_weight(name='w_en', shape=(self.input_dim_en, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # w2
        self.w_de = self.add_weight(name='w_de', shape=(self.input_dim_de, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # nu
        self.nu = self.add_weight(name='nu', shape=(self.units, 1),
                                  initializer='glorot_uniform', trainable=True)
        super(ComputeAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        # x[0] is encoder hidden state, x[1] is decoder hidden state.
        en_seq = x[0]
        de_seq = x[1]
        input_de_times = K.int_shape(de_seq)[-2]

        use_mask = False
        if len(x) == 3:
            mask = x[2]
            m_en = K.cast(mask, K.floatx())
            use_mask = True
        if len(x) == 2 and mask is not None:
            m_en = K.cast(mask[0], K.floatx())
            use_mask = True

        # compute alphas
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * self.units))
        att_en = K.repeat(att_en, input_de_times)
        att_en = K.reshape(att_en, shape=(-1, input_de_times * self.input_en_times, self.units))

        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)
        att_de = K.reshape(att_de, shape=(-1, input_de_times, self.units))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)

        co_m = att_en + att_de
        co_m = K.reshape(co_m, (-1, self.units))

        mu = K.dot(K.tanh(co_m), self.nu)

        if use_mask:
            m_en = K.repeat(m_en, input_de_times)
            m_en = K.reshape(m_en, shape=(-1, 1))
            m_en = m_en - 1
            m_en = m_en * 1000000
            mu = mu + m_en

        mu = K.reshape(mu, shape=(-1, input_de_times, self.input_en_times))
        return K.softmax(mu)

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][1], input_shape[0][1]


class HierAttentionCopy(Layer):
    def __init__(self, size, **kwargs):
        self.size = size
        super(HierAttentionCopy, self).__init__(**kwargs)

    def build(self, input_shape):
        self.blocks = input_shape[-1][-2]
        self.block_len = input_shape[-1][-1]
        self.de_len = input_shape[1][1]
        super(HierAttentionCopy, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # parameter weight, alphas, encoder_word
        weight = K.reshape(inputs[0], (-1, 1, self.blocks, 1))
        xs = []
        for i in inputs[1:-1]:
            xs.append(K.expand_dims(i, 2))
        x = K.concatenate(xs, 2)
        x = x * weight
        in_word = K.reshape(inputs[-1], (-1, self.blocks * self.block_len))
        in_one_hot = K.one_hot(K.cast(in_word, 'int32'), self.size)
        output = K.batch_dot(K.reshape(x, (-1, self.de_len, self.blocks * self.block_len)), in_one_hot)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        y = list(input_shape[0])
        y[-1] = self.size
        return tuple(y)


class CombineGenCopy(Layer):
    def __init__(self, **kwargs):
        self.supports_masking=True
        super(CombineGenCopy, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CombineGenCopy, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs[0] is p_gen, inputs[1] is gen_prob, inputs[2] is copy_prob
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class MaskedSoftmax(Layer):
    def __init__(self, mask, **kwargs):
        self.mask = mask
        self.supports_masking = True
        super(MaskedSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskedSoftmax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mask = K.constant(self.mask, K.floatx())
        m_en = mask - 1
        m_en = m_en * 1000000
        m_en = K.expand_dims(m_en, 0)
        inputs = inputs + m_en
        inputs = K.softmax(inputs)
        return inputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape


class MaskedCopyProb(Layer):
    def __init__(self, mask, **kwargs):
        self.mask = mask
        self.supports_masking = True
        super(MaskedCopyProb, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskedCopyProb, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mask = K.constant(self.mask, K.floatx())
        m_en = K.expand_dims(mask, 0)
        inputs = inputs * m_en
        return inputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape


class ComputeAlpha(Layer):
    def __init__(self, **kwargs):
        super(ComputeAlpha, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.W = self.add_weight(name='W',
                                 shape=(input_dim, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(ComputeAlpha, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mask = inputs[-1]
        outs = list()
        for i in inputs[:-1]:
            out = K.dot(i, self.W)
            outs.append(out)
        outs = K.concatenate(outs, -1)
        m_en = mask - 1
        m_en = m_en * 1000000
        outs = outs + m_en
        return K.softmax(outs)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], len(input_shape) - 1


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        super(WeightedSum, self).build(input_shape)

    def call(self, inputs, **kwargs):
        weight = K.expand_dims(inputs[0])
        xs = []
        for i in inputs[1:]:
            xs.append(K.expand_dims(i, 1))
        x = K.concatenate(xs, 1)
        out = K.sum(x * weight, 1)
        return out

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class MaskedConv2D(Conv2D):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.supports_masking = True
        super(MaskedConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def call(self, x, mask=None):
        return super(MaskedConv2D, self).call(x)

    def compute_mask(self, inputs, mask=None):
        return None


class MaskedAveragePooling2D(AveragePooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        self.supports_masking = True
        super(MaskedAveragePooling2D, self).__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )

    def call(self, x, mask=None):
        return super(MaskedAveragePooling2D, self).call(x)

    def compute_mask(self, inputs, mask=None):
        return None


def validate():
    # seq = Input(shape=(3, 4))
    # seq1 = Embedding(20, 5, mask_zero=True)(seq)
    # seq1 = Dense(5)(seq)
    # x = MaskedSoftmax([1,1,0,0,1])(seq1)
    # x = GRU(6, return_sequences=True)(x)
    # x, m = Masked(return_mask=True)(x)
    # weight = np.reshape(np.arange(25), [1, 5, 5])
    # print(weight)
    # seq = Input(shape=(4,))
    # e = Embedding(5, 5, weights=weight, trainable=False)
    # x = e(seq)
    # model = Model(seq, x)
    # en = np.array([1, 2, 3, 4]).reshape((1, 4))
    # a = model.predict(en)
    # x = Input(shape=(4, ))
    # y = Input(shape=(4, ))
    # z = Input(shape=(4, ))
    # mask = Input(shape=(3,))
    # out = ComputeAlpha()([x, y, z, mask])
    # model = Model([x,y,z,mask],out)
    # x = np.array([[0,0,0,0],[0,0,0,0]])
    # y = x
    # z = x
    # mask = np.array([[1,1,1],[1,1,0]])
    # a = model.predict([x,y,z,mask])
    # print(a)
    # 20190129 1845
    input = Input(shape=(10, 5, 20), dtype=K.floatx())
    re_in = Lambda(lambda x: K.reshape(x, (-1, 5, 20)))(input)
    rnn_h = Bidirectional(GRU(30))(re_in)
    rnn_h = Lambda(lambda x: K.reshape(x, (-1, 10, 60)))(rnn_h)
    model = Model(input, rnn_h)
    x = np.zeros(shape=(3, 10, 5, 20), dtype='float32')
    x[0, 0, 0, :] = 1.
    # print(x)
    a = model.predict(x)
    print(a)


if __name__ == "__main__":
    validate()
