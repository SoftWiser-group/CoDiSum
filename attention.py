from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Embedding, Lambda, Dense
from keras.models import Model
import numpy as np

REMOVE_FACTOR = 1000000


class Attention(Layer):
    def __init__(self, units, return_alphas=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True
        self.return_alphas = return_alphas

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Create a trainable weight variable for this layer.
        self.w_omega = self.add_weight(name='w_omega',
                                       shape=(input_dim, self.units),
                                       initializer='uniform',
                                       trainable=True)
        self.b_omega = self.add_weight(name='b_omega',
                                       shape=(self.units,),
                                       initializer='zeros',
                                       trainable=True)
        self.u_omega = self.add_weight(name='u_omega',
                                       shape=(self.units, 1),
                                       initializer='uniform',
                                       trainable=True)
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        input_dim = K.shape(x)[-1]
        v = K.tanh(K.dot(K.reshape(x, [-1, input_dim]), self.w_omega) + K.expand_dims(self.b_omega, 0))
        vu = K.dot(v, self.u_omega)
        vu = K.reshape(vu, K.shape(x)[:2])
        if mask is not None:
            m = K.cast(mask, K.floatx())
            m = m - 1
            m = m * REMOVE_FACTOR
            vu = vu + m
        alphas = K.softmax(vu)
        output = K.sum(x * K.expand_dims(alphas, -1), 1)
        if self.return_alphas:
            return [output] + [alphas]
        else:
            return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[2])
        if self.return_alphas:
            alphas_shape = [(input_shape[0], input_shape[1])]
            return [output_shape] + alphas_shape
        else:
            return output_shape


class CoAttention(Layer):
    def __init__(self, return_alphas=False, **kwargs):
        super(CoAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_alphas = return_alphas

    def build(self, input_shape):
        input_dim_t = input_shape[0][-1]
        input_dim_f = input_shape[1][-1]
        # Create a trainable weight variable for this layer.
        self.w_beta = self.add_weight(name='w_beta',
                                      shape=(input_dim_t, input_dim_f),
                                      initializer='uniform',
                                      trainable=True)
        super(CoAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        input_dim_t = K.shape(x[0])[-1]
        input_dim_f = K.shape(x[1])[-1]

        # remove padding values
        m_t = K.cast(mask[0], K.floatx())
        t = x[0] * K.expand_dims(m_t, -1)

        # remove padding values
        m_f = K.cast(mask[1], K.floatx())
        f = x[1] * K.expand_dims(m_f, -1)

        # compute affinity matrix
        C = K.dot(K.reshape(t, [-1, input_dim_t]), self.w_beta)
        C = K.reshape(C, [-1, K.shape(x[0])[1], input_dim_f])
        C = K.tanh(K.batch_dot(C, K.permute_dimensions(f, (0, 2, 1))))

        m_t = m_t - 1
        m_t = m_t * REMOVE_FACTOR
        alpha_t = K.max(C, axis=2) + m_t
        alpha_t = K.softmax(alpha_t)

        m_f = m_f - 1
        m_f = m_f * REMOVE_FACTOR
        alpha_f = K.max(C, axis=1) + m_f
        alpha_f = K.softmax(alpha_f)

        t_sum = K.sum(t * K.expand_dims(alpha_t, -1), 1)
        f_sum = K.sum(f * K.expand_dims(alpha_f, -1), 1)

        output = t_sum + f_sum

        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][2])
        if self.return_alphas:
            alphas_shape = [(input_shape[0][0], input_shape[0][1])]
            return [output_shape] + alphas_shape
        else:
            return output_shape


class MyCoAttention(Layer):
    def __init__(self, units, return_alphas=False, **kwargs):
        super(MyCoAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0
        self.return_alphas = return_alphas

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
        super(MyCoAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        en_seq = x[0]
        de_seq = x[1]
        input_de_times = K.shape(de_seq)[-2]

        # compute alphas
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * self.units))
        att_en = K.repeat(att_en, input_de_times)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * input_de_times, self.units))

        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)
        att_de = K.reshape(att_de, shape=(-1, input_de_times, self.units))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)

        co_m = att_en + att_de
        co_m = K.reshape(co_m, (-1, self.units))

        mu = K.dot(K.tanh(co_m), self.nu)

        mu = K.reshape(mu, shape=(-1, input_de_times, self.input_en_times))
        alphas = K.softmax(mu)  # (-1,input_de_times,input_en_times)
        max_mu = K.max(mu, axis=-1) # (-1,input_de_times)
        max_alphas = K.softmax(max_mu)
        h_hat = K.sum(de_seq * K.expand_dims(max_alphas, -1), 1)

        en_seq = K.reshape(en_seq, shape=(-1, self.input_en_times * self.input_dim_en))
        en_seq = K.repeat(en_seq, input_de_times)
        en_seq = K.reshape(en_seq, shape=(-1, input_de_times, self.input_en_times, self.input_dim_en))

        sum_en = K.sum(en_seq * K.expand_dims(alphas, -1), 2)

        output = K.concatenate([de_seq, sum_en, de_seq * sum_en, de_seq * K.expand_dims(h_hat, 1)], -1)
        # output = de_seq + sum_en
        if self.return_alphas:
            alphas = K.reshape(alphas, shape=(-1, input_de_times * self.input_en_times))
            # print(output)
            # print(alphas)
            # print([output] + [alphas])
            return [output] + [alphas]
        else:
            return output

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1] * 4)
        # output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1])
        if self.return_alphas:
            alpha_shape = [(input_shape[1][0], input_shape[1][1] * input_shape[0][1])]
            return [output_shape] + alpha_shape
        else:
            return output_shape


class TimeAttention(Layer):
    def __init__(self, units, return_alphas=False, **kwargs):
        super(TimeAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0
        self.return_alphas = return_alphas

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
        super(TimeAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        en_seq = x[0]
        de_seq = x[1]
        input_de_times = K.shape(de_seq)[-2]

        # compute alphas
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * self.units))
        att_en = K.repeat(att_en, input_de_times)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * input_de_times, self.units))

        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)
        att_de = K.reshape(att_de, shape=(-1, input_de_times, self.units))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)

        co_m = att_en + att_de
        co_m = K.reshape(co_m, (-1, self.units))

        mu = K.dot(K.tanh(co_m), self.nu)

        mu = K.reshape(mu, shape=(-1, input_de_times, self.input_en_times))
        alphas = K.softmax(mu)

        en_seq = K.reshape(en_seq, shape=(-1, self.input_en_times * self.input_dim_en))
        en_seq = K.repeat(en_seq, input_de_times)
        en_seq = K.reshape(en_seq, shape=(-1, input_de_times, self.input_en_times, self.input_dim_en))

        sum_en = K.sum(en_seq * K.expand_dims(alphas, -1), 2)

        output = K.concatenate([de_seq, sum_en], -1)
        # output = de_seq + sum_en
        if self.return_alphas:
            alphas = K.reshape(alphas, shape=(-1, input_de_times * self.input_en_times))
            # print(output)
            # print(alphas)
            # print([output] + [alphas])
            return [output] + [alphas]
        else:
            return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1] + input_shape[1][-1])
        # output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1])
        if self.return_alphas:
            alpha_shape = [(input_shape[1][0], input_shape[1][1], input_shape[0][1])]
            return [output_shape] + alpha_shape
        else:
            return output_shape


class MaskedTimeAttention(Layer):
    def __init__(self, units, return_alphas=False, **kwargs):
        super(MaskedTimeAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0
        self.return_alphas = return_alphas

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
        super(MaskedTimeAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        en_seq = x[0]
        de_seq = x[1]
        input_de_times = K.int_shape(de_seq)[-2]

        if len(x) == 3:
            mask = x[2]
            m_en = K.cast(mask, K.floatx())
            en_seq = en_seq * K.expand_dims(m_en, -1)

        if len(x) == 2 and mask is not None:
            # remove padding values
            m_en = K.cast(mask[0], K.floatx())
            en_seq = en_seq * K.expand_dims(m_en, -1)

        # compute alphas
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * self.units))
        att_en = K.repeat(att_en, input_de_times)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * input_de_times, self.units))

        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)
        att_de = K.reshape(att_de, shape=(-1, input_de_times, self.units))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)

        co_m = att_en + att_de
        co_m = K.reshape(co_m, (-1, self.units))

        mu = K.dot(K.tanh(co_m), self.nu)

        if len(x) == 3 or (len(x) == 2 and mask is not None):
            m_en = K.repeat(m_en, input_de_times)
            m_en = K.reshape(m_en, shape=(-1, 1))
            m_en = m_en - 1
            m_en = m_en * REMOVE_FACTOR
            mu = mu + m_en

        mu = K.reshape(mu, shape=(-1, input_de_times, self.input_en_times))
        alphas = K.softmax(mu)

        en_seq = K.reshape(en_seq, shape=(-1, self.input_en_times * self.input_dim_en))
        en_seq = K.repeat(en_seq, input_de_times)
        en_seq = K.reshape(en_seq, shape=(-1, input_de_times, self.input_en_times, self.input_dim_en))

        sum_en = K.sum(en_seq * K.expand_dims(alphas, -1), 2)

        output = K.concatenate([de_seq, sum_en], -1)

        if self.return_alphas:
            return [output, alphas]
        else:
            return output

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1] + input_shape[1][-1])
        if self.return_alphas:
            alpha_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][1])
            return [output_shape, alpha_shape]
        else:
            return output_shape


class MaskedTimeAttentionWithCoverage(Layer):
    def __init__(self, units, batch_size=1, mask_in=False, coverage_in=False,
                 return_alphas=False, return_covloss=False, **kwargs):
        super(MaskedTimeAttentionWithCoverage, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.batch_size = batch_size
        self.mask_in = mask_in
        self.coverage_in = coverage_in
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0
        self.return_alphas = return_alphas
        self.return_covloss = return_covloss

    def build(self, input_shape):
        self.input_dim_en = input_shape[0][-1]
        self.input_en_times = input_shape[0][-2]
        self.input_dim_de = input_shape[1][-1]
        if self.mask_in and self.coverage_in:
            self.mask_idx = 2
            self.coverage_idx = 3
        elif self.mask_in:
            self.mask_idx = 2
        elif self.coverage_in:
            self.coverage_idx = 2
        # Create a trainable weight variable for this layer.
        # W_h
        self.W_h = self.add_weight(name='W_h', shape=(1, 1, self.input_dim_en, self.units),
                                   initializer='glorot_uniform', trainable=True)
        # W_s
        self.W_s = self.add_weight(name='W_s', shape=(self.input_dim_de, self.units),
                                   initializer='glorot_uniform', trainable=True)
        # w_c
        self.w_c = self.add_weight(name='w_c', shape=(1, 1, 1, self.units),
                                   initializer='glorot_uniform', trainable=True)
        # v
        self.v = self.add_weight(name='v', shape=(self.units, 1),
                                 initializer='glorot_uniform', trainable=True)
        super(MaskedTimeAttentionWithCoverage, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        en_seq = x[0]
        de_seq = x[1]
        input_de_times = K.int_shape(de_seq)[-2]

        use_mask = False
        if self.mask_in and len(x) > self.mask_idx:
            use_mask = True
            mask = x[self.mask_idx]
            m_en = K.cast(mask, K.floatx())
            en_seq = en_seq * K.expand_dims(m_en, -1)

        if not self.mask_in and mask is not None and mask[0] is not None:
            use_mask = True
            # remove padding values
            m_en = K.cast(mask[0], K.floatx())
            en_seq = en_seq * K.expand_dims(m_en, -1)

        # compute alphas
        encoder_states = K.expand_dims(en_seq, axis=2)
        encoder_features = K.conv2d(encoder_states, self.W_h, (1, 1), 'same', 'channels_last')
        # shape (batch_size, input_en_times, 1, units)

        decoder_features = []
        alphas = []
        covlosses = []
        if self.coverage_in and len(x) > self.coverage_idx:
            coverage = x[self.coverage_idx]
        else:
            coverage = K.zeros([self.batch_size, self.input_en_times], dtype=K.floatx())
        # compute everything in one step
        for i in range(input_de_times):
            decoder_state = K.expand_dims(de_seq[:, i, :], 1)
            decoder_feature = K.expand_dims(K.dot(decoder_state, self.W_s), 1)
            # use mask and softmax on coverage
            # if len(x) == 3 or (len(x) == 2 and mask is not None):
            #     cov = K.reshape(self.masked_attention(coverage, m_en), (-1, self.input_en_times, 1, 1))
            # else:
            #     cov = K.reshape(K.softmax(coverage), (-1, self.input_en_times, 1, 1))
            cov = K.reshape(coverage, (-1, self.input_en_times, 1, 1))
            coverage_feature = K.conv2d(cov, self.w_c, (1, 1), 'same', 'channels_last')
            # shape (batch_size, input_en_times, 1, units)

            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t)
            mu = K.dot(K.tanh(encoder_features + decoder_feature + coverage_feature), self.v)
            # shape (batch_size, input_en_times, 1, 1)

            mu = K.reshape(mu, (-1, self.input_en_times))
            if use_mask:
                alpha = self.masked_attention(mu, m_en)
            else:
                alpha = K.softmax(mu)
            a4loss = K.expand_dims(alpha, 1)
            # shape (batch_size, 1, input_en_times)

            a4en = K.expand_dims(alpha)
            # shape (batch_size, input_en_times, 1)

            encoder_feature = K.sum(en_seq * a4en, 1, True)
            # shape (batch_size, 1, input_en_times)

            c4loss = K.reshape(cov, (-1, 1, self.input_en_times))
            covloss = K.concatenate([c4loss, a4loss], 1)
            covloss = K.min(covloss, axis=1, keepdims=True)

            coverage = coverage + alpha
            decoder_features.append(K.concatenate([decoder_state, encoder_feature], axis=-1))
            alphas.append(a4loss)
            covlosses.append(covloss)

        output = K.concatenate(decoder_features, axis=1)
        # shape (batch_size, input_de_times, input_dim_en + input_dim_de)
        alphas = K.concatenate(alphas, axis=1)
        # shape (batch_size, input_de_times, input_en_times)
        covlosses = K.concatenate(covlosses, axis=1)
        # shape (batch_size, input_de_times, input_en_times

        if self.return_alphas:
            output = [output, alphas]
        if self.return_covloss:
            if isinstance(output, list):
                output.append(covlosses)
            else:
                output = [output, covlosses]
        return output

    def masked_attention(self, x, mask):
        m_en = mask - 1
        m_en = m_en * REMOVE_FACTOR
        x = x + m_en
        return K.softmax(x)

    def compute_mask(self, inputs, mask=None):
        # output_mask = mask[1]
        # if self.return_alphas:
        #     output_mask = [mask[1], mask[1]]
        # if self.return_covloss:
        #     if isinstance(output_mask, list):
        #         output_mask.append(mask[1])
        #     else:
        #         output_mask = [mask[1], mask[1]]
        return mask[1]

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1] + input_shape[1][-1])
        alpha_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][1])
        if self.return_alphas:
            output_shape = [output_shape, alpha_shape]
        if self.return_covloss:
            if isinstance(output_shape, list):
                output_shape.append(alpha_shape)
            else:
                output_shape = [output_shape, alpha_shape]
        return output_shape


class Masked(Layer):
    def __init__(self, return_mask=False, **kwargs):
        self.supports_masking = True
        self.return_mask = return_mask
        super(Masked, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Masked, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        output = x
        if mask is not None:
            # remove padding values
            m = K.cast(mask, K.floatx())
            output = x * K.expand_dims(m, -1)
        if self.return_mask:
            return [output, mask]
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_mask:
            return [output_shape, output_shape[:-1]]
        return output_shape


class MaskedGlobalAveragePooling1D(Layer):
    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def call(self, x, mask=None):
        mask = K.cast(mask, K.floatx())
        x = x * K.expand_dims(mask, -1)
        mask_sum = K.expand_dims(K.sum(mask, axis=1), -1)
        ones = K.ones_like(mask_sum, K.floatx())
        mask_sum = K.max(K.concatenate([mask_sum, ones]), keepdims=True)
        return K.sum(x, axis=1) / mask_sum

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class MaskedGlobalMaxPooling1D(GlobalMaxPooling1D):
    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def call(self, x, mask=None):
        mask = K.cast(mask, K.floatx())
        r_mask = (mask - 1) * REMOVE_FACTOR
        x = x * K.expand_dims(mask, -1)
        x = x + K.expand_dims(r_mask, -1)
        return super(MaskedGlobalMaxPooling1D, self).call(x)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class MyOutputLayer(Layer):
    def __init__(self, **kwargs):
        super(MyOutputLayer, self).__init__(**kwargs)
        self.len = 0
        self.dim = 0

    def build(self, input_shape):
        self.len = input_shape[-2]
        self.dim = input_shape[-1]
        # Create a trainable weight variable for this layer.
        self.wp = self.add_weight(name='wp', shape=(self.dim, 1),
                                  initializer='glorot_uniform', trainable=True)
        super(MyOutputLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        output = K.dot(x, self.wp)
        output = K.reshape(output, (-1, self.len))
        return K.softmax(output)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.len


class TestLoop(Layer):
    def __init__(self, return_mask=False, **kwargs):
        super(TestLoop, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_mask = return_mask

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(TestLoop, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        a, b, c = K.int_shape(x)
        output = []
        for i in range(b):
            y = K.expand_dims(x[:, i, :], 1)
            output.append(y)
        return K.concatenate(output, 1)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape


def test():
    MAX_LABELS = 4
    MAX_WORDS = 8
    EMBEDDING_DIM = 6
    ATTENTION_SIZE = 5
    seq_en = Input(shape=(MAX_WORDS, ))
    seq_de = Input(shape=(MAX_LABELS, ))
    seq_en1 = Embedding(50, EMBEDDING_DIM, mask_zero=True)(seq_en)
    seq_de1 = Embedding(50, EMBEDDING_DIM, mask_zero=True)(seq_de)

    output, alpha, cov = MaskedTimeAttentionWithCoverage(units=ATTENTION_SIZE, return_alphas=True,
                                                          return_covloss=True)([seq_en1, seq_de1])
    # output, alpha = MaskedTimeAttention(units=ATTENTION_SIZE, return_alphas=True,)([seq_en1, seq_de1])
    output, mask = Masked(return_mask=True)(output)
    alpha = Masked()(alpha)
    cov = Masked()(cov)
    # output = Dense(1)(output)
    # dot = MyOutputLayer()(seq_en)
    # output = TestLoop()(seq_de1)
    model = Model([seq_en, seq_de], [output, alpha, cov])
    # model = Model(seq_en, dot)
    en_data = np.array([[2, 1, 0, 7, 3, 0, 0, 0]])
    de_data = np.array([[1, 3, 2, 0]])
    res, alpha, cov = model.predict([en_data, de_data])
    # res = model.predict([en_data, de_data])
    print(res, alpha, cov)
    # print(alphas.reshape((-1, MAX_LABELS, MAX_WORDS)))


if __name__ == "__main__":
    test()
