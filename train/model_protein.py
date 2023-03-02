
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import numpy as np

MAX_LEN_en = 3000
MAX_LEN_pr = 3000

EMBEDDING_DIM = 3000


from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers


class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])





def get_model():
    humans=Input(shape=(MAX_LEN_en, 1,))
    virus=Input(shape=(MAX_LEN_en, 1,))



    # emb_en=Embedding(NB_WORDS, EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(humans)
    # emb_pr=Embedding(NB_WORDS,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(virus)
    # HUMAN'S PROTEIN
    humans_conv_layer1 = Conv1D(filters=64, kernel_size =16,padding = "valid",activation='relu')(humans)
    humans_max_pool_layer1 = MaxPooling1D(pool_size=3, strides=3)(humans_conv_layer1)
    humans_conv_layer2 = Conv1D(filters=128, kernel_size=32, padding="valid", activation='relu')(humans_max_pool_layer1)
    humans_max_pool_layer2 = MaxPooling1D(pool_size=3, strides=3)(humans_conv_layer2)
    humans_conv_layer3 = Conv1D(filters=256, kernel_size=64, padding="valid", activation='relu')(humans_max_pool_layer2)
    humans_max_pool_layer3 = MaxPooling1D(pool_size=3, strides=3)(humans_conv_layer3)

    # VIRUS'S PROTEIN
    virus_conv_layer1 = Conv1D(filters=64, kernel_size=16, padding="valid", activation='relu')(virus)
    virus_max_pool_layer1 = MaxPooling1D(pool_size=3, strides=3)(virus_conv_layer1)
    virus_conv_layer2 = Conv1D(filters=128, kernel_size=32, padding = "valid",activation='relu')(virus_max_pool_layer1)
    virus_max_pool_layer2 = MaxPooling1D(pool_size=3, strides=3)(virus_conv_layer2)
    virus_conv_layer3 = Conv1D(filters=256, kernel_size=64, padding="valid", activation='relu')(virus_max_pool_layer2)
    virus_max_pool_layer3 = MaxPooling1D(pool_size=3, strides=3)(virus_conv_layer3)


    merge_layer = Concatenate(axis=1)([humans_max_pool_layer3, virus_max_pool_layer3])
    bn = BatchNormalization()(merge_layer)

    dt = Dropout(0.5)(bn)

    l_gru = Bidirectional(LSTM(50, return_sequences=True))(dt)
    l_att = AttLayer(25)(l_gru)
    l_full = Dense(8, activation='tanh')(l_att)

    preds = Dense(1, activation='sigmoid')(l_full)
    adam = Adam(lr=0.0001)

    model = Model([humans, virus], preds)
    model.compile(loss='binary_crossentropy', optimizer=adam)
    return model
