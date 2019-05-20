from models1 import CopyNetPlus
from data4CopynetV3 import Data4CopynetV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import encode_one_hot
import numpy as np
import json


def generator(t1, t2, t3, t4, start, end, vocab_size, batch_size, shuffle):
    tt1, tt2, tt3, tt4 = t1[start: end], t2[start: end], t3[start: end], t4[start: end]
    i = 0
    if shuffle:
        sf = np.arange(end - start)
        np.random.shuffle(sf)
        tt1, tt2, tt3, tt4 = tt1[sf], tt2[sf], tt3[sf], tt4[sf]
    while True:
        if i + batch_size > end - start:
            i = 0
            if shuffle:
                sf = np.arange(end - start)
                np.random.shuffle(sf)
                tt1, tt2, tt3, tt4 = tt1[sf], tt2[sf], tt3[sf], tt4[sf]
        x = [tt1[i:i + batch_size], tt2[i:i + batch_size], tt3[i:i + batch_size], tt4[i:i + batch_size, :20]]
        y = np.array([encode_one_hot(x, vocab_size) for x in tt4[i:i + batch_size, 1:]])
        i += batch_size
        yield x, y


MODEL_PATH = 'models/CopyNetPlusWED150HS16.h5'          # TODO
TR_S = 0                        # train_start_index
TR_E = 75000                    # train_end_index
VA_S = 75000                    # valid_start_index
VA_E = 83000                    # valid_end_index
TE_S = 83000                    # test_start_index
TE_E = 90661                    # test_end_index
TR_BS = 100                     # train batch size
EP = 50                         # trian epoch
TE_BS = 1                       # test batch size
E_L = 200                       # encoder len
A_N = 5                         # attribute number
D_L = 20                        # decoder len
EM_V = 24634                    # embedding vocabulary num
DE_V = 10130                    # decoder vocabulary num
SEED = 1                        # random seed
MED = 50                        # mark embedding dim
WED = 150                       # TODO: word embedding dim
HS = 16                         # TODO: hidden size
ATN = 64                        # attention num
TR_DR = 0.1                     # drop rate for train
TE_DR = 0.                      # drop rate for test
PC = 2                          # patience
BM_S = 2                        # beam size
VER = 12                        # data version


def train():
    dataset = Data4CopynetV3()
    dataset.load_data(VER)
    t1, t2, t3, t4, genmask, copymask = dataset.gen_tensor2(TR_S, VA_E, DE_V)
    del dataset
    model, _, _ = CopyNetPlus(E_L, D_L, A_N, EM_V, DE_V, MED, WED, HS, ATN, TR_DR, genmask, copymask)
    es = EarlyStopping(monitor='val_loss', patience=PC)
    cp = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True)
    model.fit_generator(generator(t1, t2, t3, t4, TR_S, TR_E, DE_V, TR_BS, True), (TR_E - TR_S) / TR_BS,
                        EP, callbacks=[es, cp],
                        validation_data=generator(t1, t2, t3, t4, VA_S, VA_E, DE_V, TR_BS, False),
                        validation_steps=(VA_E - VA_S) / TR_BS)


def predict():
    dataset = Data4CopynetV3()
    dataset.load_data(VER)
    t1, t2, t3, t4, genmask, copymask = dataset.gen_tensor2(TE_S, TE_E, DE_V)
    model, encoder, decoder = CopyNetPlus(E_L, D_L, A_N, EM_V, DE_V, MED, WED, HS, ATN, TE_DR, genmask, copymask)
    diff, msg, va = dataset.get_data(TE_S, TE_E, re_difftoken=True, re_msg=True, re_variable=True)
    model.load_weights(MODEL_PATH)
    wi, _ = dataset.get_word2index()
    iw = {k: v for v, k in wi.items()}
    # f1 = open('reference1.msg', 'w')
    f2 = open('CopyNetPlusWED150HS16_1.gen', 'w')                 
    lemmatization = json.load(open('lemmatization.json'))
    for i, j, k, l, m, n in zip(t1, t2, diff, msg, va, t3):
        variable = {k: v for v, k in m.items()}
        masked_rnn_h3, mask, m_embed_en, state1, state2, state3 = encoder.predict([np.array([i]), np.array([j]),
                                                                                   np.array([n])])
        cur_token = np.zeros((1, 1))
        cur_token[0, 0] = 1
        results = []
        predict_next_token(decoder, cur_token, np.array([j]), masked_rnn_h3, mask, m_embed_en, state1, state2, state3,
                           0, 0.0, results, [], [], [], D_L, BM_S)
        results = sorted(results, key=lambda x: x[0], reverse=True)
        # results = sorted(results, key=lambda x: len(x[1]), reverse=True)    # TODO
        de_seq = results[0][1]
        gen_word = list2sentence(de_seq, iw, variable)
        # n = [lemmatization[i.lower()] if i.lower() in lemmatization else i.lower() for i in l]
        # f1.write(' '.join(n) + '\n')
        f2.write(' '.join(gen_word) + '\n')
        # print(' '.join(gen_word))


def predict_next_token(decoder, cur_token, word_in, encoder_in, mask, m_embed, state1, state2, state3, cur_depth,
                       joint_prs, res, tags, alphas, pgens, max_len, beam_size):
    cur_depth += 1
    prs, pgen, alpha, st1, st2, st3 = decoder.predict([cur_token, word_in, encoder_in, mask, m_embed,
                                                       state1, state2, state3])
    prs = prs[0, 0, :]
    alpha = alpha[0, 0, :]
    pgen = pgen[0, 0, 0]
    # i for index, v for softmax value in i
    prs = [(i, v) for i, v in enumerate(prs)]
    prs = sorted(prs, key=lambda x: x[1], reverse=True)
    # if cur_depth == 2:                      # TODO
    #     beam_size = 2
    if cur_depth == 4:
        beam_size = 1
    for p in prs[:beam_size]:
        # end of sentence
        if p[0] == 0:
            res.append(((joint_prs + np.log(p[1]))/cur_depth, tags, alphas, pgens))
            break
        # max decode len
        if cur_depth == max_len - 1:
            res.append(((joint_prs + np.log(p[1]))/cur_depth, tags[:] + [p[0]], alphas + [alpha], pgens + [pgen]))
            break
        # generation continue
        token = np.zeros((1, 1))
        token[0, 0] = p[0]
        predict_next_token(decoder, token, word_in, encoder_in, mask, m_embed, st1, st2, st3, cur_depth,
                           joint_prs + np.log(p[1]), res, tags[:] + [p[0]], alphas + [alpha], pgens + [pgen],
                           max_len, beam_size)


def list2sentence(seq, index2word, variable):
    words = list()
    for i in seq:
        # print(i, j)
        if i == 0:
            break
        else:
            word = index2word[i]
            word = variable[word] if word in variable else word
            words.append(word.lower())
    return words


if __name__ == '__main__':
    train()        
    predict()
