import numpy as np


def load_embeddings(fn):
    embeddings_index = {}
    f = open(fn)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    return embeddings_index, len(embeddings_index.items()[0][1])


def init_embeddings_from(embeddings_source, word_index):
    embedding_dim = len(embeddings_source.items()[0][1])
    # embedding_matrix = np.zeros((len(word_index), embedding_dim))
    embedding_matrix = np.random.uniform(-0.5, 0.5, (len(word_index), embedding_dim))
    count = 0
    not_found_words = []
    for word, i in word_index.items():
        embedding_vector = embeddings_source.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count += 1
        else:
            not_found_words.append(word)
    
    print("Found %s words in embeddings source." % count)
    
    return embedding_matrix


def encode_one_hot(int_data, vocab_size):
    one_hots = np.zeros([len(int_data), vocab_size])
    for i, value in enumerate(int_data):
        one_hots[i, int(value)] = 1
        if value == 0:
            break

    return one_hots
