import numpy as np
import string
from collections import Counter
from sklearn.feature_extraction import stop_words
from sklearn.decomposition import TruncatedSVD, NMF
import nltk
from nltk import stem
import scipy

translation_table = str.maketrans('', '', string.punctuation)

def remove_stop_words(text_token):
    return [token for token in text_token if token not in stop_words.ENGLISH_STOP_WORDS]

def lem_words(text_token):
    lem = stem.WordNetLemmatizer()
    return [lem.lemmatize(token, pos='v') for token in text_token]

def distinct_corpus_words(tokens_corpus):
    tokens_corpus_flatten = [token for tokens in tokens_corpus for token in tokens]
    corpus_counter = Counter(tokens_corpus_flatten).most_common()
    return corpus_counter, len(corpus_counter)

def normalize_single_text(text, translation_table=translation_table):
    text = str(text).lower()
    text = text.translate(translation_table)

    text_tokens = nltk.word_tokenize(text)
    text_tokens = remove_stop_words(text_tokens)
    text_tokens = lem_words(text_tokens)

    return text_tokens

def normalize_text(text_corpus, translation_table=translation_table):
    return [normalize_single_text(text, translation_table=translation_table) for text in text_corpus]

def build_dictionary(corpus_counter):
    word2ind = {corpus_counter[i][0]: i for i in range(len(corpus_counter))}
    ind2word = [word[0] for word in corpus_counter]
    return word2ind, ind2word

def text_token2ind(text_token, word2ind):
    return [word2ind[token] for token in text_token if token in word2ind]

def build_bow(ind_corpus, n_tokens, max_features=None):
    """Max features takes n features with the lowest index - assumes that lower index -> higher number of occurrences"""

    if max_features:
        ind_corpus = [[ind for ind in text_ind if ind < max_features] for text_ind in ind_corpus]
        n_tokens = max_features

    values = []
    col_ind = []
    row_pointer = [0]

    for features in ind_corpus:
        feature_counter = Counter(features)
        col_ind.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        row_pointer.append(len(values))

    S = scipy.sparse.csr_matrix((values, col_ind, row_pointer),
                                shape=(len(row_pointer) - 1, n_tokens),
                                dtype=int)

    return S

def build_single_bow(text_ind, n_tokens, max_features=None):
    if max_features:
        text_ind = [ind for ind in text_ind if ind < max_features]
        n_tokens = max_features

    single_bow = np.zeros(n_tokens)
    for ind in text_ind:
        single_bow[ind] += 1

    return single_bow

def build_co_occurrence_matrix(ind_corpus, n_tokens, window_size=4):
    row_ind = []
    col_ind = []
    values = []

    for text_ind in ind_corpus:
        for i in range(len(text_ind)):
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(text_ind))):
                if i != j:
                    row_ind.extend([text_ind[i]])
                    col_ind.extend([text_ind[j]])
                    values.extend([1])

    S = scipy.sparse.coo_matrix((values, (row_ind, col_ind)), shape=(n_tokens, n_tokens))

    return S

def matrix_reduce(M, method, n_dim=2, n_iter=10):
    try:
        if method == 'svd':
            decomposition = TruncatedSVD(n_components=n_dim, n_iter=n_iter)
        elif method == 'nmf':
            decomposition = NMF(n_components=n_dim)

        M_reduced = decomposition.fit_transform(M)
        return M_reduced

    except UnboundLocalError:
        print('Choose either svd or nmf method')

def avg_svd_embeddings(text_ind, reduced_co_occurrence_matrix, word2ind):
    i = len(text_ind)

    if i >= 1:
        return sum([reduced_co_occurrence_matrix[ind] for ind in text_ind]) / i
    return np.zeros(reduced_co_occurrence_matrix.shape[1])

def avg_w2v_embeddings(text_token, w2v_model):
    words = [token for token in text_token if token in w2v_model.wv.vocab]
    if len(words)>=1:
        return np.mean(w2v_model.wv[words], axis=0)
    return np.zeros(w2v_model.trainables.layer1_size)