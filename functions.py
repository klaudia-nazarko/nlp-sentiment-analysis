import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import pickle
from collections import Counter
from sklearn.feature_extraction import stop_words
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import cross_validate
import nltk
from nltk import stem
import scipy

def load_pickle(path):
  with open(path, 'rb') as handle:
    return pickle.load(handle)

def save_pickle(variable, path):
    with open(path, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

def avg_cooc_embeddings(text_ind, reduced_co_occurrence_matrix, word2ind):
    i = len(text_ind)

    if i >= 1:
        return sum([reduced_co_occurrence_matrix[ind] for ind in text_ind]) / i
    return np.zeros(reduced_co_occurrence_matrix.shape[1])

def avg_w2v_embeddings(text_token, w2v_model):
    words = [token for token in text_token if token in w2v_model.wv.vocab]
    if len(words)>=1:
        return np.mean(w2v_model.wv[words], axis=0)
    return np.zeros(w2v_model.trainables.layer1_size)

def matrix_normalize(M):
    M_lengths = np.linalg.norm(M, axis=1)
    M_lengths = M_lengths[:, np.newaxis]
    M_normalized = np.divide(M, M_lengths, out=np.zeros_like(M), where=M_lengths!=0)
    return M_normalized

def plot_vectors_2d(M, words, word2ind):
    M_normalized = matrix_normalize(M)

    fig, ax = plt.subplots(figsize=(12,6))
    for word in words:
        i = word2ind[word]

        x = M_normalized[i][0]
        y = M_normalized[i][1]
        plt.scatter(x, y, color='#408fc7')
        plt.text(x, y, word, fontsize=12)
    plt.title('Word vectors in 2D space')
    plt.show()

def model_cv(model, embeddings, y):
    results = []
    for i in range(len(embeddings)):
        cv_results = cross_validate(model,
                                    embeddings[i],
                                    y,
                                    cv=5,
                                    scoring=('accuracy', 'precision', 'recall', 'f1'))
        results.append([np.mean(cv_results['test_accuracy']),
                        np.mean(cv_results['test_f1']),
                        np.mean(cv_results['test_precision']),
                        np.mean(cv_results['test_recall'])])
    return np.stack(results)

def df_model_cv(model_cv, embeddings_names, results_names):
    return pd.DataFrame(model_cv, index=embeddings_names, columns=results_names)