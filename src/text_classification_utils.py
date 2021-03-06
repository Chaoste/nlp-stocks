import re
import string
import itertools

import pandas as pd
import numpy as np

import gensim
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, matthews_corrcoef

from tqdm import tqdm_notebook as tqdm
import spacy
from spacy.lang.en import English
from nltk.corpus import stopwords
from scipy.sparse import issparse

import src.nlp_utils as nlp_utils

# import nltk
# nltk.download()

nlp = spacy.load('en_core_web_sm')  # en
parser = English()

START_DATE = pd.to_datetime('2010-01-04')  # first nyse prices
TRAIN_TEST_SPLIT = pd.to_datetime('2012-12-31')
END_DATE = pd.to_datetime('2013-11-20')  # last reuters article


# ---- Text Processing ------------------------------------------------------- #


def get_index_of_first_article_in_range(news, stocks_ds):
    # On all news: index=45335
    return news[~(news.date >= stocks_ds.prices.date.min()).duplicated(keep='first')].iloc[1].name


def load_news(file_path=None, start_date=None, news=None, end_date=None, quiet=False):
    # skiprows: 48000 -> 10-03-22, 47000 -> 10-02-22, 45400 -> 10-01-05
    assert start_date is not None, 'Required parameter start_date is missing'
    if news is None:
        assert file_path is not None, 'If no news are given, you need to pass a file path'
        news = pd.read_csv(file_path, index_col=0)
    news.date = pd.to_datetime(news.date)  # errors='coerce'
    news = news[news.date >= pd.to_datetime(start_date)]
    if end_date:
        news = news[news.date <= pd.to_datetime(end_date)]
    news.index.name = None
    if quiet:
        print('Amount of news articles:', len(news))
    news = news[news.content.notna()]
    news = news[news.content.str.len() > 100]
    if quiet:
        print('Amount after first filter:', len(news))
    return news


def load_news_clipped(stocks_ds, look_back, forecast, file_path=None, news=None, quiet=False):
    stock_dates = stocks_ds.get_all_prices().date.unique()
    stock_dates.sort()
    min_time = stock_dates[stock_dates.argmin() + look_back]
    max_time = stock_dates[stock_dates.argmax() - forecast]
    return load_news(file_path, min_time, news, max_time, quiet)


def get_relevant_articles(news, occs_per_article, securities_ds, min_occ=5, quiet=True):
    for index, article in tqdm(news.iterrows(), total=len(news)):
        if index not in occs_per_article.index:
            if not quiet:
                print(f'Article with id {index} is not in occs index ({occs_per_article.index[0]} ... {occs_per_article.index[-1]})')
            continue
        best_comp = occs_per_article.loc[index].idxmax()
        if occs_per_article.loc[index, best_comp] > min_occ:
            if not quiet:
                print(index, article.content[:100].replace('\n', '|').replace('\r', '|'), occs_per_article.loc[index].nlargest(1))
            if best_comp in securities_ds.get_companies_without_stocks():
                if not quiet:
                    print(f'Ignoring result for article {index} since {best_comp} is missing stock values')
                two_best_comps = occs_per_article.loc[index].nlargest(2)
                if two_best_comps.iloc[1] > min_occ:
                    if not quiet:
                        print(f'Falling back to {two_best_comps.index[1]}')
                    yield two_best_comps.index[1], article
                continue
            yield best_comp, article

# ---- Numerical Processing -------------------------------------------------- #


def get_occs_per_article(occ_file_name='../data/preprocessed/occurrences/occurrences.csv'):
    occs = pd.read_csv(occ_file_name, index_col=0)
    grouped = occs.groupby(['article_id', 'stock_symbol'], sort=False)
    occs_per_article = grouped.size().reset_index().pivot('article_id', 'stock_symbol')\
        .fillna(0).astype(int)
    occs_per_article.columns = occs_per_article.columns.droplevel(0)
    return occs_per_article


def get_stock_price(stock_symbol, news_article, stocks_ds, look_back=30, forecast=0):
    stock = stocks_ds.get_prices(stock_symbol).reset_index(drop=True)
    assert len(stock) > 0, f'Stock was empty: {stock_symbol}'
    dates = stock.date[stock.date <= news_article.date]
    assert len(dates) > 0, f'Articles date ({news_article.date}) is out of range [{stock.date.min()}, {stock.date.max()}]'
    this_day = (dates - news_article.date).dt.days.idxmax()
    end = this_day + forecast
    start = max(this_day - look_back + 1, 0)
    stock = stock[stock.date.between(stock.date[start], stock.date[end])]
    assert len(stock) > 0, 'Stock has no data for this time'
    # print(stock.date.min(), news_article.date, stock.date.max())
    assert len(stock) == look_back + forecast,\
        f'Stock size is too small ({len(stock)}) for {stock_symbol} on {news_article.date}'
    return stock


def get_movement(rel_dist, epsilon=0.01):
    labels = np.zeros_like(rel_dist)
    labels[rel_dist < -epsilon] = -1
    labels[rel_dist > epsilon] = 1
    return labels


def get_label(*args, epsilon=0.01, **kwargs):
    prices = get_stock_price(*args, **kwargs)
    rel_dist = prices.close / prices.open - 1
    return get_movement(rel_dist, epsilon=epsilon).mean()
    # return get_movement(np.array(rel_dist.mean()))


def categorize_labels(mean_movements, epsilon=0.05):
    discrete_labels = np.zeros_like(mean_movements)
    discrete_labels[mean_movements < -epsilon] = -1
    discrete_labels[mean_movements > epsilon] = 1
    return discrete_labels


def get_discrete_labels(rel_article_tuples, stocks_ds, look_back=30,
                        forecast=0, epsilon_daily_label=0.01,
                        epsilon_overall_label=0.05):
    continuous_labels = np.array([get_label(
        *x, stocks_ds, look_back=look_back, forecast=forecast,
        epsilon=epsilon_daily_label) for x in tqdm(rel_article_tuples)])
    return categorize_labels(continuous_labels, epsilon=epsilon_overall_label)


# --- Classfication Utils ---------------------------------------------------- #


def split_shuffled(rel_articles, rel_labels, ratio=0.8, split_after_shuffle=False, seed=None):
    n_samples = len(rel_articles)
    train_size = int(n_samples * ratio)
    # test_size = n_samples - train_size
    if isinstance(rel_articles[0], tuple):
        # Tuple shape: (stock_symbol, article)
        rel_articles = np.array([nlp_utils.get_plain_content(x[1]) for x in rel_articles])
    if isinstance(rel_articles, pd.DataFrame):
        # else the article contents are already given as word vectors but may as DataFrame
        rel_articles = rel_articles.values
    if seed:
        np.random.seed(seed)
    if split_after_shuffle:
        shuffled_data = list(zip(rel_articles, rel_labels))
        np.random.shuffle(shuffled_data)
        contents, labels = zip(*shuffled_data)
        X_train = contents[:train_size]
        y_train = labels[:train_size]
        X_test = contents[train_size:]
        y_test = labels[train_size:]
    else:
        X_train = rel_articles[:train_size]
        y_train = rel_labels[:train_size]
        X_test = rel_articles[train_size:]
        y_test = np.array(rel_labels[train_size:])
        shuffled_data = list(zip(X_train, y_train))
        np.random.shuffle(shuffled_data)
        X_train, y_train = zip(*shuffled_data)
    return np.array(X_train), np.array(y_train), X_test, y_test


def run(stocks_ds, securities_ds, occs_per_article, news=None, file_path=None,
        time_delta=30, epsilon_daily_label=0.01, epsilon_overall_label=0.05,
        min_occurrences=5, max_articles=None, algorithm=MultinomialNB()):
    stock_dates = stocks_ds.get_all_prices().date.unique()
    stock_dates.sort()
    look_back = abs(min(time_delta, 0))
    forecast = abs(max(time_delta, 0))
    print('-'*40, '\n', f'look_back={look_back}; forecast={forecast}')

    # Load articles for fitting range depending on look back and forecast
    news = load_news_clipped(stocks_ds, look_back, forecast, news=news, file_path=file_path)

    # Get all articles with enough occurrences
    rel_article_tuples = get_relevant_articles(
        news, occs_per_article, securities_ds, min_occ=min_occurrences)
    rel_article_tuples = (x for x in rel_article_tuples if stocks_ds.is_company_available(x[0]))
    if max_articles is not None:
        rel_article_tuples = itertools.islice(rel_article_tuples, max_articles)
    rel_article_tuples = list(rel_article_tuples)
    news = None  # Free space if possible

    assert len(rel_article_tuples) > 0, 'No relevant article tuples'
    assert len(rel_article_tuples) > 100, 'Not enough relevant article tuples'

    discrete_labels = get_discrete_labels(
        rel_article_tuples, stocks_ds, look_back=look_back, forecast=forecast,
        epsilon_daily_label=epsilon_daily_label, epsilon_overall_label=epsilon_overall_label)
    print('Distribution:', ''.join([f'"{cls}": {sum(discrete_labels == cls)} samples; ' for cls in [1, -1, 0]]))

    X_train, y_train, X_test, y_test = split_shuffled(rel_article_tuples, discrete_labels)

    vectorizer = TfidfVectorizer(tokenizer=tokenizeText, analyzer='word', ngram_range=(1, 1))
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer),
                     ('transformer', DenseTransformer()), ('clf', algorithm)])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)

    train_acc = accuracy_score(y_train, y_train_pred)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)

    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    return pipe, acc, mcc, train_acc, train_mcc, (X_train, y_train, X_test, y_test)


STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

# WORD_WITHOUT_NUMBERS = re.compile(r'\b[^\d\W]+\b')
WORD_WITHOUT_NUMBERS = re.compile(r'^[^\d\W]+$')


class CleanTextTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if issparse(X):
            return X.todense()


class W2VTransformer(TransformerMixin):
    def __init__(self, wv=None, wv_path=None):
        super().__init__()
        if wv is None:
            assert wv_path is not None, "If no WV is given, you need to specify where it is located"
            print("Loading Word Embedding. For Google News this took about 10 minutes")
            wv = gensim.models.KeyedVectors.load_word2vec_format(wv_path, binary=True)
            wv.init_sims(replace=True)
        self.wv = wv

    def word_averaging(self, words, index):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in self.wv.vocab:
                mean.append(self.wv.vectors_norm[self.wv.vocab[word].index])
                all_words.add(self.wv.vocab[word].index)
        if not mean:
            print(f"Vectors failed for {index}")
            return None
        # assert mean, f'cannot compute similarity with no input {words}'
        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def transform(self, X, **transform_params):
        processed = (cleanText(text) for text in X)
        processed = (tokenizeText(text) for text in processed)
        processed = [self.word_averaging(text, i) for i, text
                     in tqdm(enumerate(processed), total=len(X))]
        return processed

    def fit(self, X, y=None, **fit_params):
        return self


def get_params(self, deep=True):
        return {}


def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text


def tokenizeText(sample, min_len=2):
    tokens = parser(sample)
    tokens = (tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-"
              else tok.lower_ for tok in tokens)
    tokens = (tok for tok in tokens if len(tok) >= min_len)
    tokens = (tok for tok in tokens if WORD_WITHOUT_NUMBERS.match(tok))
    tokens = (tok for tok in tokens if tok not in STOPLIST)
    tokens = (tok for tok in tokens if tok not in SYMBOLS)
    return list(tokens)


def printNMostInformative(vectorizer, clf, N):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)


def inspect_results(vectorizer, clf, X_train, y_train, X_test, y_test):
    # Print 10 best words for 2 classes
    print("Top 10 features used to predict: ")
    printNMostInformative(vectorizer, clf, 10)
    # Get counts of each word
    vect_pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
    print("Training #2...")
    transform = vect_pipe.fit_transform(X_train, y_train)
    vocab = vectorizer.get_feature_names()
    for i in range(len(X_train)):
        s = ""
        indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
        numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
        for idx, num in zip(indexIntoVocab, numOccurences):
            s += str((vocab[idx], num))
