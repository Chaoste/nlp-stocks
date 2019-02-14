import string

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

import spacy
from spacy.lang.en import English
from nltk.corpus import stopwords

# import nltk
# nltk.download()

nlp = spacy.load('en_core_web_sm')  # en
parser = English()

START_DATE = pd.to_datetime('2010-01-04')  # first nyse prices
TRAIN_TEST_SPLIT = pd.to_datetime('2012-12-31')
END_DATE = pd.to_datetime('2013-11-20')  # last reuters article


# ---- Text Processing ------------------------------------------------------- #


def load_news(file_path, start_date):
    news = pd.read_csv(file_path, index_col=0)
    news.date = pd.to_datetime(news.date)  # errors='coerce'
    news = news[news.date >= pd.to_datetime(start_date)]
    #  news.columns = pd.read_csv(REUTERS, index_col=0, nrows=0).columns
    news.index.name = None
    print('Amount of news articles:', len(news))
    news = news[news.content.notna()]
    news = news[news.content.str.len() > 100]
    print('Amount after first filter:', len(news))
    return news


def get_relevant_articles(news, occs_per_article, securities_ds, min_occ=5, quiet=True):
    for index, article in news.iterrows():
        idx = f'r{index}'
        if idx not in occs_per_article.index:
            continue
        if article.date > TRAIN_TEST_SPLIT:
            continue
        best_comp = occs_per_article.loc[idx].idxmax()
        if occs_per_article.loc[idx, best_comp] > min_occ:
            if not quiet:
                print(idx, article.content[:100].replace('\n', '|').replace('\r', '|'), occs_per_article.loc[idx].nlargest(1))
            if best_comp in securities_ds.get_companies_without_stocks():
                if not quiet:
                    print(f'Ignoring result for article {idx} since {best_comp} is missing stock values')
                two_best_comps = occs_per_article.loc[idx].nlargest(2)
                if two_best_comps.iloc[1] > min_occ:
                    if not quiet:
                        print(f'Falling back to {two_best_comps.index[1]}')
                    yield two_best_comps.index[1], article
                continue
            yield best_comp, article

# ---- Numerical Processing -------------------------------------------------- #

def get_occs_per_article(occ_file_name='./reports/occurrences-reuters-v2.csv'):
    occs = pd.read_csv(occ_file_name, index_col=0)
    grouped = occs.groupby(['article_id', 'stock_symbol'], sort=False)
    occs_per_article = grouped.size().reset_index().pivot('article_id', 'stock_symbol')\
        .fillna(0).astype(int)
    occs_per_article.columns = occs_per_article.columns.droplevel(0)
    return occs_per_article


def get_stock_price(stock_symbol, news_article, stocks_ds, look_back=30, forecast=0, test=False):
    assert not test, 'Test set currently not supported'
    stock = stocks_ds.get_prices(stock_symbol).reset_index(drop=True)
    assert len(stock) > 0, f'Stock was empty: {stock_symbol}'
    if news_article.date not in stock.date:
        dates = stock.date[stock.date < news_article.date]
        this_day = (dates - news_article.date).idxmax()
    else:
        this_day = (dates == news_article.date).idxmax()
    end = this_day + forecast
    start = max(this_day - look_back + 1, 0)
    stock = stock[stock.date.between(stock.date[start], stock.date[end])]
    assert len(stock) > 0, 'Stock has no data for this time'
    # print(stock.date.min(), news_article.date, stock.date.max())
    assert len(stock) == look_back + forecast, f'Stock size is too small ({len(stock)}) for {stock_symbol} on {news_article.date}'
    return stock


def get_movement(rel_dist, epsilon=0.01):
    labels = np.zeros(len(rel_dist))
    labels[rel_dist < -epsilon] = -1
    labels[rel_dist > epsilon] = 1
    return labels


def get_label(*args, **kwargs):
    prices = get_stock_price(*args, **kwargs)
    rel_dist = prices.close / prices.open - 1
    prices['movement'] = get_movement(rel_dist)
    return prices.movement.mean()


def categorize_labels(mean_movements, epsilon=0.05):
    discrete_labels = np.zeros_like(mean_movements)
    discrete_labels[mean_movements < -epsilon] = -1
    discrete_labels[mean_movements > epsilon] = 1
    return discrete_labels

# --- Classfication Utils ---------------------------------------------------- #


def split_shuffled(rel_article_tuples, rel_labels, ratio=0.8):
    n_samples = len(rel_article_tuples)
    shuffled_data = [(*x, y) for x, y in zip(rel_article_tuples, rel_labels)]
    np.random.seed(42)
    np.random.shuffle(shuffled_data)

    contents = np.array([nlp_utils.get_plain_content(x[1]) for x in shuffled_data])
    labels = np.array([x[2] for x in shuffled_data])
    train_size = int(n_samples * ratio)
    # test_size = n_samples - train_size

    X_train = contents[:train_size]
    y_train = labels[:train_size]
    X_test = contents[train_size:]
    y_test = labels[train_size:]
    return X_train, y_train, X_test, y_test


STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]


class CleanTextTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self


def get_params(self, deep=True):
        return {}


def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text


def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


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
