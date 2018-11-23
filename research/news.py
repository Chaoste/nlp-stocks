import os
import glob
import re
from datetime import datetime

import pandas as pd
from tqdm import tqdm_notebook as tqdm

import nyse

HOME = ".."
DATA_DIR = os.path.join(HOME, "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed", "occurrences")
REUTERS = os.path.join(DATA_DIR, "bloomberg_reuters", "raw", "reuters")
BLOOMBERG = os.path.join(DATA_DIR, "bloomberg_reuters", "raw", "bloomberg")
OUTPUT_DIR = os.path.join(DATA_DIR, "bloomberg_reuters", "raw", "bloomberg")

# Define global for export

files_reuters = None
files_bloomberg = None
articles_reuters = None
articles_bloomberg = None


def load():
    global articles_reuters, articles_bloomberg, files_reuters, files_bloomberg
    files_reuters = find_files(REUTERS)
    files_bloomberg = find_files(BLOOMBERG)  # ~1min

    # ~5min
    print('Reading all Reuters files')
    articles_reuters = pd.DataFrame(
        [read_content(x, '%Y%m%d') for x in files_reuters],
        columns=['date', 'filename', 'content'])
    print('Reading all Bloomberg files')
    articles_bloomberg = pd.DataFrame(
        [read_content(x) for x in files_bloomberg],
        columns=['date', 'filename', 'content'])


def check_existence():
    if articles_reuters is None:
        # Module was reloaded with initialization
        load()


def get_articles():
    check_existence()
    return articles_reuters, articles_bloomberg


def find_files(path_dir):
    files = []
    for folder in glob.iglob(os.path.join(path_dir, '*'), recursive=True):
        for filename in glob.iglob(os.path.join(folder, '*'), recursive=True):
            assert os.path.isfile(filename), filename
            files.append(filename)
    return files


def read_content(path, datef='%Y-%m-%d'):
    complete_dir, filename = os.path.split(path)
    complete_dir, short_dir = os.path.split(complete_dir)
    with open(path, encoding='utf8') as file:
        try:
            content = file.read()
        except Exception as e:
            print('Failed reading', path)
            raise e
    publish_date = datetime.strptime(short_dir, datef)
    return publish_date, filename, content

# --- Processing ------------------------------------------------------------- #

df_occurrences_reuters = None
df_occurrences_bloomberg = None

def count_occurrences(articles, start=0, end=None):
    nyse.check_existence()
    check_existence()

    end = end if end is not None else len(articles)
    df_occurrences = pd.DataFrame(0, index=articles.index[start:end],
                                  columns=nyse.securities['Ticker symbol'])
    for idx, row in tqdm(articles.iloc[start:end].iterrows(), total=end-start):
        for symbol in df_occurrences.columns:
            company = nyse.get_name(symbol)
            occurrences = [m.start()
                           for m in re.finditer(company, row['content'])]
            if len(occurrences):
                df_occurrences[symbol][idx] = len(occurrences)
    return df_occurrences


def count_reuters():
    global df_occurrences_reuters
    df_occurrences_reuters = count_occurrences(articles_reuters)  # ~7h
    print(f'Found {df_occurrences_reuters.sum().sum()} occurrences!')
    df_occurrences_reuters.to_csv(os.path.join(PREPROCESSED_DIR, 'reuters_occurrences.csv'))


def count_bloomberg():
    global df_occurrences_bloomberg
    print("Counting: 0-100k")
    df_occurrences_bloomberg1 = count_occurrences(articles_bloomberg, 0, 100000)
    df_occurrences_bloomberg1.to_csv(os.path.join(PREPROCESSED_DIR, 'bloomberg_occurrences_1_100k.csv'))

    print("Counting: 100k-200k")
    df_occurrences_bloomberg2 = count_occurrences(articles_bloomberg, 100000, 200000)  # ~8h
    df_occurrences_bloomberg2.to_csv(os.path.join(PREPROCESSED_DIR, 'bloomberg_occurrences_2_100k.csv'))

    print("Counting: 200k-300k")
    df_occurrences_bloomberg3 = count_occurrences(articles_bloomberg, 200000, 300000)  # ~6.5h
    df_occurrences_bloomberg3.to_csv(os.path.join(PREPROCESSED_DIR, 'bloomberg_occurrences_3_100k.csv'))

    print("Counting: 300k-400k")
    df_occurrences_bloomberg4 = count_occurrences(articles_bloomberg, 300000, 400000)  # ~6.5h
    df_occurrences_bloomberg4.to_csv(os.path.join(PREPROCESSED_DIR, 'bloomberg_occurrences_4_100k.csv'))

    print("Counting: 400k-end")
    df_occurrences_bloomberg5 = count_occurrences(articles_bloomberg, 400000)  # ~3h
    df_occurrences_bloomberg5.to_csv(os.path.join(PREPROCESSED_DIR, 'bloomberg_occurrences_5_50k.csv'))

    df_occurrences_bloomberg = pd.concat([
        df_occurrences_bloomberg1, df_occurrences_bloomberg2,
        df_occurrences_bloomberg3, df_occurrences_bloomberg4,
        df_occurrences_bloomberg5])
    print(f'Found {df_occurrences_bloomberg.sum().sum()} occurrences!')
    df_occurrences_bloomberg.to_csv(os.path.join(PREPROCESSED_DIR, 'bloomberg_occurrences.csv'))
