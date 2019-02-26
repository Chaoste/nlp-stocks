#!/usr/bin/env bash

# Previously run:

# sudo apt-get install git python3.6 python3.6-pip python3.6-dev
# git clone https://Chaoste@github.com/nlp-stocks.git MA
# cd MA
# ./scripts/setup.sh

# git config --global credential.helper cache.
# python3.6 -m pip install --user virtualenv
# python3.6 -m virtualenv -p `which python3.6` venv
# source venv/bin/activate  # Will only propagated
pip install numpy pandas jupyter scikit-learn pyxdameraulevenshtein tqdm elasticsearch ipython-autotime keras spacy matplotlib statsmodels nltk
spacy download en_core_web_sm
python3.6 -m "import nltk; nltk.download('stopwords')"
./scripts/configure_jupyter.sh
