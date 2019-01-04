#!/usr/bin/env bash

# Previously run:

# sudo apt-get install git python3.6 python3.6-pip python3.6-dev
# git clone https://Chaoste@github.com/nlp-stocks.git MA
# cd MA
# ./scripts/setup.sh

git config --global credential.helper cache.
pyhon3.6 -m pip install --user virtualenv
pyhon3.6 -m virtualenv venv
source venv/bin/activate  # Will only propagated
pip install numpy pandas jupyter scikit-learn pyxdameraulevenshtein tqdm elasticsearch ipython-autotime keras spacy matplotlib statsmodels
spacy download en_core_web_sm
./scripts/configure_jupyter.sh
