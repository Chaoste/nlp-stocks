import os
import re
import requests
from datetime import datetime

import pandas as pd
import numpy as np
from yahoofinancials import YahooFinancials

HOME = ".."
DATA_DIR = "data"

NYSE_FUNDAMENTALS = os.path.join(HOME, DATA_DIR, "nyse", "fundamentals.csv")
NYSE_PRICES = os.path.join(HOME, DATA_DIR, "nyse", "prices.csv")
NYSE_PRICES_SPLIT = os.path.join(HOME, DATA_DIR, "nyse", "prices-split-adjusted.csv")
NYSE_SECURITIES = os.path.join(HOME, DATA_DIR, "nyse", "securities.csv")

# Define global for export

fundamentals = None
orig_securities = None
securities = None
prices = None
prices_split = None

def load():
    global fundamentals, orig_securities, securities, prices, prices_split
    fundamentals = pd.read_csv(NYSE_FUNDAMENTALS)
    orig_securities = pd.read_csv(NYSE_SECURITIES)
    securities = clean_up_securities(orig_securities)
    prices = pd.read_csv(NYSE_PRICES)
    prices_split = pd.read_csv(NYSE_PRICES_SPLIT)

    prices['date'] = pd.to_datetime(prices['date'], errors='coerce')  # [datetime.strptime(x[:10], '%Y-%m-%d') for x in prices['date']]
    prices_split['date'] = pd.to_datetime(prices_split['date'], errors='coerce')

def check_existence():
    if securities is None:
        # Module was reloaded with initialization
        load()
    
def get_name(symbol):
    check_existence()
    idx = securities.index[securities['Ticker symbol'] == symbol][0]
    return securities.loc[idx, 'Name']
    
def clean_up_securities(orig):
    cleaned = orig.copy()
    cleaned.drop('SEC filings', axis=1, inplace=True)
    
    cleaned['Date first added'] = pd.to_datetime(cleaned['Date first added'], errors='coerce')
    
    idx = cleaned.index[cleaned['Ticker symbol'] == 'DISCA'][0]
    cleaned.loc[idx, 'Security'] = 'Discovery Communications Class A'  # Before: Discovery Communications-A
    idx = cleaned.index[cleaned['Ticker symbol'] == 'DISCK'][0]
    cleaned.loc[idx, 'Security'] = 'Discovery Communications Class C'  # Before: Discovery Communications-C
    idx = cleaned.index[cleaned['Ticker symbol'] == 'UA'][0]
    cleaned.loc[idx, 'Security'] = 'Under Armour Class C'  # Before: Under Armour
    idx = cleaned.index[cleaned['Ticker symbol'] == 'UAA'][0]
    cleaned.loc[idx, 'Security'] = 'Under Armour Class A'  # Before: Under Armour
    
    columns = cleaned.columns.tolist()
    columns.insert(1, 'Name')
    regex_pat = re.compile(r'\WClass (A|B|C)$', flags=re.IGNORECASE)
    cleaned['Name'] = cleaned['Security'].str.replace(regex_pat, '')
    cleaned = cleaned[columns]  # Change order
    
    return cleaned


def get_securities_problems():
    check_existence()
    return orig_securities[
        (orig_securities['Ticker symbol'].str.find('GOOG') != -1) |
        (orig_securities['Ticker symbol'].str.find('FOX') != -1) |
        (orig_securities['Ticker symbol'].str.find('DISC') != -1) |
        (orig_securities['Ticker symbol'].str.find('NWS') != -1) |
        (orig_securities['Security'] == 'Under Armour')
    ]


def query_yahoo_name(symbol):
    check_existence()
    r = requests.get(f'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={symbol}&region=1&lang=en')
    response = r.json()['ResultSet']['Result']
    assert len(response), f'Could not find a match for {symbol}'
    best_result = response[0]
    return best_result['name']


# https://github.com/JECSand/yahoofinancials
def get_fundamentals(symbol):
    check_existence()
    yahoo_financials = YahooFinancials(symbol)
    stock_type = yahoo_financials.get_stock_quote_type_data()
    annual_stmt = yahoo_financials.get_financial_stmts('annual', ['balance', 'income', 'cash'])
    return stock_type, annual_stmt
