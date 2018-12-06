import os
import re
import logging

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
import pandas as pd

DATA_DIR = "data"
NYSE_SECURITIES = os.path.join(DATA_DIR, 'nyse', 'securities.csv')


class NyseSecuritiesDataset():

    def __init__(self, name: str = 'NyseSecuritiesDataset',
                 file_path: str = NYSE_SECURITIES):
        self.name = name
        self._orig_data = None
        self._data = None
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def __str__(self) -> str:
        return self.name

    def load(self):
        """Load data"""
        self.logger.debug(
            'Reading NYSE securities data (takes 100 ms)...')
        self._orig_data = pd.read_csv(self.file_path)
        self._data = self._clean_up(self._orig_data)

    def data(self) -> (pd.DataFrame):
        """Return data, load if necessary"""
        if self._data is None:
            self.load()
        return self._data

    def get_company(self, name) -> str:
        securities = self.data()
        return securities[securities['Name'] == name].iloc[0]['Ticker symbol']

    def get_company_name(self, sym) -> str:
        securities = self.data()
        return securities[securities['Ticker symbol'] == sym].iloc[0]['Name']

    def get_all_company_names(self) -> (pd.DataFrame):
        return self.data()[['Ticker symbol', 'Name']].drop_duplicates('Name')

    def get_most_similar_company(
            self, fuzzy_name, debug=False, quiet=True) -> (pd.DataFrame):
        acceptance_rate = 0.3
        # Test on all companies: https://regex101.com/r/RfZsbU/2/
        re_comp_suffix = re.compile(
            r",?[^\w](Corp\ A|A\ Corp|\&\ Co|Svc\.Gp|Corp(?:oration|'s)?|"
            "Inc|Co(?:s|mpany)?|Ltd|Limited|Plc|Group|(?:International|Int'l)"
            "(?:\ Inc)?|Int'l\ Industries)\.?$", re.IGNORECASE)
        fuzzy_name = re_comp_suffix.sub('', fuzzy_name)

        def distance(row):
            return normalized_damerau_levenshtein_distance(
                re_comp_suffix.sub('', row['Name']), fuzzy_name)

        companies = self.get_all_company_names()
        distances = pd.DataFrame(
            companies.apply(distance, axis=1),
            columns=['damerau_lev'])
        distances = pd.concat([companies['Name'], distances], axis=1)
        distances.sort_values('damerau_lev', inplace=True)
        if not quiet:
            print(distances.head())
        if distances.iloc[0]['damerau_lev'] >= acceptance_rate:
            if debug:
                return None, None
            return None
        if debug:
            return distances.iloc[0]['Name'], distances.iloc[0]['damerau_lev']
        return distances.iloc[0]['Name']

    def _clean_up(self, orig):
        cleaned = orig.copy()
        cleaned.drop('SEC filings', axis=1, inplace=True)

        cleaned['Date first added'] = pd.to_datetime(
            cleaned['Date first added'], errors='coerce')

        # Before: Discovery Communications-A
        idx = cleaned.index[cleaned['Ticker symbol'] == 'DISCA'][0]
        cleaned.loc[idx, 'Security'] = 'Discovery Communications Class A'
        # Before: Discovery Communications-C
        idx = cleaned.index[cleaned['Ticker symbol'] == 'DISCK'][0]
        cleaned.loc[idx, 'Security'] = 'Discovery Communications Class C'
        # Before: Under Armour
        idx = cleaned.index[cleaned['Ticker symbol'] == 'UA'][0]
        cleaned.loc[idx, 'Security'] = 'Under Armour Class C'
        # Before: Under Armour
        idx = cleaned.index[cleaned['Ticker symbol'] == 'UAA'][0]
        cleaned.loc[idx, 'Security'] = 'Under Armour Class A'

        columns = cleaned.columns.tolist()
        columns.insert(1, 'Name')
        regex_pat = re.compile(r'\WClass (A|B|C)$', flags=re.IGNORECASE)
        cleaned['Name'] = cleaned['Security'].str.replace(regex_pat, '')
        cleaned = cleaned[columns]  # Change order

        return cleaned

    def get_securities_problems(self) -> (pd.DataFrame):
        self.data()
        return self._orig_data[
            (self._orig_data['Ticker symbol'].str.find('GOOG') != -1) |
            (self._orig_data['Ticker symbol'].str.find('FOX') != -1) |
            (self._orig_data['Ticker symbol'].str.find('DISC') != -1) |
            (self._orig_data['Ticker symbol'].str.find('NWS') != -1) |
            (self._orig_data['Security'] == 'Under Armour')
        ]
