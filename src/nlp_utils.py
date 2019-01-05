import numpy as np
import matplotlib
from spacy import displacy
from collections import Counter
import en_core_web_sm

from src.datasets import NyseSecuritiesDataset


securities_ds = NyseSecuritiesDataset(file_path='../data/nyse/securities.csv')
companies = securities_ds.get_all_company_names()
nlp = en_core_web_sm.load()


def find_nyse_corporations(article_text, quiet=True):
    doc = nlp(article_text)
    if not quiet:
        print('Found ORG occurrences:',
              Counter([x.label_ for x in doc.ents])['ORG'])
    counts = Counter([ent.text for ent in doc.ents if ent.label_ == 'ORG'])
    matches = [[key, counts[key], securities_ds.get_most_similar_company(key)]
               for key in counts]
    matched_stocks = dict([(x[0], x[2]) for x in matches if x[2] is not None])
    if not quiet:
        display_occurrences(doc, matched_stocks)
    return [(ent, securities_ds.get_company(matched_stocks[ent.text])) for ent in doc.ents
            if ent.text in matched_stocks]


def _to_rgb(cmap, step):
    r, g, b, _ = cmap(step)
    return f'rgb({int(256*r)}, {int(256*g)}, {int(256*b)})'


def get_colors(ents):
    cmap_name = 'Set3' if len(ents) > 8 else 'Pastel2'
    steps = np.linspace(0, 1, 12 if len(ents) > 8 else 8)
    cmap = matplotlib.cm.get_cmap(cmap_name)
    return dict([(e, _to_rgb(cmap, steps[i])) for i, e in enumerate(ents)])


def display_occurrences(doc, matched_stocks):
    ents = [ent for ent in doc.ents if ent.text in matched_stocks]
    hits = [{'start': ent.start_char, 'end': ent.end_char,
             'label': f'{ent.label_} ('
             f'{securities_ds.get_company(matched_stocks[ent.text])})'}
            for ent in ents]
    print('Organisations after filtering:')
    for org, amount in Counter([x['label'] for x in hits]).most_common():
        print(f' - {org}: {amount}')
    ent_names = np.unique([x['label'] for x in hits])
    options = {'ents': ent_names, 'colors': get_colors(ent_names)}
    displacy.render({
        'text': str(doc),
        'ents': hits,
        'title': None
    }, jupyter=True, options=options, style='ent', manual=True)
