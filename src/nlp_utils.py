import re
from collections import Counter

import numpy as np
import matplotlib
from spacy import displacy
import en_core_web_sm

from src.datasets import NyseSecuritiesDataset


securities_ds = NyseSecuritiesDataset(file_path='../data/nyse/securities.csv')
companies = securities_ds.get_all_company_names()
nlp = en_core_web_sm.load(disable=['parser', 'tagger'])


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


remove_meta = re.compile(r'-- (.*)\n(?:--.*\n)+[\n\s]*')
remove_meta_2 = re.compile(r'[\s\S]{0,250}(--.*\.html\s*\n)(?:[^-](?:-[^-])*|-(?:[^-]-)*){250}')


def get_meta_indizes(article):
    match = remove_meta.match(article.content)
    if match:
        return match.start(1), match.end(1), match.end()
    else:
        match = remove_meta_2.match(article.content)
        if match:
            return -1, -1, match.end(1)
    return -1, -1, -1


def enrich_with_meta_indizes(article):
    meta_indizes = get_meta_indizes(article)
    article.title_start_idx = meta_indizes[0]
    article.title_end_idx = meta_indizes[1]
    article.head_end_idx = meta_indizes[2]
    return article


# On the first run, the header of the article wasn't producing many FPs
def filter_meta_matches(r, article, article_id, quiet=True):
    # If there's no occurence in this article, we're done
    if not len(r[r.article_id == article_id]):
        return r
    article = enrich_with_meta_indizes(article)
    if article.title_start_idx != -1:  # First regex failed so we have no title
        return r[(r.article_id != article_id) | (r.start_idx >= article.head_end_idx) |
                 (r.start_idx.between(article.title_start_idx, article.title_end_idx) &
                  r.end_idx.between(article.title_start_idx, article.title_end_idx))]
    elif article.head_end_idx != -1:  # Second regex failed so we know nothing
        return r[(r.article_id != article_id) | (r.start_idx >= article.head_end_idx)]
    if not quiet:
        print(f"No regex worked for article {article_id}")
    return r


def get_plain_content(article, quiet=True):
    title_start_idx, title_end_idx, head_end_idx = get_meta_indizes(article)
    if title_start_idx != -1:
        title = article.content[title_start_idx:title_end_idx].replace('\r', '')
        body = article.content[head_end_idx:]
        return f'{title}\n\n{body}'
    if head_end_idx != -1:
        return article.content[head_end_idx:]
    if not quiet:
        print(f"No regex worked for article {article.name}")
    return article.content


def get_cooccurrences(occs):
    grouped = occs.groupby(['article_id', 'stock_symbol'], sort=False)
    occ_per_article = grouped.size().reset_index().pivot('article_id', 'stock_symbol')\
        .fillna(0).astype(int)
    occ_per_article.columns = occ_per_article.columns.droplevel(0)
    comp_does_occure = (occ_per_article != 0).astype(int)
    cooc = comp_does_occure.T.dot(comp_does_occure)
    # comps_article_counts = np.diag(cooc)  # (occ_per_article != 0).astype(int).sum(axis=0)
    np.fill_diagonal(cooc.values, 0)
    cooc.index.levels[0].name = None
    cooc.index.levels[1].name = None
    return cooc


def readable_cooccurrences(cooc):
    beautified = cooc.stack()
    beautified.index.levels[0].name = 'Comp A'
    beautified.index.levels[1].name = 'Comp B'
    beautified = beautified[[p[0] < p[1] for p, v in beautified.iteritems()]].nlargest(100)
    beautified = beautified.reset_index()
    beautified.columns = list(beautified.columns[:2]) + ['Articles']
    beautified['Comp A'] = [f'{securities_ds.get_company_name(x)} [{x}]' for x in beautified['Comp A']]
    beautified['Comp B'] = [f'{securities_ds.get_company_name(x)} [{x}]' for x in beautified['Comp B']]
    return beautified
