# Data for Master Thesis on Stock Price Prediction

- Title: *Prediction of Stock Price Movements based on Corporate Relationships*
- Author: *Thomas Kellermeier*
- Supervisor: *Tim Repke, Ralf Krestel*


The preprocessed data in this folder is splitted into the following categories:

### 1. Financial news articles from Reuters and Bloomberg

- 554.914 articles from 2006-10-20 to 2013-11-26
- 106.518 Reuters, 448.395 Bloomberg
- The table contains a unique ID, previously used IDs (for mapping between both formats), publishing date, the original filename and the plain content
- 651 articles are missing the content (maybe the file were corrupted or the dataset just incomplete)
- The datasets contains several duplicate filenames because updates on article are given as separate files. The content might even bec
ome None, possibly because the article was removed (which explains only a few of the missing article contents).
- The raw data can be found at `/san2/data/websci/reuters` and `/san2/Data/websci/bloomberg`


### 2. S&P 500 Stock Prices (NYSE)

This collection consists of the following datasets:
- Stock Prices, Fundamentals, Securities for all 500 companies of the S&P 500 index. Source: [Kaggle NYSE](https://www.kaggle.com/dgaw
lik/nyse)
- GSPC (Common S&P 500 Index). Source: [Kaggle SP500](https://www.kaggle.com/benjibb/sp500-since-1950)
- VIX (CBOE Volatility Index for S&P 500). Source: [Kaggle VIX](https://www.kaggle.com/lp187q/vix-index-until-jan-202018)
- Historical Components of the S&P 500 index. Source: [nemozny.github.io](https://nemozny.github.io/datasets/)


### 3. Entities extracted using SpaCy NER:

- The table contains article\_id (index in news table), the original text representing the entity, the start and the end index within
the articles content, and a label provided by SpaCy
- Entities are only extracted from title and body, the residual part of meta information is mostly filtered out. This was achieved by
applying some regex logic


### 4. Occurrences of known NYSE Companies:

- The entity text is matched against the full name of companies from the S&P500 index (see nyse datset). Again, this was achieved by some regex logic which was only evaluated manually by looking at some random samples and news articles.
- The table contains article\_id (news index), the stock symbol of the matches company and again the same entity informations as above
 except for the label since it is always "ORG" for these entities.


### 5. Occurrences per article (`occurrences-matrix.csv`):

- This 554914x500 matric counts for each article and company the amount of occurrences.



### 6. Cooccurrences:

- This 500x500 matrix counts the amount of articles in which two companies occur together.
