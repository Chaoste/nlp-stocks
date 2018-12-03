from sklearn.pipeline import Pipeline

FORECAST_OUT = 30


stocks_ds = NyseStocksDataset(file_path='../data/nyse/prices.csv')
X_train =
X =

X_train, y_train, X_test, y_test =


anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
