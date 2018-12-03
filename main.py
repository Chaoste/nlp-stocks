from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

X, y = samples_generator.make_classification(
    n_informative=5, n_redundant=0, random_state=42)

anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)

print(X)
print(y)
print(anova_svm.predict(X))
print(anova_svm.score(X, y))

# Get selected features chosen by anova_filter
print(anova_svm.named_steps.anova.get_support())
