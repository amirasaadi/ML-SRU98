from Dataset.loader import load_hoda
from sklearn import model_selection
from sklearn.svm import SVC


x_train, y_train, x_test, y_test = load_hoda()

kfold = model_selection.KFold(n_splits=10)
model = SVC()
results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())