from Dataset.loader import load_hoda
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier


x_train, y_train, x_test, y_test = load_hoda()

kfold = model_selection.KFold(n_splits=10)
results = []
for k in range(1,11,2):
    model = KNeighborsClassifier(n_neighbors=k)
    result = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)
    results.append(result.mean())
print(results)
