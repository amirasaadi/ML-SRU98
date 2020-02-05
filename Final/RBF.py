from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# RBF part
rbf = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                   optimizer=None)

rbf.fit(X_train, y_train)
performance_RBF = rbf.score(X_test, y_test)
print(performance_RBF)