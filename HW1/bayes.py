from HW1 import constants
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd

data_set = io.loadmat('Data_hoda_full.mat')
all_performance=[]
# test and training set
for size in range(constants.MIN_TRAIN_SIZE,constants.MAX_TRAIN_SIZE,1000):
    train_size=size
    test_size=int(train_size*0.2)
    X_train_orginal = np.squeeze(data_set['Data'][:train_size])
    y_train = np.squeeze(data_set['labels'][:train_size])
    X_test_original = np.squeeze(data_set['Data'][train_size:train_size + test_size])
    y_test = np.squeeze(data_set['labels'][train_size:train_size + test_size])

    # resize
    X_train_10by10 = [resize(img, (10, 10)) for img in X_train_orginal]
    X_test_10by10 = [resize(img, (10, 10)) for img in X_test_original]

    # reshape
    X_train_reshaped = [x.reshape(100) for x in X_train_10by10]
    X_test_reshaped = [x.reshape(100) for x in X_test_10by10]
    del X_train_orginal,X_test_original,X_test_10by10,X_train_10by10
    # seprating
    numbers_test = [[] for _ in range(10)]
    for i in range(train_size):
        numbers_test[y_train[i]].append(X_train_reshaped[i])

    # size of each class
    number_or_each_class = []
    for i in range(10):
        number_or_each_class.append(len(numbers_test[i]))

    # prior prabability
    total_ppl = train_size
    probabilities = []
    for i in range(10):
        probabilities.append(number_or_each_class[i] / total_ppl)

    # calculating mean of each column
    data_means = []
    for i in range(10):
        data_means.append(np.mean(numbers_test[i], axis=0))

    # calculating covariance
    data_covariance = []
    for i in range(10):
        data_covariance.append(np.cov(np.asarray(numbers_test[i]).T))

    guess = []
    for test in X_test_reshaped:
        probability = []
        for i in range(10):
            d = np.linalg.det(data_covariance[i])

            t = test - data_means[i]

            c = np.log1p(d)

            sigma = np.linalg.pinv(data_covariance[i])

            probability.append(c - (t.T @ sigma @ t.T)+probabilities[i])

        guess.append(probability.index(max(probability)))


    c=0
    for i in range(len(guess)):
        if guess[i]==y_test[i]:
            c+=1
    c_performance=c*100/test_size
    print(c_performance)
    all_performance.append([c_performance,train_size])
print(all_performance)

