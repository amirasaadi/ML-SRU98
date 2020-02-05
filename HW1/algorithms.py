import numpy as np


class KNN:
    """KNN algorithm"""
    def __init__(self, k):
        self._k = k

    def exe(self, test, x_train, y_train):
        distances = []
        for i, train in enumerate(x_train):
            distance = np.sum(np.square(test - train))
            distances.append([distance, i])
        distances.sort()
        distances = distances[:self._k]
        # get the index of image
        index = [distance_i[1] for distance_i in distances]
        # get the label
        candidate_labels = [y_train[number] for number in index]
        # voting
        count = np.bincount(np.asarray(candidate_labels))
        guess = np.argmax(count)
        return guess


class Parzen:
    def __init__(self, r):
        self._r = r

    def exe(self, test, x_train, y_train):
        labels_inside_window = []
        for index, train in enumerate(x_train):
            distance = np.sum(np.square(test - train))
            if distance <= self._r:
                labels_inside_window.append(y_train[index])

        if labels_inside_window:
            count = np.bincount(labels_inside_window)
            result = np.argmax(count)
            return result
        else:
            return None


class Bayesian:

    def seperate(self):
        pass

    def calculate_size_of_class(self):
        pass

    def calculate_prior_probability(self):
        pass

    def calculate_mean_of_each_column(self):
        pass

    def calculate_covariance(self):
        pass

    def bayes_formula(self):
        pass

    def exe(self, test, x_train, y_train):
        pass
