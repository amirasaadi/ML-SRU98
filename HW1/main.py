from scipy import io
from HW1 import constants
from random import randint, sample
import numpy as np
from skimage.transform import resize
from HW1 import algorithms
from matplotlib import pyplot as plt




def load_image():
    dataset = io.loadmat('Data_hoda_full.mat')
    # loading data 1 to n
    # trains
    x_train_original = np.squeeze(dataset['Data'][:constants.SIZE_TRAIN])
    y_train = np.squeeze(dataset['labels'][:constants.SIZE_TRAIN])
    # tests
    x_test_original = np.squeeze(dataset['Data'][constants.SIZE_TRAIN:constants.SIZE_TRAIN + constants.SIZE_TEST])
    y_test = np.squeeze(dataset['labels'][constants.SIZE_TRAIN:constants.SIZE_TRAIN + constants.SIZE_TEST])
    return x_train_original, x_test_original, y_train, y_test


def load_random_image():
    # generating random numbers
    # new stack overflow code
    random_train_indexs = sample(range(0,60000),constants.SIZE_TRAIN)
    random_train_indexs = []
    while len(random_train_indexs) != constants.constants.SIZE_TRAIN:
        random_train_indexs.append(randint(0, 60000))
        set(random_train_indexs)
    dataset = io.loadmat('Data_hoda_full.mat')
    # loding data randomly
    x_train_orginal = []
    y_train = []
    x_test_original = []
    y_test = []
    random_test_indexs = []
    for number in random_train_indexs:
        x_train_orginal.append(np.squeeze(dataset['Data'][number]))
        y_train.append(np.squeeze(dataset['labels'][number]))
    for number in random_test_indexs:
        x_test_original.append(np.squeeze(dataset['Data'][constants.SIZE_TRAIN:constants.SIZE_TRAIN+constants.SIZE_TEST]))
        y_test.append(np.squeeze(dataset['labels'][constants.SIZE_TRAIN:constants.SIZE_TRAIN+constants.SIZE_TEST]))
    return None


def resize_image(x_train_orginal, x_test_original):
    # resizing
    image_size = constants.IMAGE_SIZE
    x_train_nbyn = [resize(img, (image_size, image_size)) for img in x_train_orginal]
    x_test_nby_n = [resize(img, (image_size, image_size)) for img in x_test_original]
    return x_train_nbyn, x_test_nby_n


def reshape_image(x_train_nbyn, x_test_nby_n):
    # reshape
    size = constants.IMAGE_SIZE
    x_train = [x.reshape(size * size) for x in x_train_nbyn]
    x_test = [x.reshape(size * size) for x in x_test_nby_n]
    return x_train, x_test


def calculate_knn_performance(x_train, x_test, y_train, y_test, k):
    knn_object = algorithms.KNN(k)
    knn_guesses = []
    for test in x_test:
        knn_guess = knn_object.exe(test, x_train, y_train)
        knn_guesses.append(knn_guess)

    knn_true_guess_number = 0
    for index, knn_guess in enumerate(knn_guesses):
        if knn_guess == y_test[index]:
            knn_true_guess_number += 1

    knn_performance = knn_true_guess_number * 100 / len(x_test)
    return knn_performance


def calculate_knn_performance_with_different_k(x_train, x_test, y_train, y_test):
    # list [[k,performance],...]
    knn_performances = []
    for k in range(constants.MIN_K, constants.MAX_K, 2):
        performance = calculate_knn_performance(x_train, x_test, y_train, y_test, k)
        knn_performances.append([k, performance])
    return knn_performances


def calculate_knn_performance_with_different_train_size():
    dataset = io.loadmat('Data_hoda_full.mat')
    knn_performances = []
    # loading data 1 to n
    # trains
    for train_size in range(constants.MIN_TRAIN_SIZE, constants.MAX_TRAIN_SIZE, 1000):
        x_train_original = np.squeeze(dataset['Data'][:train_size])
        y_train = np.squeeze(dataset['labels'][:train_size])
        # tests
        x_test_original = np.squeeze(dataset['Data'][train_size:train_size + int(train_size * 0.2)])
        y_test = np.squeeze(dataset['labels'][train_size:train_size + int(train_size * 0.2)])
        # resizing
        image_size = constants.IMAGE_SIZE
        x_train_nbyn = [resize(img, (image_size, image_size)) for img in x_train_original]
        x_test_nby_n = [resize(img, (image_size, image_size)) for img in x_test_original]
        # reshaping
        size = constants.IMAGE_SIZE
        x_train = [x.reshape(size * size) for x in x_train_nbyn]
        x_test = [x.reshape(size * size) for x in x_test_nby_n]

        del x_test_original, x_train_original, x_test_nby_n, x_train_nbyn
        knn_performances.append(calculate_knn_performance(x_train, x_test, y_train, y_test, k=1))
    return knn_performances


def calculate_parzen_performance(x_train, x_test, y_train, y_test, r):
    parzen_object = algorithms.Parzen(r)
    parzen_guesses = []
    for test in x_test:
        parzen_guess = parzen_object.exe(test, x_train, y_train)
        parzen_guesses.append(parzen_guess)

    parzen_true_guess_number = 0
    for index, parzen_guess in enumerate(parzen_guesses):
        if parzen_guess == y_test[index]:
            parzen_true_guess_number += 1

    parzen_performance = parzen_true_guess_number * 100 / constants.SIZE_TEST
    return parzen_performance


def calculate_parzen_performance_with_different_r(x_train, x_test, y_train, y_test):
    # list [[k,performance],...]
    parzen_performances = []
    for r in range(constants.MIN_R, constants.MAX_R):
        performance = calculate_parzen_performance(x_train, x_test, y_train, y_test, r)
        parzen_performances.append([r, performance])
    return parzen_performances


def pre_processing():
    x_train_original, x_test_original, y_train, y_test = load_image()
    x_train_nbyn, x_test_nby_n = resize_image(x_train_original, x_test_original)
    x_train, x_test = reshape_image(x_train_nbyn, x_test_nby_n)
    del x_test_original, x_train_original, x_test_nby_n, x_train_nbyn
    return x_train, x_test, y_train, y_test


def plot_knn_when_k_is_different(knn_performances):
    x_labels = ([item[0] for item in knn_performances])
    y_pos = np.arange(len(x_labels))
    performance = [item[1] for item in knn_performances]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel('Accuracy')
    plt.title('Knn accuracy with different k')
    plt.show()


def k_mean_NN(x_train, x_test, y_train, y_test):
    # seperating numbers
    numbers = [[], [], [], [], [], [], [], [], [], []]
    for i in range(constants.SIZE_TRAIN):
        if y_train[i] == 0:
            numbers[0].append(x_train[i])
        elif y_train[i] == 1:
            numbers[1].append(x_train[i])
        elif y_train[i] == 2:
            numbers[2].append(x_train[i])
        elif y_train[i] == 3:
            numbers[3].append(x_train[i])
        elif y_train[i] == 4:
            numbers[4].append(x_train[i])
        elif y_train[i] == 5:
            numbers[5].append(x_train[i])
        elif y_train[i] == 6:
            numbers[6].append(x_train[i])
        elif y_train[i] == 7:
            numbers[7].append(x_train[i])
        elif y_train[i] == 8:
            numbers[8].append(x_train[i])
        elif y_train[i] == 9:
            numbers[9].append(x_train[i])

    # calculating mean of each class
    data_means = []
    for i in range(10):
        data_means.append(np.mean(numbers[i]))

    knn_object = algorithms.KNN(k=1)

    k_mean_nn_guess = []
    for test in x_test:
        # distances = []
        guess = knn_object.exe(test,data_means,y_train)
        k_mean_nn_guess.append(guess)
    #     for i, mean_point in enumerate(data_means):
    #         # calculating distance
    #         distance = np.sum(np.sqrt(np.square(test - mean_point)))
    #         distances.append([distance, i])
    #
    #     # sorting distances
    #     distances.sort()
    #     distances = distances[0]
    #     k_mean_nn_guess.append(distances[1])

    n_correct_guess_kmnn = 0
    for i, test in enumerate(y_test):
        if test == k_mean_nn_guess[i]:
            n_correct_guess_kmnn += 1
    performance = n_correct_guess_kmnn * 100 / len(y_test)
    return performance


def main():
    x_train, x_test, y_train, y_test = pre_processing()

    #################### KNN ####################
    # list [[k,performance],...]
    # knn_performances = calculate_knn_performance_with_different_k(x_train,x_test,y_train,y_test)
    # plot_knn_when_k_is_different(knn_performances)

    # knn_performances = calculate_knn_performance(x_train,x_test,y_train,y_test)
    # knn_performances = calculate_knn_performance_with_different_train_size()
    # print(knn_performances)
    # x_labels = ('500','1500','2500','3500','4500','5500','6500','7500','8500','9500')
    # y_pos = np.arange(len(x_labels))
    # performance = knn_performances
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    # plt.xticks(y_pos, x_labels)
    # plt.ylabel('Accuracy')
    # plt.title('Knn accuracy with different train size')
    # plt.show()
    #################### Parzen ####################
    # list [[r,performance],...]
    performances = calculate_parzen_performance_with_different_r(x_train,x_test,y_train,y_test)
    print(performances)
    x_labels = ([item[0] for item in performances])
    y_pos = np.arange(len(x_labels))
    performance = [item[1] for item in performances]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel('Accuracy')
    plt.title('Parzen accuracy with different k')
    plt.show()
    #################### K-Mean ####################
    # performance = k_mean_NN(x_train, x_test, y_train, y_test)
    # print(performance)

main()
