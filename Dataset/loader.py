import numpy as np
from scipy import io
from skimage.transform import resize

# load hoda data set
def load_hoda(training_sample_size=1000, test_sample_size=200, size=5):
    # load data
    trs = training_sample_size
    tes = test_sample_size

    dataset = io.loadmat('../Dataset/Data_hoda_full.mat')
    # loading data 1 to n
    # trains
    x_train_original = np.squeeze(dataset['Data'][:trs])
    y_train = np.squeeze(dataset['labels'][:trs])
    # tests
    x_test_original = np.squeeze(dataset['Data'][trs:trs + tes])
    y_test = np.squeeze(dataset['labels'][trs:trs + tes])
    # normalaizing data by deviding it to 255
    x_train_original /= 255
    x_test_original /= 255
    # resize
    x_train_nbyn = [resize(img, (size, size)) for img in x_train_original]
    x_test_nby_n = [resize(img, (size, size)) for img in x_test_original]
    # reshape
    x_train = [x.reshape(size * size) for x in x_train_nbyn]
    x_test = [x.reshape(size * size) for x in x_test_nby_n]
    return x_train, y_train, x_test, y_test