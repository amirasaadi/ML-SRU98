import numpy as np
from scipy import io
from skimage.transform import resize
from matplotlib import pyplot as plt


np.random.seed(0)

# load hoda data set
def load_hoda(training_sample_size=100, test_sample_size=20, size=5):
    # load data
    trs = training_sample_size
    tes = test_sample_size

    dataset = io.loadmat('Data_hoda_full.mat')
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


x_train,y_train,x_test,y_test = load_hoda()

x_train = np.array(x_train)

# constants
# number of clusters
k = 12

# to store distance between sampels
distance_matrix = np.full((x_train.shape[0],x_train.shape[0]),np.inf)
for i in range(x_train.shape[0]):
    for j in range(i):
        distance_matrix[i][j] = np.linalg.norm(x_train[i] - x_train[j])


while x_train.shape[0] > k :
    # find minimum
    min_value = np.min(distance_matrix)

    # find indices of min in array
    ij_min = np.where(distance_matrix == min_value) # ij_min[0] = row , ij_min[1] = col
    row_index = ij_min[0]
    column_index = ij_min[1]
    # calculate avrage of two vector
    train_1 = x_train[row_index]
    train_2 = x_train[column_index]
    avrage_vector = ((train_1 + train_2) / 2)

    # deleting from distance matrix and x_train vector
    # from x_train
    x_train = np.delete(x_train,[row_index,column_index],axis=0)

    # from distance_matrix
    distance_matrix = np.delete(distance_matrix,[row_index,column_index],axis=0) #delete(array,where,which axis)
    distance_matrix = np.delete(distance_matrix,[row_index,column_index],axis=1)

    # inserting  to x_train vector
    x_train = np.insert(x_train, 0, avrage_vector,axis=0) # insert(array,where,value)

    # calclulate distance of new train two others
    new_vector = np.full((x_train.shape[0]),np.inf)
    for i in range(1,x_train.shape[0]):
        new_vector[i] = np.linalg.norm(x_train[i] - avrage_vector)

    # insert to distance matrix
    # insert row
    new_row = np.full((1,(distance_matrix.shape[0])),np.inf)
    distance_matrix = np.insert(distance_matrix,0,new_row,axis=0)
    # insert column
    new_column = new_vector
    distance_matrix = np.insert(distance_matrix,0,new_column,axis=1)

fig, axs = plt.subplots(nrows=4,ncols=3,)
i=0
for row in axs:
    for col in row:
        col.imshow(np.reshape(x_train[i],(-1,5)))
        i+=1

plt.show()
