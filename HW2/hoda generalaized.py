#!/usr/bin/env python
# coding: utf-8

# In[78]:


from scipy import io
import numpy as np
from skimage.transform import resize
from matplotlib import pyplot as plt
import winsound
np.random.seed(0)


# In[79]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# In[80]:


# load hoda data set
def load_hoda(training_sample_size=4000, test_sample_size=500, size=10):
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


# In[81]:
def function(training_sample_size=1000, test_sample_size=200, size=2):
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

    x_train_nbyn = [resize(img, (5, 5)) for img in x_train_nbyn]
    x_test_nby_n = [resize(img, (5, 5)) for img in x_test_nby_n]

    # reshape
    x_train = [x.reshape(5 * 5) for x in x_train_nbyn]
    x_test = [x.reshape(5 * 5) for x in x_test_nby_n]
    return x_train, y_train, x_test, y_test

train_size = 1000
test_size = 200
figure_size = 5 
x_train,y_train,x_test,y_test = function()

x = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

#transform output to one hot encoding
t = np.zeros((train_size,10))
for i in range(len(t)):
    t[i][y_train[i]] = 1


# In[82]:


epochs = 700
lr = 0.0103


# In[88]:


# inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 25, 29, 10
# n = [inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons]
n = [25,29,10]


# In[89]:


# Random weights and bias initialization
b = []
w = []
for i in range(len(n) - 1):
    w.append(np.random.randn(n[i], n[i + 1]))
    b.append(np.random.randn(1, n[i + 1]))


# In[90]:


costs = []
for _ in range(epochs):

    # feed forward
    y = [x]
    for i in range(len(w)):
        y.append(sigmoid((y[i] @ w[i]) + b[i]))
    
    # back prob
    e = t - y[-1]
    cost = 0.5 * np.sum((e) ** 2)
    costs.append(np.mean(cost))
    dw_o = e * sigmoid_derivative(y[-1])
    dw = [[] for _ in range(len(w))]
    dw[-1] = dw_o
    for i in range(len(w) - 1, 0, -1):
        error_hidden = np.dot(dw[i], w[i].T)
        dw[i - 1] = error_hidden * sigmoid_derivative(y[i])
    
    # updating
    for i in range(len(w) - 1, -1, -1):
        w[i] += (y[i].T @ dw[i] * lr)
        b[i] += (np.sum(dw[i], axis=0, keepdims=True) * lr)


# In[91]:


# plt.plot(costs)
# plt.show()


# In[92]:


# Forward Propagation for test
y = [x_test]
for i in range(len(w)):
    y.append(sigmoid((y[i] @ w[i]) + b[i]))

# finding maximum at the output of NN
javabha = []
for i in range(len(y_test)):
    javabha.append(np.argmax(y[-1][i]))

# count number of true predicted
dorost = 0
for i in range(len(y_test)):
    if y_test[i] == javabha[i]:
        dorost+=1

print('accuracy is = ',dorost*100/len(y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




