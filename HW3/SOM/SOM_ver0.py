#!/usr/bin/env python
# coding: utf-8

# ### libraries

# In[1]:


from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches


import pandas as pd
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy.optimize as opt
from sklearn.decomposition import PCA

get_ipython().run_line_magic('matplotlib', 'inline')


# ### loading HODA

# In[2]:


# load hoda data set
def load_hoda(training_sample_size=1000, test_sample_size=200, size=5):
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


# ## constants

# In[3]:



train_size = 100
test_size = 20
figure_size = 5 
x_train,y_train,x_test,y_test = load_hoda(train_size,test_size,figure_size)
raw_data = np.asarray(x_train).T
output_size = 5


# In[4]:


network_dimensions = np.array([output_size, output_size])
n_iterations = 10000
init_learning_rate = 0.01

normalise_data = True
normalise_by_column = False


# In[5]:


m = raw_data.shape[0]
n = raw_data.shape[1]

# initial neighbourhood radius
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
# radius decay parameter
time_constant = n_iterations / np.log(init_radius)

data = raw_data
if normalise_data:
    if normalise_by_column:
        col_maxes = raw_data.max(axis=0)
        data = raw_data / col_maxes[np.newaxis, :]
    else:
        data = raw_data / data.max()


# In[6]:


net = np.random.random((network_dimensions[0], network_dimensions[1], m))


# ## functions

# In[7]:


# find winner
def find_bmu(t, net, m):
    """
        Find the best matching unit for a given vector, t
        Returns: bmu and bmu_idx is the index of this vector in the SOM
    """
    bmu_idx = np.array([0, 0])
    min_dist = np.iinfo(np.int).max
    
    # calculate the distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            sq_dist = np.sum((w - t) ** 2)
            sq_dist = np.sqrt(sq_dist)
            if sq_dist < min_dist:
                min_dist = sq_dist # dist
                bmu_idx = np.array([x, y]) # id
    
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    return (bmu, bmu_idx)


# In[8]:


# sigma
def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

# learning rate
def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

# hamsayegi
def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))


# ### learning phase

# In[9]:


for i in range(n_iterations):
    # select a training example at random
    t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))
    
    # find its Best Matching Unit
    bmu, bmu_idx = find_bmu(t, net, m)
    
    # decay the SOM parameters
    r = decay_radius(init_radius, i, time_constant)
    l = decay_learning_rate(init_learning_rate, i, n_iterations)
    
    # update weight vector to move closer to input
    # and move its neighbours in 2-D vector space closer
    
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
            w_dist = np.sqrt(w_dist)
            
            if w_dist <= r:
                # calculate the degree of influence (based on the 2-D distance)
                influence = calculate_influence(w_dist, r)
                
                # new w = old w + (learning rate * influence * delta)
                # where delta = input vector (t) - old w
                new_w = w + (l * influence * (t - w))
                net[x, y, :] = new_w.reshape(1, 25)


# In[10]:


z = np.zeros([output_size**2,len(x_train[0])])
k=0
for i in range(output_size):
    for j in range(output_size):
#         print(net[i][j])
        z[k] = net[i][j]
        k+=1


# In[11]:


# PCA part
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(z)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2', 'principal component 3'])


# In[12]:


z= np.asarray(principalDf)


# In[13]:


matris = np.zeros((output_size,output_size,3))
k = 0
for i in range(output_size):
    for j in range(output_size):
        matris[i][j][:] = z[k]
        k+=1
        


# In[14]:



le = np.max(matris) - np.min(matris)
matris = matris + np.abs(np.min(matris)) 
matris = matris / le


# In[15]:


fig = plt.figure()

ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, net.shape[0]+1))
ax.set_ylim((0, net.shape[1]+1))
ax.set_title('Self-Organising Map after %d iterations' % n_iterations)

# plot
for x in range(1, net.shape[0] + 1):
    for y in range(1, net.shape[1] + 1):
        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     facecolor=matris[x-1,y-1,:],
                     edgecolor='none'))

plt.show()

