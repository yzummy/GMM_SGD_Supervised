
# coding: utf-8

# In[1]:


from sklearn import datasets
import warnings
warnings.filterwarnings(action='once')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3d
import seaborn as sns; sns.set()
import numpy as np
from sklearn.utils import shuffle
from sklearn.mixture import GMM
from sklearn.datasets import make_classification
from sklearn.datasets import make_spd_matrix
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from numpy.linalg import inv
from sklearn.utils import check_array, check_random_state
import time


# In[110]:


####################################################
#GMM Full Covariance Generator (3D)
####################################################

w = [0.04,0.01, 0.15,0.2,0.4,0.05,0.01,0.02,0.02, 0.1]
num_class = 10
num_dim = 3000
#w = [1/num_class for i in range(num_class)]
gmmMean = (np.random.rand(num_class,num_dim)-0.5)*4
gmmCov = [(make_spd_matrix(num_dim, random_state=i)*np.random.randint(20)) for i in range(num_class)]
n_samples = 200000


rng = check_random_state(0)
n_samples_comp = rng.multinomial(n_samples/20, w)
X = np.vstack([
    rng.multivariate_normal(mean, covariance, int(sample))
    for (mean, covariance, sample) in zip(
        gmmMean, gmmCov, n_samples_comp)])
Y = np.concatenate([j * np.ones(sample, dtype=int)
                   for j, sample in enumerate(n_samples_comp)])

X,Y = shuffle(X,Y,random_state=0)

fig = plt.figure()
axgmm = fig.add_subplot(111, projection='3d')
axgmm.set_xlim(left=-10,right=10)
axgmm.set_ylim(bottom=-10,top=10)
axgmm.set_zlim(bottom=-10,top=10)
axgmm.scatter(X[:,0], X[:,1], X[:,2], zdir='x', c=Y, s=40, cmap='viridis')
fig = plt.figure()
axgmm = fig.add_subplot(111, projection='3d')
axgmm.set_xlim(left=-10,right=10)
axgmm.set_ylim(bottom=-10,top=10)
axgmm.set_zlim(bottom=-10,top=10)
axgmm.scatter(X[:,0], X[:,1], X[:,2], zdir='y', c=Y, s=40, cmap='viridis')
fig = plt.figure()
axgmm = fig.add_subplot(111, projection='3d')
axgmm.set_xlim(left=-10,right=10)
axgmm.set_ylim(bottom=-10,top=10)
axgmm.set_zlim(bottom=-10,top=10)
axgmm.scatter(X[:,0], X[:,1], X[:,2], zdir='z', c=Y, s=40, cmap='viridis')


# In[ ]:


overall_num_train = [100, 200, 500, 1000, 2000, 10000]
low_num_train = [10, 20, 40, 80, 160]
num_train = low_num_train
num_trials = 5
from collections import Counter

ori_result = [0 for i in range(len(num_train))]
gmm_weighted_result = [0 for i in range(len(num_train))]

for trial in range(num_trials):
    for index, num in enumerate(num_train):
        X,Y = shuffle(X,Y,random_state=int(time.time()))

        train_X = X[:num]
        train_Y = Y[:num]
        test_X = X[-1000:]
        test_Y = Y[-1000:]

        cl = linear_model.SGDClassifier(verbose=5, class_weight=None, max_iter=30, warm_start=True)
        gmmcl = linear_model.SGDClassifier(verbose=5, class_weight=None, max_iter=30, warm_start=True)
        gmmcl_weighted = linear_model.SGDClassifier(verbose=5, class_weight=None, max_iter=30, warm_start=True)

        print("Original starting here: ")
        cl.fit(train_X, train_Y)

    #     print("GMM starting here: ")
    #     gmm = GaussianMixture(num_class, covariance_type='full', verbose = 2).fit(train_X) 
    #     gmmcl.fit(train_X, train_Y)   

        print("Weighted GMM starting here: ")
        gmm_weighted = GaussianMixture(num_class, covariance_type='full', verbose = 2).fit(train_X)    
        w = [np.dot(gmm_weighted.weights_,gmm_weighted.predict_proba([train_X[i]])[0]) for i in range(len(train_X))]
        gmmcl_weighted.fit(train_X, train_Y, sample_weight=w)


        ori_result[index] += accuracy_score(test_Y, cl.predict(test_X))
        gmm_weighted_result[index] += accuracy_score(test_Y, gmmcl_weighted.predict(test_X))
    
    
plt.plot(num_train, [x/num_trials for x in ori_result], label='Original')
plt.plot(num_train, [x/num_trials for x in gmm_weighted_result], label = 'Original+Weighted')

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')
    


# In[95]:


(gmm.predict_proba(X).max(1)**2)[0:10000]

