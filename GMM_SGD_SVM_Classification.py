
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
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# In[18]:


features = [10,100,500,1000]

for f in features:
    plt.figure(figsize=(8, 6))
    if f <= 5:
        inform = [i for i in range(f-2,f)]
    elif f <= 10:
        inform = [i for i in range(f-5,f)]
    elif f <= 100:
        inform = [i for i in range(f-50,f, 10)]
    elif f <= 1000:
        inform = [i for i in range(f-200,f,20)]
    classes = 10
    w = [1/classes for i in range(classes)]
    
    for infos in inform:
        X, Y = make_classification(n_samples=1000, 
                                   n_features = f, 
                                   n_informative = infos, 
                                   n_redundant=f-infos, 
                                   n_classes = classes, 
                                   n_clusters_per_class=1, 
                                   weights = w,
                                   class_sep = 2,
                                   random_state=0)
    
        #SGD SVM
        result = []
        n_trains = np.arange(10,200,10)
        for n_train in range(10,200,10):
            train_X = X[:n_train]
            train_Y = Y[:n_train]
            test_X = X[n_train:]
            test_Y = Y[n_train:]
            cl = linear_model.SGDClassifier()
            cl.fit(train_X, train_Y)
            pred= cl.predict(test_X)
            result.append(mean_squared_error(test_Y, pred))
        plt.plot(n_trains, result, label=infos)
        plt.title('Features='+str(f))
        plt.xlabel('Training Set Size')
        plt.ylabel('MSE')
        plt.legend(loc='best')


# In[27]:


features = 1000
infos = 1000
classes = 10
w = [1/classes for i in range(classes)]
sns.set_palette(sns.color_palette("hls", 20))
X, Y = make_classification(n_samples=1000, 
                           n_features = features, 
                           n_informative = infos, 
                           n_redundant=features-infos, 
                           n_classes = classes, 
                           n_clusters_per_class=1, 
                           weights = w,
                           class_sep = 2,
                           random_state=0)
    
#SGD SVM
org_result = []
n_trains = np.arange(20,40,10)
for n_train in range(20,40,10):
    train_X = X[:n_train]
    train_Y = Y[:n_train]
    test_X = X[n_train:]
    test_Y = Y[n_train:]
    cl = linear_model.SGDClassifier()
    cl.fit(train_X, train_Y)
    pred= cl.predict(test_X)
    org_result.append(mean_squared_error(test_Y, pred))
    
    n_components = np.arange(classes-5,classes+5)
    models = [GMM(n, covariance_type='full', random_state=0).fit(train_X) for n in n_components]

    n_sample = 500


    plt.figure(figsize=(8, 6))
    for k in range(len(n_components)):
        gmm_res = []
        for n in range(100,n_sample,100):
            samples_X = models[k].sample(n, random_state=0)
            samples_Y = models[k].predict(samples_X)
            cl = linear_model.SGDClassifier(n_jobs=2)
            cl.fit(samples_X, samples_Y)
            pred= cl.predict(test_X)
            gmm_res.append(mean_squared_error(test_Y, pred))
        plt.plot(np.arange(100,n_sample,100), gmm_res, label=str(n_components[k]))
    plt.title('Original Training Size='+str(n_train))
    plt.xlabel('Sampling Size')
    plt.ylabel('MSE')
    #plt.legend(loc='best')
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
#plt.plot(n_trains, result, label=infos)


# In[ ]:


f = 1000
plt.figure(figsize=(8, 6))
infos = 1000
classes = 10
w = [1/classes for i in range(classes)]
num_trials = 10
n_trains = np.arange(10,200,10)
result = [0 for i in range(len(n_trains))]
gmm_result = [0 for i in range(len(n_trains))]
for trial in range(num_trials):
    X, Y = make_classification(n_samples=1000, 
                               n_features = f, 
                               n_informative = infos, 
                               n_redundant=f-infos, 
                               n_classes = classes, 
                               n_clusters_per_class=1, 
                               weights = w,
                               class_sep = 2,
                               random_state=0)

    #SGD SVM

    for n_train in range(len(n_trains)):
        train_X = X[:n_trains[n_train]]
        train_Y = Y[:n_trains[n_train]]
        test_X = X[n_trains[n_train]:]
        test_Y = Y[n_trains[n_train]:]
        cl = linear_model.SGDClassifier()
        cl.fit(train_X, train_Y)
        pred= cl.predict(test_X)
        result[n_train] += (mean_squared_error(test_Y, pred))

        gmm = GMM(10, covariance_type='full', random_state=0).fit(train_X)
        samples_X = gmm.sample(8, random_state=0)
        samples_Y = gmm.predict(samples_X)
        cl = linear_model.SGDClassifier(n_jobs=2)
        cl.fit(samples_X, samples_Y)
        pred= cl.predict(test_X)
        gmm_result[n_train] += (mean_squared_error(test_Y, pred))
    plt.plot(n_trains, result/trial, label='Original')
    plt.plot(n_trains, gmm_result/trial, label = 'Original+Sampled')
    plt.title('Features='+str(f))
    plt.xlabel('Training Set Size')
    plt.ylabel('MSE')
    plt.legend(loc='best')


# In[65]:


fig = plt.figure()
newax = fig.add_subplot(111, projection='3d')
newax.scatter(X[:,0], X[:,1], X[:,2], zdir='z', c=Y, s=40, cmap='viridis')

