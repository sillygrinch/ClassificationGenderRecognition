#!/usr/bin/env python
# coding: utf-8

# #  Binary Classification Problem 
# 
# ### Problem 1: Gender Recognition by Voice
# 
# From the description file at https://data.world/ml-research/gender-recognition-by-voice:
# 
# In order to analyze gender by voice and speech, a training database was required. A database was built using thousands of samples of male and female voices, each labeled by their gender of male or female. Voice samples were collected from the following resources:
# 
# *  [The Harvard-Haskins Database of Regularly-Timed Speech](http://nsi.wegall.net/)
# *  Telecommunications & Signal Processing Laboratory (TSP) Speech Database at McGill University
# *  [VoxForge Speech Corpus](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/8kHz_16bit/)
# *  [Festvox CMU_ARCTIC Speech Database at Carnegie Mellon University](http://festvox.org/cmu_arctic/dbs_awb.html)
# 
# Each voice sample is stored as a .WAV file, which is then pre-processed for acoustic analysis using the specan function from the WarbleR R package. Specan measures 22 acoustic parameters on acoustic signals for which the start and end times are provided.
# 
# The output from the pre-processed WAV files were saved into a CSV file, containing 3168 rows and 21 columns (20 columns for each feature and one label column for the classification of male or female). You can download the pre-processed dataset in CSV format, using the link above
# Acoustic Properties Measured
# 
# The following acoustic properties of each voice are measured:
# 
# *    __duration:__ length of signal
# *    __meanfreq:__ mean frequency (in kHz)
# *    __sd:__ standard deviation of frequency
# *    __median:__ median frequency (in kHz)
# *    __Q25:__ first quantile (in kHz)
# *    __Q75:__ third quantile (in kHz)
# *    __IQR:__ interquantile range (in kHz)
# *    __skew:__ skewness (see note in specprop description)
# *    __kurt:__ kurtosis (see note in specprop description)
# *    __sp.ent:__ spectral entropy
# *    __sfm:__ spectral flatness
# *    __mode:__ mode frequency
# *    __centroid:__ frequency centroid (see specprop)
# *    __peakf:__ peak frequency (frequency with highest energy)
# *    __meanfun:__ average of fundamental frequency measured across acoustic signal
# *    __minfun:__ minimum fundamental frequency measured across acoustic signal
# *    __maxfun:__ maximum fundamental frequency measured across acoustic signal
# *    __meandom:__ average of dominant frequency measured across acoustic signal
# *    __mindom:__ minimum of dominant frequency measured across acoustic signal
# *    __maxdom:__ maximum of dominant frequency measured across acoustic signal
# *    __dfrange:__ range of dominant frequency measured across acoustic signal
# *    __modindx:__ modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
# 
# The gender of the speaker is given in the __label__ column. 
# 
# Note, the features for duration and peak frequency (peakf) were removed from training. Duration refers to the length of the recording, which for training, is cut off at 20 seconds. Peakf was omitted from calculation due to time and CPU constraints in calculating the value. In this case, all records will have the same value for duration (20) and peak frequency (0).
# 
# Load file using the code below. 
# 
# ### Questions:
# 
# 1. Preform Logistic Regression on the two features  "meanfun" and "IQR" and draw the boundary. 
# 
# 2. graphing the resulting fits. How does the two feature fit compare to the fit on all features?
# 
# 3. Preform Logistic Regression on all the features and print the socre.

# In[3]:


import pandas as pd

data = pd.read_csv("voice.csv")

data.head()


# In[4]:


names = list(data)
print(names)


# In[5]:


X = data.drop(columns=["label"]) 
y = data["label"] 
X = (X - X.mean()) / X.std()  # standarize the data


# In[6]:


y_train = pd.get_dummies(y) 
X_train = X[['meanfun',"IQR"]] 


# In[7]:


y.value_counts()


# In[15]:


from matplotlib import pyplot as plt
f, ax = plt.subplots(figsize=(5,5))

I_m = y=="female"
I_b = y=="male"

plt.plot(X["meanfun"][I_m],X["IQR"][I_m],'o',label="female")

## We set alpha=.5 to try to avoid masking, but some points still will be burried. 
plt.plot(X["meanfun"][I_b],X["IQR"][I_b],'o',label="male",alpha=.5)

plt.xlabel("meanfun",fontsize=20)
plt.ylabel("IQR",fontsize=20)
plt.legend(fontsize=15)


# In[16]:


X_train = X[['meanfun',"IQR"]]


# In[17]:


y


# In[18]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train,y)

print("Score: %.3f"%clf.score(X_train,y))


# In[21]:


import numpy as np
f, ax = plt.subplots(figsize=(8,8))

X1 = X["meanfun"]
X2 = X["IQR"]

plt.plot(X1[I_m],X2[I_m],'o',label="female")
plt.plot(X1[I_b],X2[I_b],'o',label="male",alpha=.5)

## As before we generate a meshgrid, but now we use qda.predict to guess at the label. 

xm,xM = plt.xlim()
ym,yM = plt.ylim()

XX, YY = np.meshgrid(np.linspace(xm,xM, 100),np.linspace(ym,yM, 100)) 

## We now form a 10000x2 array of the (x,y) coordiantes for each point by reshaping
## the XX and YY matricies and pasting them together. We need to feed a Nx2 vector
## into the qda.predict function, otherwise it will think we have too many features.
## We can reshape it later to get our grid back

grid=np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],axis=1)

ZZ = clf.predict(grid).reshape(XX.shape)  ## We predict, and reshape back to the origional grid

z1 = ZZ == 'female'
z2 = ZZ == 'male'
plt.plot(XX[z1],YY[z1],',',color="C0")
plt.plot(XX[z2],YY[z2],',',color="C1")

B0 = clf.intercept_
B = clf.coef_

u = np.linspace(xm,xM, 2)
v = -(u*(B[0,0]) + B0[0])/(B[0,1])
plt.plot(u,v,label="Decision Boundary",color="black")

   

## We now reset the x and y limits to make sure our view is centered tightly
## around the data. 

plt.xlabel("meanfun",fontsize=20)
plt.ylabel("IQR",fontsize=20)
plt.legend(fontsize=15)

ax.set_xlim([xm, xM])
ax.set_ylim([ym, yM])


# In[22]:


X.head()


# In[23]:


clf = LogisticRegression()
clf.fit(X,y)

print("Score: %.3f"%clf.score(X,y))


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

## Logisitic Regression

clf = LogisticRegression()
clf.fit(X_train,y_train)
print("Logistic Regression Score: %.3f"%clf.score(X_test,y_test))


# In[25]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold


# In[26]:


crossvalidation = KFold(n_splits=10, random_state=1, shuffle=True)

clf = LogisticRegression()
X_current = X
model = clf.fit(X_current, y)
scores = cross_val_score(model, X_current, y, cv=crossvalidation,n_jobs=1)
np.mean(np.abs(scores))


# In[27]:


np.abs(scores)


# In[28]:


from sklearn.preprocessing import PolynomialFeatures


# In[34]:


crossvalidation = KFold(n_splits=10, random_state=1, shuffle=True)

clf = LogisticRegression()

for i in range(1,2):
    poly = PolynomialFeatures(degree=i)
    X_current = poly.fit_transform(X)
    model = clf.fit(X_current, y)
    scores = cross_val_score(model, X_current, y, cv=crossvalidation,
 n_jobs=1)
    
    print("Degree-"+str(i)+" polynomial Score: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))


# In[ ]:





# In[ ]:




