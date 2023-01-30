#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("C:/Users/HP/Downloads/suv_data.csv")


# In[3]:


data.head()


# In[4]:


#Defining dependent variable and independent Variable
X = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values


# In[5]:


X


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[8]:


from sklearn.preprocessing import StandardScaler


# In[9]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[10]:


from sklearn.linear_model import LogisticRegression


# In[11]:


Classifier = LogisticRegression(random_state = 0)
Classifier.fit(X_train,y_train)


# In[12]:


y_pred = Classifier.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score


# In[14]:


accuracy_score(y_test,y_pred)*100

