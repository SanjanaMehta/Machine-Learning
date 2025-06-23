#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Algerian FF cleaned.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


cols=df[['day','month','year']]
df1=df.drop(cols,axis=1)


# In[6]:


df1['Classes']=np.where(df1['Classes'].str.contains('not fire'),0,1)


# In[7]:


X=df1.drop('FWI',axis=1)
y=df1['FWI']


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[9]:


plt.figure(figsize=(20,20))
cor=X_train.corr()
sns.heatmap(cor,annot=True)


# In[10]:


def correlation(df1,threshold):
    col_corr=set()
    corr_matrix=df1.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range (i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[11]:


cf=correlation(X_train,0.85)


# In[12]:


X_train.drop(cf,axis=1,inplace=True)
X_test.drop(cf,axis=1,inplace=True)


# In[13]:


X_train.shape,X_test.shape


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score


# In[16]:


scaler=StandardScaler()


# In[17]:


X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)


# In[18]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[19]:


reg.fit(X_train,y_train)


# In[20]:


y_pred=reg.predict(X_test)


# In[21]:


mae=mean_absolute_error(y_test,y_pred)


# In[22]:


score=r2_score(y_test,y_pred)


# In[23]:


mae,score


# In[24]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score
lasso=Lasso()
lasso.fit(X_train_s,y_train)
y_pred=lasso.predict(X_test_s)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
mae,score


# In[25]:


plt.scatter(y_test,y_pred)


# In[26]:


from sklearn.linear_model import LassoCV
lassocv=LassoCV(cv=5)
lassocv.fit(X_train_s,y_train)


# In[27]:


lassocv.alpha_


# In[28]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score
ridge=Ridge()
ridge.fit(X_train_s,y_train)
y_pred=ridge.predict(X_test_s)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
mae,score


# In[29]:


plt.scatter(y_test,y_pred)


# In[34]:


from sklearn.linear_model import RidgeCV
ridgecv=RidgeCV(cv=5)
ridgecv.fit(X_train_s,y_train)
y_pred=ridgecv.predict(X_test_s)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
mae,score


# In[35]:


plt.scatter(y_test,y_pred)


# In[30]:


from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score
elastic=ElasticNet()
elastic.fit(X_train_s,y_train)
y_pred=elastic.predict(X_test_s)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
mae,score


# In[36]:


from sklearn.linear_model import ElasticNetCV
elasticcv=ElasticNetCV(cv=5)
elasticcv.fit(X_train_s,y_train)
y_pred=elasticcv.predict(X_test_s)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
mae,score


# In[37]:


plt.scatter(y_test,y_pred)


# In[ ]:




