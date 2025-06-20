#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns


# In[1]:


import pandas as pd
df=pd.read_csv(r"C:\Users\Sanjana\Downloads\height-weight.csv")


# ### Import Dataset

# In[4]:


df.head()


# In[5]:


df.corr()


# ### Plot relation b/w variables

# In[8]:


plt.scatter(data=df,x='Weight',y='Height')
plt.xlabel("Weight")
plt.ylabel("Height")


# In[10]:


sns.pairplot(df)


# In[29]:


X=df[['Weight']]
X.head()
np.array(X).shape


# In[30]:


y=df['Height']


# In[31]:


np.array(y).shape


# ### Perform Data Splitting

# In[34]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[39]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)


# In[40]:


X_test=scaler.transform(X_test)


# ### Perform Linear Regression

# In[42]:


from sklearn.linear_model import LinearRegression


# In[43]:


reg=LinearRegression()


# In[44]:


reg.fit(X_train,y_train)


# ### Linear Regression Coefficient and Intercept

# In[46]:


reg.coef_


# In[47]:


reg.intercept_


# In[50]:


plt.scatter(X_train,y_train)
plt.plot(X_train,reg.predict(X_train))


# In[52]:


y_pred=reg.predict(X_test)


# ### Evaluating Model

# In[55]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[57]:


mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)


# In[58]:


print(mse,mae,rmse)


# In[59]:


from sklearn.metrics import r2_score


# In[61]:


score=r2_score(y_test,y_pred)
score


# In[64]:


1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# In[65]:


import statsmodels.api as sm


# In[66]:


model=sm.OLS(y_train,X_train).fit()


# In[67]:


model.predict(X_test)


# In[68]:


model.summary()


# ### Prediction on model

# In[70]:


reg.predict(scaler.transform([[72]]))

