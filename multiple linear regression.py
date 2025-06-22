#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\Sanjana\Downloads\economic_index.csv")


# In[4]:


df.head()


# In[19]:


df=df.drop(columns=['year','month'],axis=1)


# In[20]:


df.info()


# In[21]:


df.describe()


# In[22]:


plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
sns.boxplot(df['interest_rate'])
plt.subplot(1,3,2)
sns.boxplot(df['unemployment_rate'])
plt.subplot(1,3,3)
sns.boxplot(df['index_price'])


# In[23]:


cor=df.corr()
sns.heatmap(cor,annot=True)


# In[24]:


sns.pairplot(data=df)


# In[28]:


y=df['index_price']
np.array(y).shape


# In[31]:


X=df[['interest_rate','unemployment_rate']]
np.array(X).shape


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[35]:


from sklearn.preprocessing import StandardScaler


# In[37]:


scaler=StandardScaler()


# In[38]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[40]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)


# In[41]:


from sklearn.model_selection import cross_val_score


# In[42]:


validation_score=cross_val_score(reg,X_train,y_train,scoring='neg_mean_squared_error',cv=3)


# In[43]:


np.mean(validation_score)


# In[44]:


y_pred=reg.predict(X_test)


# In[45]:


y_pred


# In[52]:


reg.coef_


# In[47]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse,mae,rmse)


# In[48]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
score


# In[49]:


1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# In[ ]:




