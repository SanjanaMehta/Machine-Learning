#!/usr/bin/env python
# coding: utf-8

# ### Import Dataset and Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\Sanjana\Downloads\Algerian_forest_fires_dataset.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# ### Basic Data Cleaning

# In[5]:


df[df.isnull().any(axis=1)]


# In[6]:


df=df.drop(122)


# In[7]:


df.loc[:123,"Region"]=0
df.loc[123:,"Region"]=1


# In[8]:


df['Region']=df['Region'].astype(int)


# In[9]:


df.isnull().sum()


# In[10]:


df=df.dropna().reset_index(drop=True)


# In[11]:


df.iloc[[122]]


# In[12]:


df=df.drop(122).reset_index(drop=True)


# In[13]:


df.iloc[[122]]


# In[14]:


df.columns=df.columns.str.strip()
df.columns


# In[15]:


df[['day','month','year','Temperature','RH','Ws']]=df[['day','month','year','Temperature','RH','Ws']].astype(int)


# In[16]:


df[['Rain','FFMC','DMC','DC','ISI','BUI','FWI']]=df[['Rain','FFMC','DMC','DC','ISI','BUI','FWI']].astype(float)


# In[17]:


df.info()


# In[18]:


df.to_csv("Algerian FF cleaned.csv",index=False)


# ### Exploratory Data Analysis 

# In[19]:


df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)


# In[20]:


df['Classes'].value_counts()


# In[21]:


cols=df[['day','month','year']]
df1=df.drop(cols,axis=1)


# In[22]:


df1


# In[23]:


sns.countplot(df1['Classes'])


# In[24]:


df1.hist(bins=20,figsize=(20,20))


# In[25]:


df1.boxplot(figsize=(20,20))


# In[26]:


plt.figure(figsize=(20,15))
cor=df1.corr()
sns.heatmap(cor,annot=True)


# In[27]:


sns.pairplot(df1)


# In[40]:


dftemp=df.loc[df['Region']==1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x="month",hue="Classes",data=df)
plt.title("Monthly fire analysis of Sidi-Bel Abbes Region")
plt.show()


# In[41]:


dftemp=df.loc[df['Region']==0]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x="month",hue="Classes",data=df)
plt.title("Monthly fire analysis of Bejaia Region")
plt.show()


# In[ ]:




