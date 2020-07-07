#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# ### Creating a Series using Pandas
# 
# You could convert a list,numpy array, or dictionary to a Series in the following manner

# In[2]:


labels = ['w','x','y','z']
list = [10,20,30,40]
array = np.array([10,20,30,40])
dict = {'w':10,'x':20,'y':30,'z':40}


# In[3]:


pd.Series(data=list)


# In[4]:


pd.Series(data=list,index=labels)


# In[5]:


pd.Series(list,labels)


# ** Using NumPy Arrays to create Series **

# In[6]:


pd.Series(array)


# In[7]:


pd.Series(array,labels)


# ** Using Dictionary to create series **

# In[9]:


pd.Series(dict)


# ## Using an Index

# In[10]:


sports1 = pd.Series([1,2,3,4],index = ['Cricket', 'Football','Basketball', 'Golf'])                                   


# In[11]:


sports1


# # DataFrames
# 
# DataFrames concept in python is similar to that of R programming language. DataFrame is a collection of Series combined together to share the same index positions.

# In[14]:


from numpy.random import randn


# In[15]:


dataframe = pd.DataFrame(randn(10,5),index='A B C D E F G H I J'.split(),columns='Score1 Score2 Score3 Score4 Score5'.split())


# In[16]:


dataframe


# ## Selection and Indexing
# 
# Ways in which we can grab data from a DataFrame

# In[17]:


dataframe['Score3']


# In[18]:


# Pass a list of column names in any order necessary
dataframe[['Score2','Score1']]


# **Adding a new column to the DataFrame**

# In[25]:


dataframe['Score6']=dataframe['Score1'] + dataframe['Score2'] 


# In[26]:


dataframe


# ** Removing Columns from DataFrame**

# In[21]:


dataframe.drop('Score6',axis=1)              # Use axis=0 for dropping rows and axis=1 for dropping columns


# In[22]:


# column is not dropped unless inplace input is TRUE
dataframe


# In[27]:


dataframe.drop('Score6',axis=1,inplace=True)
dataframe


# In[28]:


#Dropping rows using axis=0
dataframe.drop('A',axis=0)     
# Row will also be dropped only if inplace=TRUE is given as input


# ** Selecting Rows**

# In[29]:


dataframe.loc['F']


# In[31]:


dataframe.iloc[2]


# In[32]:


dataframe.loc['A','Score1']


# In[33]:


dataframe.loc[['A','B'],['Score1','Score2']]


# ### Conditional Selection
# 
# Similar to NumPy, we can make conditional selections using Brackets

# In[34]:


dataframe>0.5


# In[35]:


dataframe[dataframe>0.5]


# In[36]:


dataframe[dataframe['Score1']>0.5]


# # Missing Data
# 
# Methods to deal with missing data in Pandas

# In[37]:


dataframe = pd.DataFrame({'Cricket':[1,2,np.nan,4,6,7,2,np.nan],
                  'Baseball':[5,np.nan,np.nan,5,7,2,4,5],
                  'Tennis':[1,2,3,4,5,6,7,8]})


# In[38]:


dataframe


# In[39]:


dataframe.dropna()


# In[40]:


dataframe.dropna(axis=1)       # Use axis=1 for dropping columns with nan values


# In[41]:


dataframe.dropna(thresh=2)


# In[42]:


dataframe.fillna(value=0)


# In[ ]:





# # Data Input and Output
# 
# Reading DataFrames from external sources using pd.read functions

# In[3]:


dataframe = pd.read_csv('pandas-train.csv')


# In[4]:


dataframe.to_csv('train2.csv',index=False) 


# In[ ]:




