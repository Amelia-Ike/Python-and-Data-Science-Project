#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Capstone Project

# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# Just go along with this notebook and try to complete the instructions or answer the questions in bold using your Python and Data Science skills!

# ## Data and Setup

# ____
# ** Import numpy and pandas **

# In[1]:


import numpy as np
import pandas as pd


# ** Import visualization libraries and set %matplotlib inline. **

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Read in the csv file as a dataframe called df **

# In[3]:


df = pd.read_csv('911.csv')


# ** Check the info() of the df **

# In[4]:


df.info()


# ** Check the head of df **

# In[5]:


df.head()


# ## Basic Questions

# ** What are the top 5 zipcodes for 911 calls? **

# In[5]:


df['zip'].value_counts().head()


# ** What are the top 5 townships (twp) for 911 calls? **

# In[6]:


df['twp'].value_counts().head()


# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[7]:


df['title'].nunique()


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[8]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# **OR**

# In[9]:


df_title_split = df['title'].str.split(':').str[0]
df_reason = df.insert(loc=1, column='Reason', value=df_title_split)


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[10]:


df['Reason'].value_counts()


# In[ ]:





# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[11]:


sns.countplot(x='Reason', data=df, palette="dark")#try palette viridis


# In[ ]:





# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[12]:


type(df['timeStamp'].iloc[0])


# In[ ]:





# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[13]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# **OR**

# In[12]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'], format='%Y-%m-%d %H:%M:%S')


# In[14]:


time = df['timeStamp'].iloc[24]
time.hour


# ** You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# In[12]:


df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)


# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[13]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap, na_action=None)


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[17]:


sns.countplot(x='Day of Week', data=df, hue='Reason', palette='dark')#try palette viridis


# In[18]:


sns.countplot(x='Day of Week', data=df, hue='Reason', palette='dark')#try palette viridis
#to relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:





# **Now do the same for Month:**

# In[17]:


sns.countplot(x='Month', data=df, hue='Reason', palette='dark')


# In[19]:


sns.countplot(x='Month', data=df, hue='Reason', palette='dark')
#to relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:





# **Did you notice something strange about the Plot?**
# 
# _____
# 
# ** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas... **

# ** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# In[21]:


byMonth = df.groupby('Month').count()
byMonth.head()


# In[153]:


type(byMonth)


# In[ ]:





# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[22]:


#could be any column
byMonth['twp'].plot()


# In[19]:


sns.lineplot(x=byMonth.index, y='twp', data=byMonth)


# In[ ]:





# ** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[20]:


#sns.lmplot(x='total_bill', y='tip', row='sex', col='time', data=tips)
byMonth = byMonth.reset_index()
byMonth.head()


# In[21]:


sns.lmplot(x='Month', y='twp', data=byMonth)


# In[ ]:





# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[36]:


df['Date'] = df['timeStamp'].apply(lambda x: x.date().strftime('%b-%d-%Y'))
df.head()


# In[ ]:





# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[37]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# In[ ]:





# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[38]:


reason_grp = df.groupby('Reason').count()
reason_grp


# In[39]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[41]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:





# In[42]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# In[ ]:





# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[49]:


df_hour = df.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()
df_hour


# In[81]:


df_hour = df.groupby(['Hour','Day of Week']).count()
df_hour.unstack(level=0).head()


# In[ ]:





# ** Now create a HeatMap using this new DataFrame. **

# In[45]:


plt.figure(figsize=(12,6))
sns.heatmap(df_hour, cmap='coolwarm')


# In[ ]:





# ** Now create a clustermap using this DataFrame. **

# In[44]:


sns.clustermap(df_hour, cmap='viridis')


# In[ ]:





# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[50]:


df_Month = df.groupby(by=['Day of Week', 'Month']).count()['Reason'].unstack()
df_Month


# In[ ]:





# In[51]:


plt.figure(figsize=(12,6))
sns.heatmap(df_Month, cmap='coolwarm')


# In[ ]:





# In[52]:


sns.clustermap(df_Month, cmap='viridis')


# In[ ]:





# **Continue exploring the Data however you see fit!**
# # Great Job!
