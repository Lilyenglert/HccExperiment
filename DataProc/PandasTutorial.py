
# coding: utf-8

# We'll need a bunch of packages for our data analysis. Let's load them first.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Let's load some sample data to play with and have a look.

# In[2]:


data = sns.load_dataset('attention')
print(data.head(5))


# In a next step, let's find out what the ranges of the data are

# In[5]:


print('Attention:', set(data['attention']))
print('Subjects:', data['subject'].unique())
print('Solutions:', data['solutions'].unique())
print('Scores:', data['score'].describe())


# We can also plot the raw data:

# In[6]:


sns.factorplot(x='solutions', y='score', hue='attention', col='subject', data=data, col_wrap=4)
plt.show()


# However, we might actually be more interested in the influence of the attention parameter. We can do just that.

# In[7]:


data.groupby('attention')['score'].describe()


# We can also plot this.

# In[8]:


sns.barplot(x='score', y='attention', data=data)
plt.show()


# You might be wondering whether there are differences between participants. Let's have a look.

# In[9]:


data.groupby(['subject', 'attention'])['score'].mean()


# As we saw in out initial plotting each participant did three "solutions". We can compute a new column to compute how much they improved from the first to the last.

# In[10]:


def compute_improvement(group):
    last_score = group[group['solutions'] == 3]['score']
    first_score = group[group['solutions'] == 1]['score']
    return pd.Series({'improvement': float(last_score) - float(first_score)})

improvement = data.groupby(['subject', 'attention']).apply(compute_improvement).reset_index()
print(improvement)


# Let's plot this.

# In[11]:


sns.boxplot(x='improvement', y='attention', data=improvement)
plt.show()


# In[12]:


improvement.describe()

