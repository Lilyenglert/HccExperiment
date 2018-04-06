
# coding: utf-8

# Let's start again by loading some packages:

# In[1]:


import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import multipletests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Sampling

# In any experiment we only look at a sample from a much larger population. We don't get 5 billion people to test our new keyboard, we maybe only have 10 people take the experiment. But what happens when we sample? Let's simulate and find out. We can use the height distribution of people born in Denmark in 1979, as reported in ''The evolution of adult height in Europe: a brief note''.

# In[2]:


male = dict(mean=185.4, std=6.8)
female = dict(mean=167.5, std=4.3)

height = np.linspace(140, 220, 100)
male_pdf = [stats.norm.pdf(x, male['mean'], male['std']) for x in height]
female_pdf = [stats.norm.pdf(x, female['mean'], female['std']) for x in height]
plt.plot(height, male_pdf)
plt.plot(height, female_pdf)
plt.show()


# Overall, we see that there is a clear difference for male and female heights. But what happens if we only look at a more limited sample, say 10 males and 10 females?

# In[3]:


male_sample = np.random.normal(male['mean'], male['std'], 10)
female_sample = np.random.normal(female['mean'], female['std'], 10)
plt.plot(height, male_pdf)
plt.vlines(np.mean(male_sample), 0, 0.03)
plt.plot(height, female_pdf)
plt.vlines(np.mean(female_sample), 0, 0.03)
plt.show()


# We do see a difference, but we also see that the mean of the sample isn't quite the actual mean. We can compare the difference and deviation directly.

# In[4]:


male_error = np.mean(male_sample) - male['mean']
female_error = np.mean(female_sample) - female['mean']
difference = np.mean(male_sample) - np.mean(female_sample)
diff_error = difference - (male['mean'] - female['mean'])
print(male_error, female_error, difference, diff_error)


# How does sample size relate to the error? We can simulate this by drawing a large number of samples and then looking at how the error is distributed. Let's try this with the male heights.

# In[6]:


def simulate_sample_error(n):
    sample = np.random.normal(male['mean'], male['std'], n)
    return np.mean(sample) - male['mean']

simulation = [{'n': n, 'error': simulate_sample_error(n)} for t in range(100) for n in [1, 5, 10, 50, 100, 500, 1000]]
simulation = pd.DataFrame(simulation)

sns.pointplot(x='n', y='error', data=simulation)
plt.show()


# As the plot shows, the more samples we have, the closer we are to the "true" mean. Let's assume we only have 5 male and 5 female participants. How does this sampling issue extent to the height differences between males and females?

# In[6]:


def simulate_sample_error(n):
    male_sample = np.random.normal(male['mean'], male['std'], n)
    female_sample = np.random.normal(female['mean'], female['std'], n)
    return np.mean(male_sample) - np.mean(female_sample)

simulation = [simulate_sample_error(5) for t in range(1000)]
sns.distplot(simulation)
plt.show()

print(np.min(simulation), np.max(simulation), np.mean(simulation))


# Depending on our sample, the height difference we find varies drastically. Based on the dataset used above, we would expect a difference of 17.9cm between males and females. Our average difference is actually very close to that. But in some experiments, we find differences less than half that or much larger than that. In the case of height, it might be obvious to us when we are running one of those outlier experiments with, for example, only very tall female participants. But what if what we measure isn't as immediately obvious? For example, consider that you compare two interfaces. You would likely not have an easy way to judge whether a participant would excell with the device or not just by looking at them.

# # Significance testing
# ## T-Test

# For our look into stats we want to look at analysis of experimental data. Instead of actual data, we can use generated data. This also allows us to simulate a large number of experiments without running all those participants. Let's define a function that generates experimental data:

# In[7]:


def run_experiment(participants, conditions):
    data = {'participant': np.arange(1, participants + 1)}
    for cname, cdist, params in conditions:
        data[cname] = cdist(*params, size=participants)
    
    data = pd.DataFrame(data)
    return data.melt(id_vars='participant', var_name='condition', value_name='measurement')


# Here's an example experiment: we have 10 participants try both a mouse and a keyboard interface. Let's assume they are worse when using the keyboard interface.

# In[8]:


conditions = [('mouse', np.random.lognormal, (2.0, 0.5)), ('keyboard', np.random.lognormal, (3.0, 0.5))]    
data = run_experiment(10, conditions)

print(data)

sns.barplot(x='condition', y='measurement', data=data)
plt.show()


# We can run a t-test to directly compare the two groups and see whether we can indeed detect a statistical difference (as we should).

# In[9]:


mouse = data[data['condition'] == 'mouse']['measurement']
keyboard = data[data['condition'] == 'keyboard']['measurement']
stats.ttest_rel(mouse, keyboard)


# We indeed find a significant result with a p-value < 0.05. Great! What if we run that same experiment many times? Do the results hold?

# In[10]:


def analyze_experiment(data):
    groups = data.groupby('condition')['measurement']
    groups = [list(g) for _, g in groups]
    return stats.ttest_rel(*groups).pvalue

runs = [analyze_experiment(run_experiment(10, conditions)) for x in range(100)]
sns.boxplot(runs)
plt.show()


# Most of our experiments found a significant difference, but not all of them! Let's look at those a bit closer.

# In[11]:


significant = len([x for x in runs if x < 0.05])
non_significant = len([x for x in runs if x >= 0.05])
print(significant, "significant results")
print(non_significant, "non-significant results")
print(100 * significant / (significant + non_significant), "percent of results are significant")


# In the previous setup there actually was a difference between the conditions. But what happens if there is none?

# In[12]:


conditions = [('male', np.random.normal, (5.0, 1.0)), ('female', np.random.normal, (5.0, 1.0))]    
runs = [analyze_experiment(run_experiment(10, conditions)) for x in range(1000)]
sns.boxplot(runs)
plt.show()


# Most runs indeed return non-significant results, but some are actually significant! 

# In[13]:


print(100 * len([x for x in runs if x < 0.05]) / len(runs), "percent of runs are accidentally significant")


# ## ANOVA

# Let's first create some (fake) experimental data. We have two groups of users (old vs. young) which are using two different devices (mouse vs. keyboard). This is a between subjects study, thus every participant uses either the mouse or the keyboard. We measure performance, which we here assume to be dependent on both age and device used.

# In[6]:


data = []
for gid, group in enumerate(["old people", "young people"]):
    for did, device in enumerate(["mouse", "keyboard"]):
        for n in range(6):
            sample = np.random.multivariate_normal(
                mean=[5 + gid * 5, 7 + did * 3],
                cov=[[5, 4], [4, 4]], size=1)
            data.append({'group': group, 'participant': gid * 1000 + n, 'device': device, 'performance': sample.sum()})
data = pd.DataFrame(data)
print(data.head(10))


# We can also plot the data to get an initial idea of how our independent variables influenced the outcome.

# In[7]:


sns.barplot(x='device', y='performance', hue='group', data=data)
plt.show()


# At this point, we would probably like to know whether the keyboard is actually better than the mouse. And are yound people doing better than old people? We can't analyze this with just one t-test. But we can use an ANOVA here.

# In[8]:


model = ols('performance ~ C(group) * C(device)', data=data).fit()
table = sm.stats.anova_lm(model, typ=2)
print(table)


# The results show (your results might differ) that there is indeed a main effect for the user group. Old people perform worse than young people. However, there is no main effect for the device used. There also is no significant interaction between group and device. But if you look at how we setup the experiment above, you might notice that there actually was a difference in performance due to both. Why did we not find it? Our sample size might just be too low. You can explore this by changing how many participants are in the study and running the experiment again. We might also just have been unlucky with our specific sample. 

# ## Post-hoc Testing / Multiple Comparisons

# In the ANOVA above, we look at effects for the groups, but we don't really 
# compare individual conditions. Are old people using a keyboard faster than 
# young people with a mouse? For this we run post-hoc tests that compare these 
# individual conditions. As we only compare two conditions here you could just 
# use a t-test as before. This depends on your data. Let's run an example comparison: 

# In[9]:


old_keyboard = data[(data['group'] == 'old people') & (data['device'] == 'keyboard')]['performance']
young_mouse = data[(data['group'] == 'young people') & (data['device'] == 'mouse')]['performance']
okym = stats.ttest_ind(old_keyboard, young_mouse)
print(okym.pvalue)


# We don't find a significant difference here. What about the other three cases:

# In[10]:


old_mouse = data[(data['group'] == 'old people') & (data['device'] == 'mouse')]['performance']
young_keyboard = data[(data['group'] == 'young people') & (data['device'] == 'keyboard')]['performance']
okyk = stats.ttest_ind(old_keyboard, young_keyboard)
omym = stats.ttest_ind(old_mouse, young_mouse)
omyk = stats.ttest_ind(old_mouse, young_keyboard)
print(okyk.pvalue, omym.pvalue, omyk.pvalue)


# Hey, one was actually significant. But wait, there is something going wrong 
# here. We already saw that we might find a significant result even when the 
# conditions don't really differ. So if we have many different conditions and 
# thus run a whole lot of pairwise comparisons, aren't we likely to find something 
# at some point, just out of luck? Indeed, this is the case and we have to correct 
# our p-value to account for this. For t-tests a common way to do this is Bonferroni 
# correction.

# In[11]:
print("blabla")

newresult = (multipletests([okym.pvalue, okyk.pvalue, omym.pvalue, omyk.pvalue], 0.05, 'bonferroni'))
print(newresult)


# Oops, once we do this, our one found effect disappears. However, note that 
# this does not invalidate the found main effect.
