#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import mailbox

MBOX = '../final_project/prattischool.mbox'
mbox = mailbox.mbox(MBOX)


# In[5]:


def get_body(message):
  if message.is_multipart():
    return get_body(message.get_payload()[0]) # the assumption is that multipart messages will always return two differently-formatted versions of the same message, and that it's okay to only keep the first one
  else:
    return message.get_payload()


# In[6]:


mbox_dict = {}
for i, msg in enumerate(mbox):
    mbox_dict[i] = {}
    for header in msg.keys():
        mbox_dict[i][header] = msg[header]
    mbox_dict[i]['Body'] = get_body(msg)

df = pd.DataFrame.from_dict(mbox_dict, orient='index')
df


# In[7]:


corpus = df['Body']


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()


# In[9]:


X_count = count_vect.fit_transform(corpus)


# In[10]:


print(X_count.toarray())


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = df['Body']
v = TfidfVectorizer(stop_words='english', strip_accents='ascii', ngram_range=(1,3), max_df=0.80, min_df=0.01, max_features=50000) 
transform = v.fit_transform(corpus)


# In[14]:


X = transform.toarray()
X.shape


# In[15]:


x_df = pd.DataFrame(X, columns=v.get_feature_names())
x_df


# In[16]:


terms = v.get_feature_names()
print(terms)


# In[17]:


# from sklearn.cluster import KMeans

# model = KMeans(n_clusters=5, n_init=10)
# model.fit(X)

# y_predicted = model.fit_predict(X)
# plt.scatter(X[:,0],X[:,1], s=10, c=y_predicted);


# In[18]:


from sklearn.cluster import KMeans

k = 4
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10)
model.fit(X)


# In[22]:


import matplotlib.pyplot as plt
y_predicted = model.fit_predict(X)
plt.scatter(X[:,0],X[:,1], s=10, c=y_predicted)
plt.savefig('kmeans.png', transparent=True);


# In[ ]:




