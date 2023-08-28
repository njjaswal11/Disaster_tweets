#!/usr/bin/env python
# coding: utf-8

# # Disaster Tweets
# 
#    * Smartphones enables people to announce an emergency they’re observing in real-time. Since organizations like NGO cannot look for each and every message, they have to use programmatical approach. But there is a problem, there maybe fake message or there maybe words which are describing a normal situation but the program may consider it as a "disaster tweet". If our program can understand our language, it will be  able to separate disaster tweets from normal tweets respectively.
#    
#    
#    * This approach can be used in similar occasions such as while searching jobs/internship in linkedIn.
#    
#   

# ### In this notebook we will predict whether tweet is about a real disaster or not on the basis of a kaggle dataset.
# 
# In this project, i"ll be performing text preprocessing and then i"ll take advantage of Random-Forest Classifier as well as Naive Bayes Classifier for the classification purpose respectively.
# Also, i"ll be using Bag of Words Model and TF-IDF Vectorizer to convert text into vectors.

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Reading training data
train_data = pd.read_csv('train.csv')
train_data.head()


# In[3]:


train_data.shape


# In[4]:


train_data.id.nunique()


# ### Since all the ids are unique, we wont get any useful information from it. So lets drop it.

# In[5]:


train_data = train_data.drop('id', axis=1)


# In[6]:


train_data.info()


# In[7]:


train_data.location.nunique()


# #### There are large number of locations in the dataset and if the locations in the dataset is ocurring only once or twice, we won't get anything useful for that.
# 
# ### why is location useful??
#  It's because maybe the location is more prone to natural calamities. For example hills, areas near to river, forests(wildfire). 
# 
# #### So I am considering all locations that are repeated atleast 10 times respectively.

# In[8]:


val_locations = []
for i,j in train_data.location.value_counts().iteritems():
    if j > 10 : 
        val_locations.append(i)
        
val_locations


# ### After seeing the locations that are repeated more than 10 times, it is very generalized locations(country) and not specific cities. 
# ### Now I think we won't get any useful informations from locations.
# ### So lets drop it.

# In[9]:


train_data = train_data.drop('location', axis=1)


# In[10]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)


# In[11]:


train_data.keyword.unique()


# In[12]:


val_keys = train_data.keyword.unique()


# ## Data Preprocessing Required for Text Data

# In[13]:


for i,j in train_data['keyword'].iteritems():
    if(not j in val_keys):
        train_data['keyword'][i] = 'random_key'


# In[14]:


ps = PorterStemmer()
for i in range(0, train_data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', train_data['keyword'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    train_data['keyword'][i] = review


# In[15]:


ps = PorterStemmer()
corpus = []
for i in range(0, train_data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', train_data['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# ## NATURAL LANGUAGE PROCESSING
# WE WILL BE USING BAG OF WORDS MODEL & TF-IDF VECTORIZER FOR CONVERTING TEXT DATA INTO VECTORS
# 
# FOR BOTH RANDOM FOREST CLASSIFIER & NAIVE BAYES CLASSIFIER

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB


# # CountVectorizer

# In[17]:


# Implementing Bag of words model
cv = CountVectorizer(max_features=2500, ngram_range=(2, 2))
X = cv.fit_transform(corpus).toarray()
X = pd.DataFrame(X)
X.head()


# In[18]:


# Appending it to the main Dataframe
X = pd.concat([train_data, X], axis=1)
X.head()


# In[19]:


y = X['target']
X = X.drop(['text','target'], axis=1)


# In[20]:


X = pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.15,random_state=2020)

print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)


# ### RANDOM FOREST CLASSIFIER

# In[21]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[22]:


RDclassifier = RandomForestClassifier(random_state=0)
RDclassifier.fit(X_train,y_train)


# In[23]:


print(confusion_matrix(y_test,RDclassifier.predict(X_test)))
print(classification_report(y_test,RDclassifier.predict(X_test)))
print(accuracy_score(y_test, RDclassifier.predict(X_test)))


# ### XGBoost

# In[24]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# In[25]:


print(confusion_matrix(y_test,xgb.predict(X_test)))
print(classification_report(y_test,xgb.predict(X_test)))
print(accuracy_score(y_test, xgb.predict(X_test)))


# ### MULTINOMIAL NAIVE BAYES

# In[26]:


mnb = MultinomialNB()
mnb.fit(X_train,y_train)


# In[27]:


print(confusion_matrix(y_test,mnb.predict(X_test)))
print(classification_report(y_test,mnb.predict(X_test)))
print(accuracy_score(y_test, mnb.predict(X_test)))


# # TF-IDF VECTORIZER

# In[28]:


tfidf = TfidfVectorizer(max_features=2500)
X = tfidf.fit_transform(corpus).toarray()
X = pd.DataFrame(X)
X.head()


# In[29]:


# Appending it to the main Dataframe
X = pd.concat([train_data, X], axis=1)
X.head()


# In[30]:


y = X['target']
X = X.drop(['text','target'], axis=1)


# In[31]:


X = pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=2020)

print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)


# ### RANDOM FOREST CLASSIFIER

# In[32]:


RDclassifier = RandomForestClassifier(random_state=0)
RDclassifier.fit(X_train,y_train)


# In[33]:


print(confusion_matrix(y_test,RDclassifier.predict(X_test)))
print(classification_report(y_test,RDclassifier.predict(X_test)))
print(accuracy_score(y_test, RDclassifier.predict(X_test)))


# ### XGBoost

# In[34]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# In[35]:


print(confusion_matrix(y_test,xgb.predict(X_test)))
print(classification_report(y_test,xgb.predict(X_test)))
print(accuracy_score(y_test, xgb.predict(X_test)))


# ### MULTINOMIAL NAIVE BAYES

# In[36]:


mnb = MultinomialNB()
mnb.fit(X_train,y_train)


# In[37]:


print(confusion_matrix(y_test,mnb.predict(X_test)))
print(classification_report(y_test,mnb.predict(X_test)))
print(accuracy_score(y_test, mnb.predict(X_test)))


# In[38]:


from sklearn.ensemble import AdaBoostClassifier

adb = AdaBoostClassifier()
adb.fit(X_train, y_train)


# In[39]:


print(confusion_matrix(y_test,adb.predict(X_test)))
print(classification_report(y_test,adb.predict(X_test)))
print(accuracy_score(y_test, adb.predict(X_test)))


# In[40]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[41]:


print(confusion_matrix(y_test,dtc.predict(X_test)))
print(classification_report(y_test,dtc.predict(X_test)))
print(accuracy_score(y_test, dtc.predict(X_test)))


# ### Multinomial naive bayes and random forest classifier with TfidfVectorizer gives a good accuracy score and f1 score
# 
# ### We can further increase the accuracy score and f1 score by hyper parameter tuning. 
