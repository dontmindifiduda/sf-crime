#!/usr/bin/env python
# coding: utf-8

# # CatBoost Model for Stacking Ensemble

# In[1]:


import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 25)


# ### Load Pre-Processed Data and Original Training Data for Labels

# In[15]:


X = pd.read_csv('p_train.csv', low_memory=False)
X_test = pd.read_csv('p_test.csv', low_memory=False)
train_df = pd.read_csv('train.csv', low_memory=False)


# ### Generate Training Labels

# In[13]:


train_df.drop_duplicates(inplace=True)

y_cats = train_df['Category']
unique_cats = np.sort(y_cats.unique())

le = LabelEncoder()
y_train = le.fit_transform(y_cats)


# ### scikit-learn Wrapper Class

# In[4]:


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train, verbose=True)
        
    def predict(self, x):
        return self.clf.predict(x)
    
    def predict_proba(self, x):
        return self.clf.predict_proba(x)
    
    def fit(self, x, y):
        return self.clf.fit(x,y, verbose=True)
    
    def feature_importance(self, x, y):
        print(self.clf.fit(x,y).featue_importances_)


# ### Model Parameters

# In[5]:


categorical_features = ["DayOfWeek", "PdDistrict", "Intersection", "Special Time", "XYcluster"]

cat_params = {
    
    'n_estimators': 5000, 
    'task_type': 'GPU', # NOTE:  training with a GPU is highly recommended 
    'learning_rate': 0.05, 
    'classes_count': 39, 
    'loss_function': 'MultiClass',
#     'cat_features': categorical_features, 
    'verbose': 1
    
}


# ### Out-of-Fold Predictions

# In[6]:


ntrain = X.shape[0]
ntest = X_test.shape[0]
nclass = 39

SEED = 1
NFOLDS = 5

skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=1)


# In[7]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((len(x_train), nclass))
    oof_test = np.zeros((len(x_test), nclass))
    oof_test_skf = np.empty((NFOLDS, len(x_test), nclass))
    
    fold = 0

    for train_index, test_index in skf.split(x_train, y_train):
        print('Fold: {}'.format(fold+1))
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)
        oof_test_skf[fold, : ] = clf.predict_proba(x_test)

        fold += 1
    
    oof_test[:] = oof_test_skf.mean(axis=0)
    
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[8]:


cat_model = SklearnHelper(clf=CatBoostClassifier, seed=SEED, params=cat_params)


# In[9]:


X_train = X.values
X_test = X_test.values


# In[14]:


cat_oof_train, cat_oof_test = get_oof(cat_model, X_train, y_train, X_test) 


# In[ ]:


cat_oof_train = pd.DataFrame(cat_oof_train)
cat_oof_test = pd.DataFrame(cat_oof_test)


# In[ ]:


cat_oof_train.to_csv('cat_oof_train.csv', index=False)
cat_oof_test.to_csv('cat_oof_test.csv', index=False)


# In[ ]:


sub_df = pd.DataFrame(cat_oof_test.to_numpy().reshape(-1, 39), columns=unique_cats)

sub_df.index = sub_df.index.set_names(['Id'])
sub_df.reset_index(drop=False, inplace=True)

sub_df.to_csv('cat_stack.csv', index=False)

