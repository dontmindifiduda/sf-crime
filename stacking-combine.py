#!/usr/bin/env python
# coding: utf-8

# # Stacking Ensemble

# In[17]:


import numpy as np 
import pandas as pd 
import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import lightgbm as lgb

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 25)

import gensim


# ### Load Pre-Processed Data

# In[18]:


lgb_oof_train = pd.read_csv('lgb_oof_train.csv', low_memory=False)
lgb_oof_test = pd.read_csv('lgb_oof_test.csv', low_memory=False)


# In[19]:


rf_oof_train = pd.read_csv('rf_oof_train.csv', low_memory=False)
rf_oof_test = pd.read_csv('rf_oof_test.csv', low_memory=False)


# In[20]:


cat_oof_train = pd.read_csv('cat_oof_train.csv', low_memory=False)
cat_oof_test = pd.read_csv('cat_oof_test.csv', low_memory=False)


# In[ ]:


nn_oof_train = pd.read_csv('nn_oof_train.csv', low_memory=False)
nn_oof_test = pd.read_csv('nn_oof_test.csv', low_memory=False)


# In[ ]:


xgb_oof_train = pd.read_csv('xgb_oof_train.csv', low_memory=False)
xgb_oof_test = pd.read_csv('xgb_oof_test.csv', low_memory=False)


# ### Load Original Training Data for Labels

# In[23]:


train_df = pd.read_csv('train.csv', low_memory=False)

train_df.drop_duplicates(inplace=True)

y_cats = train_df['Category']
unique_cats = np.sort(y_cats.unique())

le = LabelEncoder()
y_train = le.fit_transform(y_cats)


# ### Reshape Model Output Files

# In[24]:


lgb_oof_train = lgb_oof_train.to_numpy().reshape(-1, 39)
lgb_oof_test = lgb_oof_test.to_numpy().reshape(-1, 39)


# In[ ]:


rf_oof_train = rf_oof_train.to_numpy().reshape(-1, 39)
rf_oof_test = rf_oof_test.to_numpy().reshape(-1, 39)


# In[25]:


cat_oof_train = cat_oof_train.to_numpy().reshape(-1, 39)
cat_oof_test = cat_oof_test.to_numpy().reshape(-1, 39)


# In[26]:


nn_oof_train = nn_oof_train.to_numpy().reshape(-1, 39)
nn_oof_test = nn_oof_test.to_numpy().reshape(-1, 39)


# In[ ]:


xgb_oof_train = xgb_oof_train.to_numpy().reshape(-1, 39)
xgb_oof_test = xgb_oof_test.to_numpy().reshape(-1, 39)


# ### Create Column Names for Each Set of Model Predictions

# In[27]:


lgb_cols = ['LGB_' + x for x in unique_cats]
lgb_train = pd.DataFrame(lgb_oof_train, columns=lgb_cols)
lgb_test = pd.DataFrame(lgb_oof_test, columns=lgb_cols)


# In[ ]:


rf_cols = ['RF_' + x for x in unique_cats]
rf_train = pd.DataFrame(rf_oof_train, columns=rf_cols)
rf_test = pd.DataFrame(rf_oof_test, columns=rf_cols)


# In[28]:


cat_cols = ['Cat_' + x for x in unique_cats]
cat_train = pd.DataFrame(cat_oof_train, columns=cat_cols)
cat_test = pd.DataFrame(cat_oof_test, columns=cat_cols)


# In[29]:


nn_cols = ['NN_' + x for x in unique_cats]
nn_train = pd.DataFrame(nn_oof_train, columns=nn_cols)
nn_test = pd.DataFrame(nn_oof_test, columns=nn_cols)


# In[ ]:


xgb_cols = ['XGB_' + x for x in unique_cats]
xgb_train = pd.DataFrame(xgb_oof_train, columns=xgb_cols)
xgb_test = pd.DataFrame(xgb_oof_test, columns=xgb_cols)


# ### Combine Model Predictions

# In[30]:


X_train = pd.concat([lgb_train, rf_train, cat_train, nn_train, xgb_train], axis=1, sort=False)
X_test = pd.concat([lgb_test, rf_test, cat_test, nn_test, xgb_test], axis=1, sort=False)


# In[31]:


def get_model(x_tr, y_tr):
    
    param = {
         'num_classes': 39,
         'learning_rate': 0.01,
         'objective': 'multiclass',
         'boosting': "gbdt",
         'metric': 'multi_logloss',
         'verbosity': 1
    }
    
    train_ds = lgb.Dataset(x_tr, label=y_tr)
        
    num_round = 1000
    mod_results = {}
    
    model = lgb.train(param, train_ds, num_round, valid_sets=[train_ds], valid_names=['train'], 
                      evals_result=mod_results)
            

    return model, mod_results


# In[32]:


mod = get_model(X_train, y_train)


# In[33]:


preds = mod[0].predict(X_test)


# In[34]:


sub_df = pd.DataFrame(preds, columns=unique_cats)

sub_df.index = sub_df.index.set_names(['Id'])
sub_df.reset_index(drop=False, inplace=True)

sub_df.to_csv('stack_13.csv', index=False)


# In[ ]:




