#!/usr/bin/env python
# coding: utf-8

# # Preprocessing Script
# 
# This script takes train.csv and test.csv as input and generates the following preprocessed output files:
# 
# - p_train.csv - training data
# - p_test.csv - test data
# - p_train_y_nn.csv - training data labels processed for neural network
# - p_train_y_label.csv - training data labels processed for tree-based models
# - p_train_std.csv - standardized training data for neural network
# - p_test_std.csv - standardized test data for neural network

# In[3]:


import numpy as np 
import pandas as pd 
import datetime
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 25)

import gensim

# import warnings
# warnings.filterwarnings('ignore')


# In[4]:


train_df = pd.read_csv('train.csv', low_memory=False)  # update path as needed
test_df = pd.read_csv('test.csv', low_memory=False) # update path as needed


# ### Generate Features from 'Date'

# In[5]:


def process_dates(df):
    
    df['Dates'] = df['Dates'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['Year'] = df['Dates'].apply(lambda x: x.year)
    df['Month'] = df['Dates'].apply(lambda x: x.month)
    df['Day'] = df['Dates'].apply(lambda x: x.day)
    df['Hour'] = df['Dates'].apply(lambda x: x.hour)
    df['Minute'] = df['Dates'].apply(lambda x: x.minute)
    df['Special Time'] = df['Minute'].isin([0,30]).astype(int)
    df.drop('Dates', axis=1, inplace=True)
    
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x == 'Saturday' or x == 'Sunday' else 0)
    df['Night'] = df['Hour'].apply(lambda x: 1 if x > 6 and x < 18 else 0)
    
    
    return df


# ### Drop Duplicate Training Entries and Unuseful Columns, Manage Outliers

# In[6]:


# drop duplicates
train_df.drop_duplicates(inplace=True)


# In[7]:


# manage outliers
train_df.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
test_df.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='mean')

for district in train_df['PdDistrict'].unique():
    train_df.loc[train_df['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train_df.loc[train_df['PdDistrict'] == district, ['X', 'Y']])
    test_df.loc[test_df['PdDistrict'] == district, ['X', 'Y']] = imp.transform(
        test_df.loc[test_df['PdDistrict'] == district, ['X', 'Y']])


# In[8]:


# drop unuseful columns 

drop_cols = ['Descript', 'Resolution', 'Id']

for col in drop_cols:
    if col in train_df.columns:
        train_df.drop(col, axis=1, inplace=True)
    if col in test_df.columns:
        test_df.drop(col, axis=1, inplace=True)
        
X = train_df.drop('Category', axis=1)
X_test = test_df


# ### Encode Training Labels

# In[9]:


y_cats = train_df['Category']
unique_cats = np.sort(y_cats.unique())

# neural network
y = np.zeros((y_cats.shape[0], 39))
for idx, target in enumerate(list(y_cats)):
    y[idx, np.where(unique_cats == target)] = 1

y_nn = pd.DataFrame(y, columns = unique_cats)


# tree-based models
y_label = train_df['Category']
le = LabelEncoder()
y_label = le.fit_transform(y_label)


# ### Create Combined Dataset

# In[10]:


train_length = X.shape[0]

combined = pd.concat([X, X_test], ignore_index=True)
combined = process_dates(combined)   


# ### Generate Address Embeddings

# In[11]:


address_list = [address.split(' ') for address in combined['Address']]
address_model = gensim.models.Word2Vec(address_list, min_count=1)
encoded_address = np.zeros((combined.shape[0], 100))
for i in range(len(address_list)):
    for j in range(len(address_list[i])):
        encoded_address[i] += address_model.wv[address_list[i][j]]
    encoded_address[i] /= len(address_list[i])


# ### Address Features

# In[12]:


combined['Intersection'] = combined['Address'].apply(lambda x: 1 if '/' in x else 0)


# ### Transformations & Aggregations

# In[13]:


# xy_scaler = StandardScaler()
# xy_scaler.fit(combined[['X', 'Y']])
# combined[['X', 'Y']] = xy_scaler.transform(combined[['X', 'Y']])

X_median = combined["X"].median()
Y_median = combined["Y"].median()

combined["X+Y"] = combined["X"] + combined["Y"]
combined["X-Y"] = combined["X"] - combined["Y"]

# combined["XY45_1"] = combined["X"] * np.cos(np.pi / 4) + combined["Y"] * np.sin(np.pi / 4)
combined["XY45_2"] = combined["Y"] * np.cos(np.pi / 4) - combined["X"] * np.sin(np.pi / 4)

combined["XY30_1"] = combined["X"] * np.cos(np.pi / 6) + combined["Y"] * np.sin(np.pi / 6)
combined["XY30_2"] = combined["Y"] * np.cos(np.pi / 6) - combined["X"] * np.sin(np.pi / 6)

combined["XY60_1"] = combined["X"] * np.cos(np.pi / 3) + combined["Y"] * np.sin(np.pi / 3)
combined["XY60_2"] = combined["Y"] * np.cos(np.pi / 3) - combined["X"] * np.sin(np.pi / 3)


combined["XY1"] = (combined["X"] - combined["X"].min()) ** 2 + (combined["Y"] - combined["Y"].min()) ** 2
combined["XY2"] = (combined["X"].max() - combined["X"]) ** 2 + (combined["Y"] - combined["Y"].min()) ** 2
combined["XY3"] = (combined["X"] - combined["X"].min()) ** 2 + (combined["Y"].max() - combined["Y"]) ** 2
combined["XY4"] = (combined["X"].max() - combined["X"]) ** 2 + (combined["Y"].max() - combined["Y"]) ** 2
combined["XY5"] = (combined["X"] - X_median) ** 2 + (combined["Y"] - Y_median) ** 2

combined["XY_rad"] = np.sqrt(np.power(combined['Y'], 2) + np.power(combined['X'], 2))


# ### Principal Components Analysis

# In[14]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(combined[["X", "Y"]])
XYt = pca.transform(combined[["X", "Y"]])

combined["XYpca1"] = XYt[:, 0]
combined["XYpca2"] = XYt[:, 1]


# ### Gaussian Mixture Model

# In[15]:


from sklearn.mixture import GaussianMixture

clf = GaussianMixture(n_components=150, covariance_type="diag",
                      random_state=0).fit(combined[["X", "Y"]])
combined["XYcluster"] = clf.predict(combined[["X", "Y"]])


# ## Log Odds
# 
# NOTE:  Log odds are also calculated for each individual crime category for each of the groupings below. The resulting values included a large amount of redundancy and hurt model performance, so they were omitted.
# 
# ### Log Odds - Location

# In[ ]:


# Training Data

addresses = sorted(train_df['Address'].unique())
categories = sorted(train_df['Category'].unique())

C_counts = train_df.groupby(['Category']).size()
A_C_counts = train_df.groupby(['Address', 'Category']).size()
A_counts = train_df.groupby(['Address']).size()

logodds = {}
logoddsPA = {}

MIN_CAT_COUNTS = 2

default_logodds = np.log(C_counts / len(train_df)) - np.log(1.0 - C_counts / float(len(train_df)))

for addr in addresses:
    
    PA = A_counts[addr] / float(len(train_df))
    logoddsPA[addr] = np.log(PA)- np.log(1.-PA)
    logodds[addr] = deepcopy(default_logodds)
    
    for cat in A_C_counts[addr].keys():        
        if (A_C_counts[addr][cat] > MIN_CAT_COUNTS) and A_C_counts[addr][cat] < A_counts[addr]:
            PA = A_C_counts[addr][cat] / float(A_counts[addr])
            logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0-PA)

    logodds[addr] = pd.Series(logodds[addr])
    logodds[addr].index = range(len(categories))
    
# Test Data

new_addresses = sorted(test_df["Address"].unique())
new_A_counts = test_df.groupby("Address").size()

only_new = set(new_addresses + addresses) - set(addresses)
only_old = set(new_addresses + addresses) - set(new_addresses)
in_both = set(new_addresses).intersection(addresses)

for addr in only_new:
    PA = new_A_counts[addr] / float(len(test_df) + len(train_df))
    logoddsPA[addr] = np.log(PA) - np.log(1.0 - PA)
    logodds[addr] = deepcopy(default_logodds)
    logodds[addr].index = range(len(categories))
for addr in in_both:
    PA = (A_counts[addr] + new_A_counts[addr]) / float(len(test_df) + len(train_df))
    logoddsPA[addr] = np.log(PA) - np.log(1.0 - PA)
    
address_features = combined['Address'].apply(lambda x: logodds[x])
address_features.columns = ['Address Logodds ' + str(x) for x in range(len(address_features.columns))]
combined["logoddsPA"] = combined["Address"].apply(lambda x: logoddsPA[x])

# combined = pd.concat([combined, address_features], axis=1, sort=False)


# ### Log Odds - Hour

# In[ ]:


# Training Data
train_df = process_dates(train_df)

hours = sorted(train_df['Hour'].unique())
# categories = sorted(train_df['Category'].unique())

# C_counts = train_df.groupby(['Category']).size()
H_C_counts = train_df.groupby(['Hour', 'Category']).size()
H_counts = train_df.groupby(['Hour']).size()

hour_logodds = {}
hour_logoddsPA = {}

MIN_CAT_COUNTS = 2

default_hour_logodds = np.log(C_counts / len(train_df)) - np.log(1.0 - C_counts / float(len(train_df)))

for hr in hours:
    
    PH = H_counts[hr] / float(len(train_df))
    hour_logoddsPA[hr] = np.log(PH)- np.log(1.-PH)
    hour_logodds[hr] = deepcopy(default_hour_logodds)
    
    for cat in H_C_counts[hr].keys():        
        if (H_C_counts[hr][cat] > MIN_CAT_COUNTS) and H_C_counts[hr][cat] < H_counts[hr]:
            PH = H_C_counts[hr][cat] / float(H_counts[hr])
            hour_logodds[hr][categories.index(cat)] = np.log(PH) - np.log(1.0-PH)

    hour_logodds[hr] = pd.Series(hour_logodds[hr])
    hour_logodds[hr].index = range(len(categories))
    
# Test Data
test_df = process_dates(test_df)

new_hours = sorted(test_df["Hour"].unique())
new_H_counts = test_df.groupby("Hour").size()

only_new = set(new_hours + hours) - set(hours)
only_old = set(new_hours + hours) - set(new_hours)
in_both = set(new_hours).intersection(hours)

for hr in only_new:
    PH = new_H_counts[hr] / float(len(test_df) + len(train_df))
    hour_logoddsPA[hr] = np.log(PH) - np.log(1.0 - PH)
    hour_logodds[hr] = deepcopy(default_hour_logodds)
    hour_logodds[hr].index = range(len(categories))
for hr in in_both:
    PH = (H_counts[hr] + new_H_counts[hr]) / float(len(test_df) + len(train_df))
    hour_logoddsPA[hr] = np.log(PH) - np.log(1.0 - PH)

hour_features = combined['Hour'].apply(lambda x: hour_logodds[x])
hour_features.columns = ['Hour Logodds ' + str(x) for x in range(len(hour_features.columns))]
combined["hour logoddsPA"] = combined["Hour"].apply(lambda x: hour_logoddsPA[x])

# combined = pd.concat([combined, hour_features], axis=1, sort=False)


# ### Log Odds - DayOfWeek

# In[ ]:


# Training Data

dows = sorted(train_df['DayOfWeek'].unique())
# categories = sorted(train_df['Category'].unique())

# C_counts = train_df.groupby(['Category']).size()
D_C_counts = train_df.groupby(['DayOfWeek', 'Category']).size()
D_counts = train_df.groupby(['DayOfWeek']).size()

dow_logodds = {}
dow_logoddsPA = {}

MIN_CAT_COUNTS = 2

default_dow_logodds = np.log(C_counts / len(train_df)) - np.log(1.0 - C_counts / float(len(train_df)))

for dow in dows:
    
    PD = D_counts[dow] / float(len(train_df))
    dow_logoddsPA[dow] = np.log(PD)- np.log(1.-PD)
    dow_logodds[dow] = deepcopy(default_dow_logodds)
    
    for cat in D_C_counts[dow].keys():        
        if (D_C_counts[dow][cat] > MIN_CAT_COUNTS) and D_C_counts[dow][cat] < D_counts[dow]:
            PD = D_C_counts[dow][cat] / float(D_counts[dow])
            dow_logodds[dow][categories.index(cat)] = np.log(PD) - np.log(1.0-PD)

    dow_logodds[dow] = pd.Series(dow_logodds[dow])
    dow_logodds[dow].index = range(len(categories))
    
new_dows = sorted(test_df["DayOfWeek"].unique())
new_D_counts = test_df.groupby("DayOfWeek").size()

only_new = set(new_dows + dows) - set(dows)
only_old = set(new_dows + dows) - set(new_dows)
in_both = set(new_dows).intersection(dows)

for dow in only_new:
    PD = new_D_counts[dow] / float(len(test_df) + len(train_df))
    dow_logoddsPD[dow] = np.log(PD) - np.log(1.0 - PD)
    dow_logodds[dow] = deepcopy(default_dow_logodds)
    dow_logodds[dow].index = range(len(categories))
    
for dow in in_both:
    PD = (D_counts[dow] + new_D_counts[dow]) / float(len(test_df) + len(train_df))
    dow_logoddsPA[dow] = np.log(PD) - np.log(1.0 - PD)
    
dow_features = combined['DayOfWeek'].apply(lambda x: dow_logodds[x])
dow_features.columns = ['DOW Logodds ' + str(x) for x in range(len(dow_features.columns))]
combined["dow logoddsPA"] = combined["DayOfWeek"].apply(lambda x: dow_logoddsPA[x])

# combined = pd.concat([combined, dow_features], axis=1, sort=False)


# ### Add Address Embeddings

# In[ ]:


enc_cols = []

for i in range(encoded_address.shape[1]):
    enc_cols.append("EncodedAddress{}".format(i))
    
enc_add_df = pd.DataFrame(encoded_address, columns=enc_cols)

combined = pd.concat([combined, enc_add_df], axis=1, sort=False)

combined.drop('Address', axis=1, inplace=True)


# ### Encoding

# In[ ]:


categorical_features = ["DayOfWeek", "PdDistrict", "Intersection", "Special Time", "XYcluster"]

for col in combined.columns:
    if col in categorical_features:
        oe = OrdinalEncoder()
        combined[col] = oe.fit_transform(combined[col].values.reshape(-1,1))
        combined[col] = combined[col].astype(int)
    elif combined.dtypes[col] == 'object':
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])


# ### Cyclical Time Features

# In[ ]:


# combined['HourCos'] = combined['Hour'].apply(lambda x: np.cos(x*2*np.pi)/24) 
# combined['DayOfWeekCos'] = combined['DayOfWeek'].apply(lambda x: np.cos(x*2*np.pi)/7) 
# combined['MonthCos'] = combined['Month'].apply(lambda x: np.cos(x*2*np.pi)/12) 


# ### Split Back Into Train / Test

# In[ ]:


X = combined[:train_length]
X_test = combined[train_length:]


# ### Standardization for Neural Network
# 
# NOTE:  This step should be done within each cross validation fold in the stacking script instead of on the whole dataset. We will update it in the future.

# In[ ]:


scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_test_std = scaler.transform(X_test)

X_std = pd.DataFrame(X_std, columns=combined.columns)
X_test_std = pd.DataFrame(X_test_std, columns=combined.columns)


# ### Generate Preprocessed Training / Test Data Files

# In[ ]:


y_label = pd.DataFrame(y_label, columns=['Category'])

X.to_csv('p_train.csv', index=False)
X_test.to_csv('p_test.csv', index=False)

X_std.to_csv('p_train_std.csv', index=False)
X_test_std.to_csv('p_test_std.csv', index=False)

