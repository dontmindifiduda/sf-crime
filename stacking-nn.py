#!/usr/bin/env python
# coding: utf-8

# # Neural Network for Stacking Ensemble

# In[1]:


import numpy as np 
import pandas as pd 

from tensorflow.keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda, BatchNormalization, Activation, PReLU, ReLU

from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 25)


# import warnings
# warnings.filterwarnings('ignore')


# ### Load Pre-Processed Data and Original Training Data for Labels

# In[2]:


X = pd.read_csv('p_train_std_geo.csv', low_memory=False)
X_test = pd.read_csv('p_test_std_geo.csv', low_memory=False)
y_train = pd.read_csv('p_train_y_nn_geo.csv', low_memory=False)
train_df = pd.read_csv('train.csv', low_memory=False)


# ### Generate Training Labels

# In[ ]:


train_df.drop_duplicates(inplace=True)

y_cats = train_df['Category']
unique_cats = np.sort(y_cats.unique())

y_train = train_df['Category']


# ### Get Neural Network

# In[ ]:


def get_model(x_tr, y_tr):
    K.clear_session()
    inp = Input(shape = (x_tr.shape[1],))
    
    dl_1 = 2048  
    drop_1 = 0.5
    dl_2 = 1024 
    drop_2 = 0.4
    dl_3 = 512
    drop_3 = 0.3 
    
    x = Dense(dl_1, input_dim=x_tr.shape[1])(inp) 
    x = Activation('relu')(x)
    x = Dropout(drop_1)(x)
    x = BatchNormalization()(x)
    
    x = Dense(dl_2)(x)
    x = Activation('relu')(x)
    x = Dropout(drop_2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(dl_3)(x)
    x = Activation('relu')(x)
    x = Dropout(drop_3)(x)
    x = BatchNormalization()(x)
    
    out = Dense(39, activation='softmax')(x)
    
    
    bsz = 256
    steps = x_tr.shape[0]/bsz
    
    model = Model(inp,out)
    
    opt = Adam(learning_rate=0.0008)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[])
    
    model.fit(x_tr, y_tr, epochs=50, batch_size=bsz, verbose=1)
    
    
    
#     es = EarlyStopping(monitor='val_loss', patience=10) 

#     y_tr = np.asarray(y_tr)
#     y_val = np.asarray(y_val)
#     history = model.fit(x_tr, y_tr, epochs=40, batch_size=bsz, verbose=1)

    return model


# ### Out-of-Fold Predictions

# In[ ]:


from sklearn.model_selection import StratifiedKFold

ntrain = X.shape[0]
ntest = X_test.shape[0]
nclass = 39

SEED = 1
NFOLDS = 5

skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=1)


# In[ ]:


X_train = X.to_numpy()


# In[ ]:


def get_oof(x_in, y_in, x_test):
    oof_train = np.zeros((len(x_in), nclass))
    oof_test = np.zeros((len(x_test), nclass))
    oof_test_skf = np.empty((NFOLDS, len(x_test), nclass))
    
    fold = 0
    

    for train_index, test_index in skf.split(x_in, y_in):
        print('Fold: {}'.format(fold+1))
        x_tr = x_in[train_index]
        y_tr = y_in.iloc[train_index]
        x_te = x_in[test_index]
        
        
        y = np.zeros((y_tr.shape[0], 39))
        for idx, target in enumerate(list(y_tr)):
            y[idx, np.where(unique_cats == target)] = 1

        # y = pd.DataFrame(y, columns = unique_cats)
        

        clf = get_model(x_tr, y)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[fold, : ] = clf.predict(x_test)

        fold += 1
    
    oof_test[:] = oof_test_skf.mean(axis=0)
    
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


nn_oof_train, nn_oof_test = get_oof(X_train, y_train, X_test) 


# In[ ]:


nn_oof_train = pd.DataFrame(nn_oof_train)
nn_oof_test = pd.DataFrame(nn_oof_test)


# In[ ]:


nn_oof_train.to_csv('nn_oof_train_geo.csv', index=False)
nn_oof_test.to_csv('nn_oof_test_geo.csv', index=False)


# In[ ]:


sub_df = pd.DataFrame(nn_oof_test.to_numpy().reshape(-1, 39), columns=unique_cats)

sub_df.index = sub_df.index.set_names(['Id'])
sub_df.reset_index(drop=False, inplace=True)

sub_df.to_csv('nn_stack_geo.csv', index=False)

