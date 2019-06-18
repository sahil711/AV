print("****************** freq_train_test.py *******************")


import pandas as pd
import numpy as np
import copy
import pickle
import sys
import gc
from encoding import FreqeuncyEncoding
import pickle

DATADIR = ''

users=pd.read_csv(DATADIR + 'user_features.csv')
train=pd.read_csv(DATADIR + 'train.csv',dtype={'is_chat': 'uint8', 'node1_id': 'uint32', 'node2_id': 'uint32'})
test=pd.read_csv(DATADIR + 'test.csv',dtype={'id': 'uint32', 'node1_id': 'uint32', 'node2_id': 'uint32'})
df = pd.concat((train.iloc[:,:2],test.iloc[:,1:3]),axis=0)

degree = pickle.load(open("degrees_contact.pkl"))

y = train['is_chat']
train.drop('is_chat', axis=1, inplace=True)

degree_df=pd.DataFrame(degree.items(),columns=['node_id','degree'])
degree_df.node_id=degree_df.node_id.astype('uint32')

df=df.merge(degree_df, left_on='node1_id', right_on='node_id', how='left')
df.drop('node_id',axis=1, inplace=True)
df.rename(columns={'degree':'degree_source'},inplace=True)


df=df.merge(degree_df,left_on='node2_id',right_on='node_id',how='left')
df.drop('node_id',axis=1, inplace=True)
df.rename(columns={'degree':'degree_target'},inplace=True)

df=df.merge(users,left_on='node1_id',right_on='node_id',how='left')
df.drop('node_id',axis=1,inplace=True)
temp=dict(zip(df.columns[df.columns.str.contains('f')],df.columns[df.columns.str.contains('f')]+'_source'))
df.rename(columns=temp,inplace=True)


df=df.merge(users,left_on='node2_id',right_on='node_id',how='left')
df.drop('node_id',axis=1,inplace=True)
temp=dict(zip(df.columns[df.columns.str.contains('f')],df.columns[df.columns.str.contains('f')]+'_target'))
df.rename(columns=temp,inplace=True)



train=df.iloc[:train.shape[0],:]
train['is_chat']=y
test=df.iloc[train.shape[0]:,:]


train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

df=pd.concat((train[['node1_id','node2_id']],test[['node1_id','node2_id']]),axis=0)

fe = FreqeuncyEncoding(categorical_columns=['node1_id','node2_id'],return_df=True,normalize=False)
df=fe.fit_transform(df)


df.node1_id=df.node1_id.astype('uint16')
df.node2_id=df.node2_id.astype('uint16')

train[['node1_id','node2_id']]=df.iloc[:train.shape[0],:]
test[['node1_id','node2_id']]=df.iloc[train.shape[0]:,:]


train['is_chat'] = y

train.degree_source= train.degree_source.astype('float32')
train.degree_target= train.degree_target.astype('float32')

test.degree_source= test.degree_source.astype('float32')
test.degree_target= test.degree_target.astype('float32')


for c in [ u'f1_source_target', u'f2_source_target', u'f3_source_target',
       u'f4_source_target', u'f5_source_target', u'f6_source_target',
       u'f7_source_target', u'f8_source_target', u'f9_source_target',
       u'f10_source_target', u'f11_source_target', u'f12_source_target',
       u'f13_source_target', u'f1_target', u'f2_target', u'f3_target',
       u'f4_target', u'f5_target', u'f6_target', u'f7_target', u'f8_target',
       u'f9_target', u'f10_target', u'f11_target', u'f12_target',
       u'f13_target', u'is_chat']:
    train[c] = train[c].astype('uint8')

train.to_pickle('freq_new_train.pkl')


for c in [ u'f1_source_target', u'f2_source_target', u'f3_source_target',
       u'f4_source_target', u'f5_source_target', u'f6_source_target',
       u'f7_source_target', u'f8_source_target', u'f9_source_target',
       u'f10_source_target', u'f11_source_target', u'f12_source_target',
       u'f13_source_target', u'f1_target', u'f2_target', u'f3_target',
       u'f4_target', u'f5_target', u'f6_target', u'f7_target', u'f8_target',
       u'f9_target', u'f10_target', u'f11_target', u'f12_target',
       u'f13_target']:
    test[c] = test[c].astype('uint8')

test.to_pickle('freq_new_test.pkl')
