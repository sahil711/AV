print("****************** leak_analysis.py *******************")
'''
Creates Reverse Features
'''

import pandas as pd
import numpy as np

DATADIR = ''
train = pd.read_csv(DATADIR + 'train.csv', dtype={"is_chat": np.int8})
test = pd.read_csv(DATADIR + 'test.csv',dtype={"is_chat": np.int8})

df = pd.concat((train.iloc[:,:2], test.iloc[:,1:3]), axis=0)

#follows the logic to identify the reverse connections and creating features for that
# sorted the nodes and joining them, to get edges with reverse connection
a = df[['node1_id', 'node2_id']].min(axis=1).astype(str) +'_'+ df[['node1_id', 'node2_id']].max(axis=1).astype(str)
# get all the duplicated edges
b = a.duplicated()
# keeping only the duplicated edges()
dup = a[b]

df_new = df.copy(deep=True)
df_new = df_new.reset_index(drop=True)

df_new['a'] = a.values
df_new['b'] = b.values
df_new['is_chat']= train.is_chat


### data frame containing only duplicated enteries
df_new_2 = df_new[df_new.a.isin(dup)]
df_new_2 = df_new_2.sort_values(by='a')
df_new_2.reset_index(drop=True, inplace=True)

df_new_2['flag'] = df_new_2.index%2
df_new_2['is_chat_1'] = df_new_2.is_chat

df1, df2 = df_new_2[df_new_2.flag==0], df_new_2[df_new_2.flag==1]

df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df1['leak_feature']=df2.is_chat_1
df2['leak_feature']=df1.is_chat_1

temp=pd.concat((df1[['node1_id','node2_id','is_chat','leak_feature','b']],
           df2[['node1_id','node2_id','is_chat','leak_feature','b']]),axis=0).reset_index(drop=True)


temp_df=df.merge(temp[['node1_id','node2_id','leak_feature','b']],on=['node1_id','node2_id'],how='left')

temp_df.leak_feature.fillna(-1,inplace=True)
temp_df.b.fillna(False,inplace=True)

temp_df.b=temp_df.b.astype('int8')
temp_df.leak_feature=temp_df.leak_feature.astype('int8')

temp_df[['leak_feature','b']].to_pickle('leak_feature.pkl')
