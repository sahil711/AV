print("****************** directed_degrees_feature_creation.py *******************")

import pandas as pd
import numpy as np
import networkx as nx

DATA_DIR = ''
train = pd.read_csv(DATA_DIR + 'train.csv', usecols=['node1_id','node2_id'], dtype={"is_chat": np.int8})
test = pd.read_csv(DATA_DIR + 'test.csv', usecols=['node1_id','node2_id'], dtype={"is_chat": np.int8})

print( "DataSet: Train-{}, Test-{}".format(train.shape, test.shape))
df = pd.concat([train, test], axis=0)
del train, test

# creating directed graph
graph = nx.from_pandas_edgelist(df=df, source='node1_id', target='node2_id', create_using=nx.DiGraph)

# getting directed degree, directed degree in, directed degree out for all the nodes
directed_degree = nx.algorithms.degree_centrality(G=graph)
directed_degree_out = nx.algorithms.out_degree_centrality(G=graph)
directed_degree_in = nx.algorithms.in_degree_centrality(G=graph)

# merginging degrees with the dataframe 
degree_df=pd.DataFrame(directed_degree.items(),columns=['node_id','directed_degree'])
df=df.merge(degree_df,left_on='node1_id',right_on='node_id',how='left')
df.drop('node_id',axis=1,inplace=True)
df.rename(columns={'directed_degree':'directed_degree_source'},inplace=True)

df=df.merge(degree_df,left_on='node2_id',right_on='node_id',how='left')
df.drop('node_id',axis=1,inplace=True)
df.rename(columns={'directed_degree':'directed_degree_target'},inplace=True)


temp=pd.DataFrame(directed_degree_in.items(),columns=['node_id','directed_degree'])
df=df.merge(temp,left_on='node1_id',right_on='node_id',how='left')
df.drop('node_id',axis=1,inplace=True)
df.rename(columns={'directed_degree':'directed_degree_source_in'},inplace=True)

df=df.merge(temp,left_on='node2_id',right_on='node_id',how='left')
df.drop('node_id',axis=1,inplace=True)
df.rename(columns={'directed_degree':'directed_degree_target_in'},inplace=True)

temp=pd.DataFrame(directed_degree_out.items(),columns=['node_id','directed_degree'])
df=df.merge(temp,left_on='node1_id',right_on='node_id',how='left')
df.drop('node_id',axis=1,inplace=True)
df.rename(columns={'directed_degree':'directed_degree_source_out'},inplace=True)


df=df.merge(temp,left_on='node2_id',right_on='node_id',how='left')
df.drop('node_id',axis=1,inplace=True)
df.rename(columns={'directed_degree':'directed_degree_target_out'},inplace=True)


df.iloc[:,2:].to_pickle('directed_degrees.pkl')
