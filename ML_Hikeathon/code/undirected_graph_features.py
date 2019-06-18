print("****************** undirected_graph_features.py *******************")

import pandas as pd
import numpy as np
import networkx as nx

DATA_DIR = ''
train = pd.read_csv(DATA_DIR + 'train.csv', usecols=['node1_id','node2_id'], dtype={"is_chat": np.int8})
test = pd.read_csv(DATA_DIR + 'test.csv', usecols=['node1_id','node2_id'], dtype={"is_chat": np.int8})

print( "DataSet: Train-{}, Test-{}".format(train.shape, test.shape))
df = pd.concat([train, test], axis=0)
del train, test

# creating undirected graph
graph = nx.from_pandas_edgelist(df=df, source='node1_id', target='node2_id')

# Creating JC, RSA, PA, AA values for undirected graph  and saving it in a csv
with open( "jc_rsa_pa_aai.csv", "w") as myfile:
    myfile.write("jc,rsa,pa,aa\n")
    for i, row in df.iterrows():
        a, b = row['node1_id'], row['node2_id']
        jc = nx.jaccard_coefficient(G=graph, ebunch=[(a,b)]).next()[2]
        rsa = nx.resource_allocation_index(G=graph, ebunch=[(a,b)]).next()[2]
        pa = nx.preferential_attachment(G=graph,ebunch=[(a,b)]).next()[2]
        try:
            aai = nx.adamic_adar_index(G=graph,ebunch=[(a,b)]).next()[2]
        except:
            aai = ''
        myfile.write("{},{},{},{}\n".format(jc, rsa, pa, aai))


# getting triangles for all the nodes
triangles=nx.algorithms.cluster.triangles(graph)
tri=pd.DataFrame(triangles.items(),columns=['node_id','num_triangle'])

# merging traingles for source and target nodes
df_triangles=df.merge(tri,left_on='node1_id',right_on='node_id',how='left')
df_triangles.drop('node_id',axis=1,inplace=True)
df_triangles.rename(columns={'num_triangle':'triangles_source'},inplace=True)

df_triangles=df_triangles.merge(tri,left_on='node2_id',right_on='node_id',how='left')
df_triangles.drop('node_id',axis=1,inplace=True)
df_triangles.rename(columns={'num_triangle':'triangles_target'},inplace=True)

df_triangles.triangles_source=df_triangles.triangles_source.astype('int16')
df_triangles.triangles_target=df_triangles.triangles_target.astype('int16')

df_triangles.iloc[:,2:].to_pickle('triangles.pkl')



# getting cluster coefficients for all the nodes
clusters = nx.clustering(graph)
clust = pd.DataFrame(clusters.items(), columns=['node_id', 'clust_coeff'])

# merging cluster coefficents  for source and target nodes
df_cluster=df.merge(clust,left_on='node1_id',right_on='node_id',how='left')
df_cluster.drop('node_id',axis=1,inplace=True)
df_cluster.rename(columns={'clust_coeff':'clust_source'},inplace=True)

df_cluster=df_cluster.merge(clust,left_on='node2_id',right_on='node_id',how='left')
df_cluster.drop('node_id',axis=1,inplace=True)
df_cluster.rename(columns={'clust_coeff':'clust_target'},inplace=True)

df_cluster.clust_source=df_cluster.clust_source.astype('float16')
df_cluster.clust_target=df_cluster.clust_target.astype('float16')
df_cluster.iloc[:,2:].to_pickle('cluster_coeffs.pkl')


# getting degree for all the nodes from the graph
undirected_degree = nx.algorithms.degree_centrality(G=graph)

with open('degrees_contact.pkl', 'wb') as output_file:
    pickle.dump(undirected_degree, output_file)
