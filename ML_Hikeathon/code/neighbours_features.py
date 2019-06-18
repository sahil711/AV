print("****************** neighbours_features.py *******************")

import pandas as pd
import numpy as np
import networkx as nx

DATADIR = ''
train = pd.read_csv(DATADIR + 'train.csv', usecols=['node1_id', 'node2_id'])
test = pd.read_csv(DATADIR + 'test.csv', usecols=['node1_id', 'node2_id'])

# concatinating the datasets
df = pd.concat([train, test], axis=0)

# creating undirected graph and getting neighbours for all the nodes
graph = nx.from_pandas_edgelist(df=df, source='node1_id', target='node2_id')
neighbour = {node:[n for n in graph.neighbors(node)] for node in graph.nodes}

# creating undirected chat graph and getting chat neighbours for all the nodes
train = pd.read_csv(DATADIR + 'train.csv')
chat = train[train.is_chat==1]
chat_graph = nx.from_pandas_edgelist(df=chat.iloc[:,:2],source='node1_id',target='node2_id')
is_chat_neighbour = {node:[n for n in chat_graph.neighbors(node)] for node in chat_graph.nodes}


# clearing graph from memory
graph.clear()
chat_graph.clear()
del graph, chat_graph, train, test

# creating all neighbours features for the source and target pairs in the dataset
with open('neigbours_vars.csv', 'w') as myfile:
    for i, row in df.iterrows():
        a, b = row['node1_id'], row['node2_id']
        # getting neigbours of source and target
        neighbour_a, neighbour_b = set(neighbour[int(a)]), set(neighbour[int(b)])
        # getting chat neighbour of source and target
        try:
            chat_neighbour_a=set(is_chat_neighbour[int(a)])
        except:
            chat_neighbour_a=set()
        try:
            chat_neighbour_b=set(is_chat_neighbour[int(b)])
        except:
            chat_neighbour_b=set()

        # removing source and target nodes from neigbours
        if a in neighbour_a: neighbour_a.remove(a)
        if b in neighbour_a: neighbour_a.remove(b)
        if a in neighbour_b:neighbour_b.remove(a)
        if b in neighbour_b:neighbour_b.remove(b)

        # removing source and target nodes from chat neigbours
        if a in chat_neighbour_a: chat_neighbour_a.remove(a)
        if b in chat_neighbour_a: chat_neighbour_a.remove(b)
        if a in chat_neighbour_b:chat_neighbour_b.remove(a)
        if b in chat_neighbour_b:chat_neighbour_b.remove(b)

        # getting number of chat neighours of the all the neighbours of source
        na_n_c_n = [len(is_chat_neighbour.get(node, [])) for node in neighbour_a]
        # getting number of chat neighours of the all the neighbours of target
        nb_n_c_n = [len(is_chat_neighbour.get(node, [])) for node in neighbour_b]

        # getting number of chat neighours of the all the mutual neighbours of source and target
        mutual_n_c_n = [len(is_chat_neighbour.get(node, [])) for node in neighbour_a.intersection(neighbour_b)]
        # getting number of chat neighours of the all the neighbours of source and target
        all_n_c_n = [len(is_chat_neighbour.get(node, [])) for node in neighbour_a.union(neighbour_b)]

        # getting number of chat neighours of the all the mutual chat neighbours of source and target
        mutual_chat_n_c_n = [len(is_chat_neighbour.get(node, [])) for node in chat_neighbour_a.intersection(chat_neighbour_b)]

        # getting number of chat neighours of the all chat neighbours of source and target
        all_chat_n_c_n = [len(is_chat_neighbour.get(node, [])) for node in chat_neighbour_a.union(chat_neighbour_b)]

        # creating final variables and saving it to one file
        myfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                len(neighbour_a),
                len(neighbour_b),
                len(chat_neighbour_a),
                len(chat_neighbour_b),
                len(neighbour_a.intersection(neighbour_b)),
                len(chat_neighbour_a.intersection(chat_neighbour_b)),
                len(chat_neighbour_a.intersection(neighbour_a.intersection(neighbour_b))),
                len(chat_neighbour_b.intersection(neighbour_a.intersection(neighbour_b))),

                sum(na_n_c_n),
                len(na_n_c_n),
                sum(nb_n_c_n),
                len(nb_n_c_n),
                sum(mutual_n_c_n),
                len(mutual_n_c_n),
                sum(all_n_c_n),
                len(all_n_c_n),
                sum(mutual_chat_n_c_n),
                len(mutual_chat_n_c_n),
                sum(all_chat_n_c_n),
                len(all_chat_n_c_n),
        ))


all_neighbour = pd.read_csv('neigbours_vars.csv', header=None)

df_deg_2_neigh = pd.DataFrame()
df_deg_2_neigh['degree_2_neighs_chat_sum_source'] = all_neighbour[8]
df_deg_2_neigh['degree_2_neighs_chat_avg_source'] = all_neighbour[8]/all_neighbour[9]
df_deg_2_neigh['degree_2_neighs_chat_sum_target'] = all_neighbour[10]
df_deg_2_neigh['degree_2_neighs_chat_avg_target'] = all_neighbour[10]/all_neighbour[11]
df_deg_2_neigh['mutual_neighs_avg_chat_sum'] = all_neighbour[12]
df_deg_2_neigh['mutual_neighs_avg_chat_avg'] = all_neighbour[12]/all_neighbour[13]
df_deg_2_neigh['union_neighs_avg_chat_sum'] = all_neighbour[14]
df_deg_2_neigh['union_neighs_avg_chat_avg'] = all_neighbour[14]/all_neighbour[15]
df_deg_2_neigh.to_pickle('degree_2_neighbour_feats.pkl')

df_neighbours1 = all_neighbour[[4,6,7]]
df_neighbours1.to_csv('neigbours_vars_sahil_1.csv', index=False)

df_neighbours2 = all_neighbour[[2,3]]
df_neighbours2.to_csv('neigbours_vars_sahil_2.csv', index=False)

df_leftover = pd.DataFrame()
df_leftover[0] = all_neighbour[16]
df_leftover[1] = all_neighbour[16]/all_neighbour[17]
df_leftover[2] = all_neighbour[18]
df_leftover[3] = all_neighbour[18]/all_neighbour[19]
df_leftover.to_pickle('neigbours_vars_pat_leftover_2.pkl')
