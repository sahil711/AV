print("****************** train_script.py *******************")

import pandas as pd
import numpy as np
import copy
import pickle
import sys
import gc
from lightgbm import LGBMClassifier
from custom_estimator import Estimator
from sklearn.externals import joblib

# function to create more variables
def create_feats(df):
    df.degree_source=df.degree_source.astype('float32')
    df.degree_target=df.degree_target.astype('float32')

    df['degree_ratio']=df.degree_source/(1+df.degree_target)
    df['degree_delta']=df.degree_source-df.degree_target

    df['directed_degree_ratio']=df.directed_degree_source/(1+df.directed_degree_target)
    df['directed_degree_delta']=df.directed_degree_source-df.directed_degree_target

    df['directed_degree_ratio_in']=df.directed_degree_source_in/(1+df.directed_degree_target_in)
    df['directed_degree_delta_in']=df.directed_degree_source_in-df.directed_degree_target_in

    df['directed_degree_ratio_out']=df.directed_degree_source_out/(1+df.directed_degree_target_out)
    df['directed_degree_delta_out']=df.directed_degree_source_out-df.directed_degree_target_out

    df['node_sum']=df['node1_id']+df['node2_id']
    df['node_ratio']=(df['node1_id']/df['node2_id']).astype('float16')

    df['is_chat_diff']=df['source_is_chat_count']-df['target_is_chat_count']
    df['is_chat_ratio']=(df['source_is_chat_count']/(df['target_is_chat_count']+1).astype('float16'))

    df['mutual_chat_diff']=df['source_mutual_is_chat']-df['target_mutual_is_chat']
    df['mutual_chat_ratio']=(df['source_mutual_is_chat']/(df['target_mutual_is_chat']+1).astype('float16'))


    df['delta_triangle']=df['node1_triangles']- df['node2_triangles']
    df['ratio_triangle']=df['node1_triangles']/(1+df['node2_triangles'])
    df['triangle_degree_delta_source']=(df.degree_source*8264276).astype('int16')-df['node1_triangles']
    df['triangle_degree_delta_target']=(df.degree_target*8264276).astype('int16')-df['node1_triangles']

    df['clust_prod']=df['node1_cluster']* df['node2_cluster']
    df['clust_diff']=df['node1_cluster']- df['node2_cluster']

    df['source_net_act']=df[['f1_source_target', 'f2_source_target', 'f3_source_target', 'f4_source_target', 'f5_source_target',
     'f6_source_target', 'f7_source_target', 'f8_source_target', 'f9_source_target', 'f10_source_target',
     'f11_source_target', 'f12_source_target']].sum(axis=1)

    df['target_net_act']=df[['f1_target', 'f2_target', 'f3_target',
     'f4_target', 'f5_target', 'f6_target', 'f7_target', 'f8_target', 'f9_target', 'f10_target', 'f11_target',
     'f12_target']].sum(axis=1)

    df['net_act_diff']=df['source_net_act']- df['target_net_act']

    df['f14_source']=df['f1_source_target']+df['f4_source_target']+df['f7_source_target']+df['f10_source_target']
    df['f15_source']=df['f5_source_target']+df['f8_source_target']+df['f11_source_target']
    df['f16_source']=df['f6_source_target']+df['f9_source_target']+df['f12_source_target']

    df['f14_target']=df['f1_target']+df['f4_target']+df['f7_target']+df['f10_target']
    df['f15_target']=df['f5_target']+df['f8_target']+df['f11_target']
    df['f16_target']=df['f6_target']+df['f9_target']+df['f12_target']

    df['fdiff_1']=df['f1_source_target']-df['f1_target']
    df['fdiff_2']=df['f2_source_target']-df['f2_target']
    df['fdiff_3']=df['f3_source_target']-df['f3_target']
    df['fdiff_4']=df['f4_source_target']-df['f4_target']
    df['fdiff_5']=df['f5_source_target']-df['f5_target']
    df['fdiff_6']=df['f6_source_target']-df['f6_target']
    df['fdiff_7']=df['f7_source_target']-df['f7_target']
    df['fdiff_8']=df['f8_source_target']-df['f8_target']
    df['fdiff_9']=df['f9_source_target']-df['f9_target']
    df['fdiff_10']=df['f10_source_target']-df['f10_target']
    df['fdiff_11']=df['f11_source_target']-df['f11_target']
    df['fdiff_12']=df['f12_source_target']-df['f12_target']
    df['fdiff_13']=df['f13_source_target']-df['f13_target']
    df['fdiff_14']=df['f14_target']-df['f14_source']
    df['fdiff_15']=df['f15_target']-df['f15_source']
    df['fdiff_16']=df['f16_target']-df['f16_source']

    df['fmult_1']=df['f1_source_target']/(df['f1_target']+1).astype('float16')
    df['fmult_2']=df['f2_source_target']/(1+df['f2_target']).astype('float16')
    df['fmult_3']=df['f3_source_target']/(1+df['f3_target']).astype('float16')
    df['fmult_4']=df['f4_source_target']/(1+df['f4_target']).astype('float16')
    df['fmult_5']=df['f5_source_target']/(1+df['f5_target']).astype('float16')
    df['fmult_6']=df['f6_source_target']/(1+df['f6_target']).astype('float16')
    df['fmult_7']=df['f7_source_target']/(1+df['f7_target']).astype('float16')
    df['fmult_8']=df['f8_source_target']/(1+df['f8_target']).astype('float16')
    df['fmult_9']=df['f9_source_target']/(1+df['f9_target']).astype('float16')
    df['fmult_10']=df['f10_source_target']/(1+df['f10_target']).astype('float16')
    df['fmult_11']=df['f11_source_target']/(1+df['f11_target']).astype('float16')
    df['fmult_12']=df['f12_source_target']/(1+df['f12_target']).astype('float16')
    df['fmult_13']=df['f13_source_target']/(1+df['f13_target']).astype('float16')
    df['fmult_14']=df['f14_source']/(1+df['f14_target']).astype('float16')
    df['fmult_15']=df['f15_source']/(1+df['f15_target']).astype('float16')
    df['fmult_16']=df['f16_source']/(1+df['f16_target']).astype('float16')

    df['norm_user_diff']=np.sqrt(np.square(df[df.columns[df.columns.str.contains('diff')]].astype('int16')).sum(axis=1))

    df['norm_user_diff_1']=np.sqrt(np.square(train[['fdiff_1','fdiff_4','fdiff_7','fdiff_10']].astype('int16')).sum(axis=1))
    df['norm_user_diff_2']=np.sqrt(np.square(train[['fdiff_2','fdiff_5','fdiff_8','fdiff_11']].astype('int16')).sum(axis=1))
    df['norm_user_diff_3']=np.sqrt(np.square(train[['fdiff_3','fdiff_6','fdiff_9','fdiff_12']].astype('int16')).sum(axis=1))

    df['source_net_act']=df['source_net_act'].astype('int16')
    df['target_net_act']=df['target_net_act'].astype('int16')
    df['net_act_diff']=df['net_act_diff'].astype('int16')
    df['norm_user_diff']=df['norm_user_diff'].astype('float16')

    return df

# reading all the files and merging it with train dataset
train=pd.read_pickle('freq_new_train.pkl')
clust=pd.read_pickle('cluster_coeffs.pkl')
triangles=pd.read_pickle('triangles.pkl')
train['node1_cluster']=clust.clust_source.iloc[:train.shape[0]].values
train['node2_cluster']=clust.clust_target.iloc[:train.shape[0]].values
train['node1_triangles']=triangles.triangles_source.iloc[:train.shape[0]].values
train['node2_triangles']=triangles.triangles_target.iloc[:train.shape[0]].values

temp_df=pd.read_pickle('neigbours_vars_pat_leftover_2.pkl')
temp_df.columns=['deg2_feat1','deg2_feat2','deg2_feat3','deg2_feat4']
train=pd.concat((train,temp_df.iloc[:train.shape[0],:]),axis=1)

deg_2_neigh=pd.read_pickle('degree_2_neighbour_feats.pkl')
train=pd.concat((train,deg_2_neigh.iloc[:train.shape[0],:]),axis=1)

dir_degrees=pd.read_pickle('directed_degrees.pkl')
train['directed_degree_source']= dir_degrees['directed_degree_source'].iloc[:train.shape[0]].values
train['directed_degree_target']= dir_degrees['directed_degree_target'].iloc[:train.shape[0]].values
train['directed_degree_source_in']= dir_degrees['directed_degree_source_in'].iloc[:train.shape[0]].values
train['directed_degree_target_in']= dir_degrees['directed_degree_target_in'].iloc[:train.shape[0]].values
train['directed_degree_source_out']= dir_degrees['directed_degree_source_out'].iloc[:train.shape[0]].values
train['directed_degree_target_out']= dir_degrees['directed_degree_target_out'].iloc[:train.shape[0]].values

neighbours=pd.read_csv('neigbours_vars_sahil_1.csv')
train['source_mutual_is_chat']=neighbours.iloc[:train.shape[0],1].values
train['target_mutual_is_chat']=neighbours.iloc[:train.shape[0],2].values
train['mutual_neighbours']=neighbours.iloc[:train.shape[0],0].values

neighbours2=pd.read_csv('neigbours_vars_sahil_2.csv')
train['source_is_chat_count']=neighbours2.iloc[:train.shape[0],0].values
train['target_is_chat_count']=neighbours2.iloc[:train.shape[0],1].values

def change_dtype(a,dt):
    return a.astype(dt)

for col in train.columns[train.columns.str.contains('f[0-9]')].tolist():
    train[col]=change_dtype(train[col],'int16')

extra_feats=pd.read_csv('jc_rsa_pa_aai.csv')
train['jc']= extra_feats.jc.iloc[:train.shape[0]].values
train['rsa']= extra_feats.rsa.iloc[:train.shape[0]].values
train['pa']= extra_feats.pa.iloc[:train.shape[0]].values
train['adamic_adar']= extra_feats.aa.iloc[:train.shape[0]].values


leak_feature=pd.read_pickle('leak_feature.pkl')
train['leak_feature']=leak_feature.leak_feature.iloc[:train.shape[0]].values

# function for negative undersampling
def neg_sample_estimator(train_df, est, seed, frac=0.05):
    temp_train=train_df[train_df.is_chat==0].sample(frac=frac,random_state=seed)
    df=pd.concat((temp_train,train_df[train_df.is_chat==1]),axis=0)
    df=df.sample(frac=1,random_state=seed)
    y=df.is_chat
    df.drop('is_chat',inplace=True,axis=1)
    gc.collect()
    est.get_repeated_out_of_folds(df.values,y.values)
    return est

train = create_feats(train)

#model params
params={'num_leaves': 256, 'n_jobs': -1, 'colsample_bytree': 0.8,
        'learning_rate': 0.1, 'min_child_weight': 200.0, 'n_estimators': 1000,
        'subsample': 1.0, 'objective': 'binary', 'bagging_freq': 5, 'boosting_type': 'gbdt'}

mod = Estimator(model=LGBMClassifier(**params),n_jobs=-1,n_splits=5,random_state=100,shuffle=True,early_stopping_rounds=100)

# fitting models with different seeds
for i in range(10):
    random_state = 100*(i+1)
    mod2 = neg_sample_estimator(est=mod, frac=0.05, seed=random_state, train_df=train)
    print 'mod2', mod2.cv_scores, mod2.avg_cv_score
    joblib.dump(mod2, 'model_{}.pkl'.format(random_state))
