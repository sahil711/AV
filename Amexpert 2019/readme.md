Brief Approach:

    Crux of the problem lied in creating feature such that you don’t introduce leak in them. Most of the
    participants were struggling with this. Apart from this feature creation was the key. There was a lot of
    scope to create features. I have my feature creation approach below.

Which data-preprocessing / feature engineering ideas really worked?

    For a given customer coupon pair, find all the items for that coupon and find the propensity of the
    customer to buy that product using the historical transactions.Similar logic can be applied at a brand and
    category level. Apart from propensity, we can also calculate a bunch of other variables for example,
    what is the total amount spent,total discount availed etc.
    Apart from customer coupon level, it tried similar approaches at only coupon and only campaign level.
    For example, what is the probability that items under one coupon code use coupon discount(you can get
    his by filtering historical transactions for the items belonging to one coupon)
    To avoid leakage it was important to filter the historical data prior to the campaign date at a row level.
    These variables were the most powerful ones for this problem and I got a clue from reading the problem
    statement page multiple times. If you had read the last 4 lines of the problem statement, where in the
    entire process was described in 3 points, the ideas would have clicked.

Final Model Description:

    My Final model is a linear blend of Catboost, Lightgbm and Xgboost built over 2 datasets. The only
    difference between the two datasets was the number of features each had. One had more number of
    statistics being calculated from the historical data.



Steps to replicate:

All the required packages are in requirements.txt file

The code is divided into two folders, one for feature creation and other for modelling.
First we run the feature creation scripts. Order is as follows:

    1) feature_creation_code/data_prep_v2.ipynb
    2) feature_creation_code/fc_ideas.ipynb
    3) feature_creation_code/feats_user_coup_item.ipynb
    4) feature_creation_code/feats_user_coup_item_brand_feats.ipynb

This is will create 12 csvs in the current folder. 

Now we will run the modelling script. Order is as follows:
  
    1) modelling/modelling_adding_camp_cust_more_feats.ipynb
    2) modelling/modelling_adding_camp_cust_more_feats-xgb.ipynb
    3) modelling/modelling_adding_camp_cust_more_feats-CATBOOST.ipynb
    4) modelling/modelling_adding_camp_cust_more_feats_v2.ipynb
    5) modelling/modelling_adding_camp_cust_more_feats_v2_xgb.ipynb
    6) modelling/modelling_adding_camp_cust_more_feats_v2-CATBOOST.ipynb
    7) modelling/modelling_adding_camp_cust_more_feats_group_k_camp_xgb_lgb.ipynb
    8) modelling/modelling_adding_camp_cust_more_feats_group_k_camp_xgb_lgb_v2.ipynb
    9) modelling/modelling_adding_camp_cust_more_feats_group_k_camp_CATBOOST.ipynb
    10) modelling/modelling_adding_camp_cust_more_feats_group_k_camp_v2-CATBOOST.ipynb
    11) modelling/ensembling.ipynb

final_sub_2.csv will contain the final submission

