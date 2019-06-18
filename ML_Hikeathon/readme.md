Solution Overview:

    Our solution heavily depends on negative under-sampling, which means we use all positive examples (i.e., is_chat== 1) and down-sampled negative examples on model training. It discards about 95% of negative examples, but we didn't see much performance deterioration when we tested with our initial features. Moreover, we could get better performance when creating a submission by bagging ten predictors trained on ten sampled datasets created from different random seeds. This technique allowed us to experiment with multiple features while keeping LGB training time reasonable. 

Feature Engineering:

    The final dataset consisted of three broad categories of features, which are as follows: 
    
        ● User activity features: 
            ○ We were given a user activity matrix in 13 dimensions, which was used to create metrics on similarity of users based on activity 
        ● Graph Features: 
            ○ We created three graphs from the data set: 
                i. Undirected Contact Graph ii. Directed Contact Graph iii. Chat Graph (capturing the data with is_chat=1) 
            ○ Some of the metrics used included: 
                i. Jaccard coefficient ii. Resource allocation index iii. Degrees of nodes(in case of directed graph, in/out degrees) 
        ● User Social Circle Activity: 
            ○ These variables proved to the most crucial for boosting model performance 
            ○ Number of mutual nodes was calculated between the node pairs. Higher this number, more the chances they interact 
            ○ With how many mutual nodes does each node pair interact? Let’s take an example: 
                i. We have to find the chat probability between A&B. X,Y,Z are the mutual 
                contacts between A&B 
                ii. Now if A&B are chatting with all three then there is a higher chance they 
                will chat with each other
                iii. If they are not chatting with anyone, then it’s highly likely that X,Y,Z are 
                customer care numbers :p 
            ○ How many time is each node involved in a chat? For A and B to have a higher chance of chatting, they need to be chatting with other people too. 
            ○ How chatty is the neighbourhood?If this number is high for both the nodes, then it indicates they are part of a more talkative neighbourhood and hence higher the chances of chatting. 
            ○ Inverse Links, this features captured the inverse relationships present in the data shared across train and test. 

Final Model:

    ● The final model is an ensemble of 10 LightGBM classifiers with each model fitted over a five-fold random stratified CV. 
    ● Each of the above model was built on a different subset of data with negative undersampling mentioned above. 
    ● A little to no effort was made on tuning the hyper-parameters and more focus was on creating valuable/meaningful features. 


Our VM Specs while working out these steps:

    Ram - 156 GB
    CPU - 32 cores
    Min free hard disk space - 100GB

Approx Run Time - 3 days

Environment Setup:

    Create Virtual Environment with Python- 2.7.15

Move train.csv, user_features.csv, test.csv to code directory and then run the script main_script.sh
