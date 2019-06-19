Link: https://datahack.analyticsvidhya.com/contest/club-mahindra-dataolympics/

Result:
    
    Private LB Score: 95.8128335535(100* RMSE) 
    Private LB Rank: 1st
    Public LB Score: 94.8817701988(100* RMSE) 
    Public LB Rank: 1st
    
Feature Engineering: 

● In this dataset groupby variables added a lot of value. Intuition for this was based on seeing how the model extracted value from the raw data set i.e. without any engineered feature. On raw features, feature importance threw out member_id(encoded) and resort_id(encoded) as two of the most important features. 

● Along with these two if you, it was important to consider the effect of holidays/non-holidays. To capture this number of visits on a particular day was used. Higher this value, more likely it is that the day was a holiday. 

● To utilise the member_id to full extent, two sets of features were created: 
    
    ○ Temporal - these capture trends across visits: 
        ■ Lead/Lag features: 
            ● Days since last visit 
            ● Days for next visit 
        ■ Rolling avg of numerical features like num_of_adults, num_of_children etc. 
        ■ Visit Number 
        ■ And many more 
    ○ Non Temporal: 
        ■ Average, min and max values of numerical features at a member level
    
● These variables essentially captured the user level characteristics and added the most value 

● Similar to what we created for member_id, same activity was carried out for resort_id. Again the idea was to introduce as much info about the resort as possible. Some of the features which capture this are: 

    1) Total number of visits to the resort 
    2) Total number of visits to the resort on a particular day 
    3) Total number of people visiting the resort 
    4) Avg room nights booked at the resort 
    
● Resort level variables very not as important as the user level variables but they added some value 

● I also created similar variables for state variables, but they did not add a lot of value. Intuition was again the same to extract maximum info about the state. 

● Apart from these group-by variables, there were many interaction level variables too. For example, ratio of adults to children, ratio of adult to room-nights etc. 

● Finally, I was able to create two data set from these exercise, one with 118 features and the other with 946 features. 

Models:

CV strategy: Random 5 fold Split was used 

    ● Model1: Lightgbm over 118 dataset; CV~96.43; LB~95.07 
    ● Model 2: Lightgbm over 964 dataset; CV~96.16; LB~94.93 
    ● Model 3: Catboost over 118 dataset; CV~96.31; LB~95.09 
    ● Meta Model: Linear Regression; CV~95.99; LB~94.88 Each of the above models were built over 2 seed and the final submission was the average of the predictions of the meta model over two seeds. 


VM Specs while working out these steps

    Ram - 60 GB    
    CPU - 24 cores
    Min hard disk space - 50GB

Environment Setup:

    Install Anaconda Version - 4.5.2
    Create Virtual Environment with Python- 2.7.15 and do pip install -r requirement.txt to install all the dependencies 

Execution Steps:

    Run the following notebooks in order (Files generated from one might be used in notebooks that follows later)

    1. basic_data_processing.ipynb
        Run time - approx 5 mins 

    2. lgb_946_model_building.ipynb
        Run time - approx 90 mins

    3. sahil_1200_catboost.ipynb
        Run time - approx 20 mins

    4. lgbm_118_feats.ipynb
        Run time - approx 10 mins

    5. catboost_118_feats.ipynb
        Run time - approx 8 hours

    6. ensembling.ipynb
        Run time - approx 10 mins
    
    
Generates 'final_sub_2.csv' as the final output
