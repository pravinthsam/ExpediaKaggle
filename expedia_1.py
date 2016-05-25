# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:19:44 2016

@author: Pravinth Samuel Vethanayagam
"""
import load_data
import random
from sklearn.decomposition import PCA
import pandas as pd
import time

start_time = time.time()

NUM_USERS = 100000
DEST_COMP = 3

def expand_dt(dt_df, attrs):
    colname = dt_df.columns[0]
    new_df = dt_df[[]]
    
    for attr in attrs:
        new_df[colname + '_' + attr] = dt_df[colname].apply(lambda x: x.__getattribute__(attr))
    
    return new_df

if __name__ == '__main__':
    trainreader, testreader = load_data.load_expdata_chunks()
    
    
    # Generating list of users
    setOfAllUsersTest = load_data.setOfAllUsers(testreader)
    setOfAllUsersTrain = load_data.setOfAllUsers(trainreader)
    setOfAllCommonUsers = setOfAllUsersTest.intersection(setOfAllUsersTrain)
    
    print 'Selecting', NUM_USERS, 'out of', len(setOfAllCommonUsers), 'users in the test dataset...'
    subsetOfUsers = random.sample(setOfAllCommonUsers, NUM_USERS)
    
    # Creating new dataset with limited users
    print 'Generating the new dataset with only the subset of users...'
    trainreader, testreader = load_data.load_expdata_chunks()
    traindf = load_data.subsetDataset(trainreader, subsetOfUsers)
    testdf = load_data.subsetDataset(testreader, subsetOfUsers)
    
    # Filtering dataset for only booking events
    traindf = traindf[traindf.is_booking==1]
    
    # Removing nan values
    traindf = traindf.dropna()
    testdf = testdf.dropna()
    
    # Most common clusters
    clusters_ordered = list(traindf.hotel_cluster.value_counts().index)
    most_common_clusters = clusters_ordered[:5]
    
    # Making a simple prediction
    # Find 5 most common clusters and give that as the solution for all the rows
    print 'Calculating results for a very simple solution where all answers are the same...'
    _, testreader = load_data.load_expdata_chunks()
    testdfAll = load_data.subsetDataset(testreader, setOfAllUsersTest)
    
    simple_solution = [most_common_clusters for i in range(len(testdfAll))]
    
    load_data.createSubmissionFile(testdfAll[['id']], simple_solution, 'results/first_submit.csv')
    
    '''# Simple rule for aggregating across orig_destination_distance and hotel_market
    print 'Calculating the results when aggregated over orig_destination_distance and hotel_market...'
    dist_market_map =  {}   
    for index, t in traindf.iterrows():
        type(t)
        key = (t.orig_destination_distance, t.hotel_market)
        
        if key not in dist_market_map.keys():
            dist_market_map[key] = {t.hotel_cluster:1}
        else:
            if t.hotel_cluster not in dist_market_map[key].keys():
                dist_market_map[key][t.hotel_cluster] = 1
            else:
                dist_market_map[key][t.hotel_cluster] = dist_market_map[key][t.hotel_cluster] + 1
    '''
    
    # Cluster the different destinations together     
    destinations = load_data.load_destinations()
    
    pca_dest = PCA(n_components=DEST_COMP)
    dest_small = pca_dest.fit_transform(destinations[['d{0}'.format(i) for i in range(1,150)]])
    dest_small = pd.DataFrame(dest_small)
    
    # TODO check how much data has been lost
    dest_small['srch_destination_id'] = destinations['srch_destination_id']
    dest_small_columns = ['dest_feat_{0}'.format(i) for i in range(1,DEST_COMP+1)]
    dest_small_columns.append('srch_destination_id')
    dest_small.columns = dest_small_columns
    
    # Convert dates and times
    dt_fields1 = ['year', 'month', 'day', 'dayofweek', 'hour']
    dt_fields2 = ['year', 'month', 'day', 'dayofweek']
    
    traindf['date_time'] = pd.to_datetime(traindf['date_time'])
    traindf['srch_ci'] = pd.to_datetime(traindf['srch_ci'])
    traindf['srch_co'] = pd.to_datetime(traindf['srch_co'])
    traindf = traindf.join(expand_dt(traindf[['date_time']], dt_fields1))
    traindf = traindf.join(expand_dt(traindf[['srch_ci']], dt_fields2))
    traindf = traindf.join(expand_dt(traindf[['srch_co']], dt_fields2))
    
    testdf['date_time'] = pd.to_datetime(testdf['date_time'])
    testdf['srch_ci'] = pd.to_datetime(testdf['srch_ci'])
    testdf['srch_co'] = pd.to_datetime(testdf['srch_co'])
    testdf = testdf.join(expand_dt(testdf[['date_time']], dt_fields1))
    testdf = testdf.join(expand_dt(testdf[['srch_ci']], dt_fields2))
    testdf = testdf.join(expand_dt(testdf[['srch_co']], dt_fields2))
    
    
        
    
    # Adding stay time
    traindf['stay_time'] = (traindf['srch_co'] - traindf['srch_ci']).astype('timedelta64[D]').astype(int)
    testdf['stay_time'] = (testdf['srch_co'] - testdf['srch_ci']).astype('timedelta64[D]').astype(int)
    
    # Attach destinations features
    traindf = traindf.join(dest_small, how='left', on='srch_destination_id', rsuffix='dest')
    testdf = testdf.join(dest_small, how='left', on='srch_destination_id', rsuffix='dest')
    
    #TODO remove extra srch_d_id column?
    
    # filling null values with -1
    traindf.fillna(-1, inplace=True)
    testdf.fillna(-1, inplace=True)
    
    
    # Random Forest
    predictors = [c for c in traindf.columns if c not in ['hotel_cluster', 'date_time', 'srch_ci', 'srch_co']]
            
    from sklearn import cross_validation
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
    scores = cross_validation.cross_val_score(clf, traindf[predictors], traindf['hotel_cluster'].apply(str), cv=3)
    
    
    
    
end_time = time.time()
print 'Program took', (end_time-start_time), 'seconds to run.'    