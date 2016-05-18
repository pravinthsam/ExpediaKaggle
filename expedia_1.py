# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:19:44 2016

@author: Pravinth Samuel Vethanayagam
"""
import load_data
import random

NUM_USERS = 50000

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
    
    # Simple rule for aggregating across orig_destination_distance and hotel_market
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
        
    
    
    
    