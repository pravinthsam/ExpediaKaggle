# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:19:44 2016

@author: Pravinth Samuel Vethanayagam
"""
import load_data
import random

NUM_USERS = 10000

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
    traindf_booking = traindf[traindf.is_booking==1]
    
    # Most common clusters
    clusters_ordered = list(traindf_booking.hotel_cluster.value_counts().index)
    most_common_clusters = clusters_ordered[:5]
    
    # Making a simple prediction
    _, testreader = load_data.load_expdata_chunks()
    testdfAll = load_data.subsetDataset(testreader, setOfAllUsersTest)
    
    commonHotel_cluster = ' '.join([str(c) for c in most_common_clusters])
    resultdf_simple = testdfAll[['id']]
    resultdf_simple['hotel_cluster'] = commonHotel_cluster
    resultdf_simple.to_csv('results/first_submit.csv', index=False)
    
    
    
    