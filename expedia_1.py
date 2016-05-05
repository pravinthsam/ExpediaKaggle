# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:19:44 2016

@author: Pravinth Samuel Vethanayagam
"""
import load_data
import random

if __name__ == '__main__':
    trainreader, testreader = load_data.load_expdata_chunks()
    
    
    # Generating list of users
    NUM_USERS = 100
    
    setOfAllUsers = set([])
    for chunk in trainreader:
        setOfAllUsers = setOfAllUsers.union(set(chunk.user_id))
        
    setOfUsers = random.sample(setOfAllUsers, NUM_USERS)
    
    # Creating new dataset with limited users
    trainreader, _ = load_data.load_expdata_chunks()
    traindf = chunk[0:0]
    cnt = 0
    for chunk in trainreader:
        traindf.add(chunk[chunk.user_id.isin(setOfUsers)])
        cnt = cnt+1
        if cnt>5:
            break
    
    