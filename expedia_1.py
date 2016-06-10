# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:19:44 2016

@author: Pravinth Samuel Vethanayagam
"""
import load_data
import random
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import time

start_time = time.time()

NUM_USERS = 500000
DEST_COMP = 3

'''def expand_dt(dt_df, attrs):
    colname = dt_df.columns[0]
    new_df = dt_df[[]]
    
    for attr in attrs:
        new_df[colname + '_' + attr] = dt_df[colname].apply(lambda x: x.__getattribute__(attr))
    
    return new_df'''
    
'''def expand_dt(dt_df, attrs):
    colname = dt_df.columns[0]
    
    new_df = dt_df[colname].apply(lambda x: pd.Series([x.__getattribute__(attr) for attr in attrs]))
    new_df.columns = [colname + '_' + x for x in attrs]
    return new_df
'''

def expand_dt(dt_df, attrs):
    colname = dt_df.columns[0]
    print 'Expanding', colname, '...'
    
    def datesplitfunc(dt):
        return [dt.__getattribute__(attr) for attr in attrs]
        
    newarr = map(datesplitfunc, dt_df[colname])
    new_df = pd.DataFrame(newarr).astype('uint8')
    new_df.columns = [colname + '_' + x for x in attrs]
    new_df.index = dt_df.index
    
    return new_df
try:    
    if __name__ == '__main__':


        trainreader, testreader = load_data.load_expdata_chunks()
        
        
        # Generating list of users
        print 'Loading data sets...'
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
        # TODO Use click data as well
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
        simple_solution = None
        
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
        print 'Generating features from destinations...'
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
        print 'Adding features from date columns...'
        dt_fields1 = ['year', 'month', 'day', 'dayofweek', 'hour']
        dt_fields2 = ['year', 'month', 'day', 'dayofweek']
        
        def datetimeexpand(df1):
            df = df1.copy(True)
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
            df['srch_ci'] = pd.to_datetime(df['srch_ci'], errors='coerce')
            df['srch_co'] = pd.to_datetime(df['srch_co'], errors='coerce')
            df['date_time'].fillna(pd.to_datetime(0), inplace=True)
            df['srch_ci'].fillna(pd.to_datetime(0), inplace=True)
            df['srch_co'].fillna(pd.to_datetime(0), inplace=True)
            
            tmp_df = expand_dt(df[['date_time']], dt_fields1)
            df[tmp_df.columns] = tmp_df
            tmp_df = expand_dt(df[['srch_ci']], dt_fields2)
            df[tmp_df.columns] = tmp_df
            tmp_df = expand_dt(df[['srch_co']], dt_fields2)
            df[tmp_df.columns] = tmp_df       
            
            return df
        
        traindf = datetimeexpand(traindf)
        testdf = datetimeexpand(testdf)
        testdfAll = datetimeexpand(testdfAll)
        
        # Adding stay time
        traindf['stay_time'] = (traindf['srch_co'] - traindf['srch_ci']).astype('timedelta64[D]').astype(int)
        testdf['stay_time'] = (testdf['srch_co'] - testdf['srch_ci']).astype('timedelta64[D]').astype(int)
        testdfAll['stay_time'] = (testdfAll['srch_co'] - testdfAll['srch_ci']).astype('timedelta64[D]').astype(int)
        
        # TODO add cumulative data as features. Per User, Per cluster, Per destination id
            
        # Attach destinations features
        traindf = traindf.join(dest_small, how='left', on='srch_destination_id', rsuffix='dest')
        testdf = testdf.join(dest_small, how='left', on='srch_destination_id', rsuffix='dest')
        testdfAll = testdfAll.join(dest_small, how='left', on='srch_destination_id', rsuffix='dest')
        
        #TODO remove extra srch_d_id column?
        
        # filling null values with -1
        traindf.fillna(-1, inplace=True)
        testdf.fillna(-1, inplace=True)
        testdfAll.fillna(-1, inplace=True)
        
        # Random Forest
        print 'Using random forest on all the hotel clusters as classes...'
        predictors = [c for c in traindf.columns if c not in ['hotel_cluster', 'date_time', 'srch_ci', 'srch_co']]
                
        from sklearn import cross_validation
        from sklearn.ensemble import RandomForestClassifier
    
        clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
        scores = cross_validation.cross_val_score(clf, traindf[predictors], traindf['hotel_cluster'].apply(str), cv=3)
        
        # binary classifiers
        print 'Doing binary classification for each of the clusters...'
        from sklearn.cross_validation import KFold
        from itertools import chain
        import heapq
        
        all_tr_probs = []
        all_te_probs = [[] for x in range(len(testdfAll))]
        all_clusters = traindf['hotel_cluster'].unique()
        jx = 0
        
        for cluster in all_clusters:
            print jx
            jx = jx+1
            traindf['target'] = 0
            traindf['target'][traindf['hotel_cluster']==cluster] = 1
            
            predictors = [col for col in traindf.columns if col not in ['hotel_cluster', 'cnt', 'is_booking', 'date_time', 'srch_ci', 'srch_co', 'target']]
            
            tr_probs = []
            cv = KFold(len(traindf['target']), n_folds=10) 
            # TODO currently not doing random
            # TODO increase number of folds
            
            clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
            
            for ix, (tr_ix, te_ix) in enumerate(cv):
                clf.fit(traindf[predictors].iloc[tr_ix], traindf['target'].iloc[tr_ix])
                tr_preds = clf.predict_proba(traindf[predictors].iloc[te_ix])
                tr_probs.append([p[1] for p in tr_preds])
                
            tr_probs = chain.from_iterable(tr_probs)
            all_tr_probs.append(list(tr_probs))
            
            te_preds = clf.predict_proba(testdfAll[predictors])
            te_probs = [p[1] for p in te_preds]
            
            if jx <= 5:
                for (kx, val) in enumerate(te_probs):
                    heapq.heappush(all_te_probs[kx], (val, cluster))
            else:
                for (kx, val) in enumerate(te_probs):
                    if all_te_probs[kx][0][0] < val:
                        heapq.heappop(all_te_probs[kx])
                        heapq.heappush(all_te_probs[kx], (val, cluster))
                    
        binary_preds = []
        binary_preds_probs = []
        binary_preds_confidence = []        
            
        for te in all_te_probs:
            te_sorted = sorted(te, reverse=True)
            binary_preds.append([t[1] for t in te_sorted])
            binary_preds_probs.append([t[0] for t in te_sorted])
            binary_preds_confidence.append(np.sum([t[0] for t in te_sorted]))
        
        '''all_te_probs = np.array(all_te_probs).T
        
        
        
        for row in all_te_probs:
            row_indices = list((-row).argsort()[:5])
            binary_preds.append(all_clusters[row_indices])
            binary_preds_probs.append(row[row_indices])
            binary_preds_confidence.append(row[row_indices].sum())
        '''
        load_data.createSubmissionFile(testdfAll[['id']], binary_preds, 'results/binary_predictions_submit.csv')
finally:
    end_time = time.time()
    print 'Program took', (end_time-start_time), 'seconds to run.'    