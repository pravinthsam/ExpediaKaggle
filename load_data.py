# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:46:28 2016

@author: Pravinth Samuel Vethanayagam
"""

import os
import gzip
import pandas as pd

CHUNK_SIZE = 1000000

def unzip_datasets(folderpath = './data/'):
    listOfFiles = ['destinations.csv', 'sample_submission.csv', 'test.csv', 'train.csv']
    
    # unzip all files
    for filename in listOfFiles:
        if not os.path.isfile(folderpath + filename):
            print 'Extracting', filename, '...'
            with gzip.open(folderpath + filename + '.gz', "rb") as inF:
                outF = open(folderpath + filename, 'wb')
                outF.write( inF.read() )
                inF.close()
                outF.close()
                
def load_expdata_chunks(folderpath = './data/'):
    unzip_datasets(folderpath)
    
    trainreader = pd.read_csv(folderpath + 'train.csv', chunksize=CHUNK_SIZE)
    testreader = pd.read_csv(folderpath + 'test.csv', chunksize=CHUNK_SIZE)
    
    return trainreader, testreader
    
def setOfAllUsers(df_reader):
    
    setAll = set([])
    for chunk in df_reader:
        setAll = setAll.union(set(chunk.user_id))
    
    return setAll
    
def subsetDataset(df_reader, subsetUsers):
    df = None
    
    for chunk in df_reader:
        if df is None:
            df = chunk[0:0]
        print len(df)
        df = df.append(chunk[chunk.user_id.isin(subsetUsers)])
        
    return df

def createSubmissionFile(idList, allClusters, filename = 'results/model_results.csv'):
    results = idList
    results['hotel_cluster'] = [ ' '.join([str(c) for c in clusters]) for clusters in allClusters]
    results.to_csv(filename, index=False)
    
          
if __name__ == '__main__':
    unzip_datasets()