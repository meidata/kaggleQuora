#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:05:04 2017

@author: meiyi
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from tqdm import tqdm



locations = pd.read_csv(path_data + "cities.csv")


def locationMatching(df,file,city=True):
    
    if city:
        location_type = '_city_'
        cities = set(locations['City'].dropna(inplace=False).values.tolist())
        all_cities = cities
        regex = "|".join(set(all_cities))
        
    else:
        location_type = '_country_'
        country = set(locations['Country'].dropna(inplace=False).values.tolist())
        all_country = country
        regex_c = "|".join(sorted(set(all_country)))
        
        
    results = []
    
    for index, row in tqdm(df.iterrows()):
        
        q1 = str(row['question1'])
        q2 = str(row['question2'])
    
        rr = {}
    
        q1_matches = []
        q2_matches = []
        
        if (len(q1) > 0):
            for i in re.findall(regex, q1, flags=re.IGNORECASE):
                 if i != ' ':
                     print(i.lower())
               
#        
#    
#        if (len(q1) > 0):
#            q1_matches = [i.lower() for i in re.findall(regex, q1, flags=re.IGNORECASE)]
#            
#            
#    
#        if (len(q2) > 0):
#            q2_matches = [i.lower() for i in re.findall(regex, q2, flags=re.IGNORECASE)]
#    
#        rr['z_q1_place_num'] = len(q1_matches)
#        rr['z_q1_has_place'] =len(q1_matches) > 0
#    
#        rr['z_q2_place_num'] = len(q2_matches) 
#        rr['z_q2_has_place'] = len(q2_matches) > 0
#    
#        rr['z_place_match_num'] = len(set(q1_matches).intersection(set(q2_matches)))
#        rr['z_place_match'] = rr['z_place_match_num'] > 0
#    
#        rr['z_place_mismatch_num'] = len(set(q1_matches).difference(set(q2_matches)))
#        rr['z_place_mismatch'] = rr['z_place_mismatch_num'] > 0
#    
#        results.append(rr)     
#    
#    
#    out_df = pd.DataFrame.from_dict(results)
#    out_df.to_pickle(path_feature + str(file)+location_type +'_location_matching.pkl')
    
    
    
locationMatching(train_data,'train',True)
locationMatching(test_data,'test')


train_location = pd.read_pickle(path_feature +'train_location_matching.pkl')
test_location = pd.read_pickle(path_feature +'test_location_matching.pkl')

for i in re.findall(regex, train_data.question1[1], flags=re.IGNORECASE):
    print(train_data.question1[1][i])
    


    