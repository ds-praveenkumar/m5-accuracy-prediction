# github link: https://github.com/ds-praveenkumar/kaggle
# Author: ds-praveenkumar
# file: forcasting/generate_submission.py/
# Created by ds-praveenkumar at 13-06-2020 19 39
# feature:

import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.utility.timeit import timeit

ROOT_DIR = Path(__file__).parent.parent.parent
print('ROOT_DIR:', ROOT_DIR)

@timeit
def get_map():
    path = os.path.join(ROOT_DIR,'data','preprocessed','idmap.csv')
    df = pd.read_csv(path,names=['idx','id'])
    return df

@timeit
def get_submission():
    path = os.path.join(ROOT_DIR,'data','raw','sample_submission.csv')
    df = pd.read_csv(path)
    return df

@timeit
def prepare_submission():
    df_list = []
    path = os.path.join(ROOT_DIR,'data','predictions')
    walk_dir = os.walk(path)
    for root,_,files in walk_dir:
        for file in files[:10]:
            print('file:',file)
            df = pd.read_csv(os.path.join(root,file)).T
            df['idx'] = file.split('.')[0].split('_')[1]
            df_list.append(df)
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    return merged_df

def main():
    map_df = get_map()
    sub_df = get_submission()
    map_df = map_df.merge(sub_df,on='id',how='inner')
    print('map_df:',map_df.dtypes,end='\n')
    #print(map_df.tail(),end='\n')
    merged_df = prepare_submission()
    merged_df['idx'] = merged_df['idx'].astype(np.int64)
    print('merged:',merged_df.dtypes)
    submission = merged_df.merge(map_df,on='idx',how='left').fillna(0)
    print(submission.head())


if __name__ == '__main__':
    main()