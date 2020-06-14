# github link: https://github.com/ds-praveenkumar/kaggle
# Author: ds-praveenkumar
# file: forcasting/prepare_train_data.py/
# Created by ds-praveenkumar at 13-06-2020 15 34
# feature:

import os
import pandas as pd
import numpy as np
import click
from src.utility.timeit import timeit

root = os.path.dirname(os.getcwd())
train_data_path = os.path.join(root, 'data', 'training')
preprocess_data_path = os.path.join(root, 'data', 'preprocess')



@timeit
def prepare_train_data(prep_path, train_path):

    prep_df = pd.DataFrame(np.load(os.path.join(prep_path, 'sales_mat.npy')))
    prep_df = prep_df.T
    prep_df['ds'] =  pd.date_range(end='2016-06-19',periods=1913).values
    for column in prep_df.iloc[:,:30489]:
        train_items = prep_df[['ds',column]][-365:]
        train_items.rename(columns={column:'y'},inplace=True)
        save_at = os.path.join(train_path,f"{column}.csv")
        train_items.to_csv(save_at,index=False)
        print(f"file saved at {save_at}")



@click.command()
@click.argument('preprocess_data_path', type=click.Path(exists=True))
@click.argument('train_data_path', type=click.Path())
def main(preprocess_data_path, train_data_path):
    prepare_train_data(preprocess_data_path, train_data_path)


if __name__=='__main__':
    main()