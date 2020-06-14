# -*- coding: utf-8 -*-
import os

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

from src.utility.timeit import timeit

@timeit
def load_data(data_path: str):
    """
        reads .csv files from data folder
    """
    logger = logging.getLogger(__name__)
    logger.info('loading from raw data')
    df_dict = {}
    if os.path.exists(data_path):

        walk_dir = os.walk(data_path)

        for root, _, files in walk_dir:
            for file in files:
                if file.endswith('.csv'):
                    try:
                        path = os.path.join(root, file)
                        filename = file.split('.')[0]
                        df = pd.read_csv(path)
                        df_dict[filename + '_df'] = df
                        print(f"file:{file}    shape:{df.shape}")
                    except Exception as e:
                        print(e)

        logger.info(f"file loading finished ")

    else:
        raise (FileNotFoundError, PermissionError)

    for names in df_dict.keys():
        if len(df_dict[names].columns) < 20:
            print(f"{names}: {df_dict[names].columns}")
    return df_dict



@timeit
def split_merge(df_dict):
    """
        Prepares data for by merging the Data frames from different .CSV
    :args: dictionary of data frames
    :return: Merged Data frame
    """
    logging.info("inside Split_merge()")
    try:
        sell_df = df_dict['sell_prices_df']
        sales_df = df_dict['sales_train_validation_df']
        cal_df = df_dict['calendar_df']
        print(f"cal_df max_date:{cal_df.date.max()}, min_date:{cal_df.date.min()}\n")
        sell_df['id'] = sell_df['item_id'] + '_' + sell_df['store_id'] + '_' + 'validation'
        #print('cal_sell_df\n',cal_sell_df)
        sell_df = pd.DataFrame(sell_df[['item_id', 'sell_price','id']].groupby(['item_id'])\
                               ['sell_price'].sum())
        print('sell_df\n', sell_df.head())
        sales_df = sell_df.merge(sales_df, on='item_id', how='left')
        print('sales_df\n',sales_df.head())
        days_mat = sales_df.iloc[:,7:].values
        sell_price = sales_df.sell_price.values
        sales_mat =  days_mat * sell_price.reshape((-1,1))
        sales_mat = sales_mat.T
        print('sales_mat:\n', sales_mat)
        print(f"sales_mat shape: {sales_mat.shape}")
        path_to_save = os.path.join(project_dir, 'data', 'preprocessed', 'sales_mat.npy')
        sell_df.T.to_csv(os.path.join(project_dir, 'data', 'preprocessed', 'sell.csv'), index=True)

    except KeyError:
        raise KeyError
    except MemoryError:
        raise MemoryError
    except ValueError:
        raise ValueError

    else:
        #print(f"sales_df 5 columns: {sales_df.columns.values[:6]}")

        np.save(path_to_save, sales_mat.T)

    return sales_mat






@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df_dict = load_data(input_filepath)
    split_merge(df_dict)
    logging.info(F"Data Processing complete. ")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
