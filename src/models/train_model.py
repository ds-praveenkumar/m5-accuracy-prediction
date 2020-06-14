# github link: https://github.com/ds-praveenkumar/kaggle
# Author: ds-praveenkumar
# file: forcasting/train_model.py/
# Created by ds-praveenkumar at 13-06-2020 18 16
# feature:

import os

from  pathlib import Path
import pandas as pd
import pickle
import  numpy as np

from src.utility.timeit import timeit

ROOT_DIR = Path(__file__).parent.parent.parent
print('ROOT_DIR:', ROOT_DIR)

@timeit
def load_model():
    model_path = os.path.join(ROOT_DIR,'models','prophet_1.0.pkl')
    model = pickle.load(open(model_path,mode='rb'))
    return model

def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0
@timeit
def train_model():

    train_data_path = os.path.join(ROOT_DIR,'data','training')
    walk_dir = os.walk(train_data_path)

    for root,_,files in walk_dir:
        for file in files:
            try:
                id = file.split('.')[0]
                df = pd.read_csv(os.path.join(root,file))
                df['nfl_sunday'] = df['ds'].apply(nfl_sunday)
                df['y'] = np.log1p(df.y.astype(float) + 1)
                model = load_model()
                model.fit(df)
                future = model.make_future_dataframe(periods=28)
                future['nfl_sunday'] = future['ds'].apply(nfl_sunday)
                forecast = model.predict(future)
                path = os.path.join(ROOT_DIR,'data','predictions','pred_'+id+'.csv')
                forecast.yhat[-28:].to_csv(path,index=False)
                print('Prediction saved at: ',path)
            except Exception as e:
                print("Exception:",e)
                pass



def main():
    train_model()


if __name__ == '__main__':
    main()