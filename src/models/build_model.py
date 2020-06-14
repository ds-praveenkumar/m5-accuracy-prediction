# github link: https://github.com/ds-praveenkumar/kaggle
# Author: ds-praveenkumar
# file: forcasting/build_model.py/
# Created by ds-praveenkumar at 13-06-2020 02 09
# feature:

import os
import psutil
from fbprophet import Prophet
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from src.utility.timeit import timeit

ROOT_DIR = Path(__file__).parent.parent.parent
print('ROOT_DIR:', ROOT_DIR)

@timeit
def us_public_holidays():
    ny = pd.DataFrame({'holiday': "New Year's Day", 'ds': pd.to_datetime(['2016-01-01', '2017-01-01'])})
    mlk = pd.DataFrame(
        {'holiday': 'Birthday of Martin Luther King, Jr.', 'ds': pd.to_datetime(['2016-01-18', '2017-01-16'])})
    wash = pd.DataFrame({'holiday': "Washington's Birthday", 'ds': pd.to_datetime(['2016-02-15', '2017-02-20'])})
    mem = pd.DataFrame({'holiday': 'Memorial Day', 'ds': pd.to_datetime(['2016-05-30', '2017-05-29'])})
    ind = pd.DataFrame(
        {'holiday': 'Independence Day', 'ds': pd.to_datetime(['2015-07-04', '2016-07-04', '2017-07-04'])})
    lab = pd.DataFrame({'holiday': 'Labor Day', 'ds': pd.to_datetime(['2015-09-07', '2016-09-05', '2017-09-04'])})
    col = pd.DataFrame({'holiday': 'Columbus Day', 'ds': pd.to_datetime(['2015-10-12', '2016-10-10', '2017-10-09'])})
    vet = pd.DataFrame({'holiday': "Veteran's Day", 'ds': pd.to_datetime(['2015-11-11', '2016-11-11', '2017-11-11'])})
    thanks = pd.DataFrame({'holiday': 'Thanksgiving Day', 'ds': pd.to_datetime(['2015-11-26', '2016-11-24'])})
    christ = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2015-12-25', '2016-12-25'])})
    inaug = pd.DataFrame({'holiday': 'Inauguration Day', 'ds': pd.to_datetime(['2017-01-20'])})
    us_public_holidays = pd.concat([ny, mlk, wash, mem, ind, lab, col, vet, thanks, christ, inaug])
    return us_public_holidays


def is_nfl_season(ds):
    date = pd.to_datetime(ds)
    return (date.month > 8 or date.month < 2)


def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0

@timeit
def build_model():
    df = pd.read_csv('H:\\forcasting\\data\\training\\10655.csv')
    df['y'] = np.log1p(df.y.astype(float) + 1)
    print(df)
    model = Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.15,
        daily_seasonality=True,
        holidays=us_public_holidays(),

        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model.add_seasonality(
        name='weekly', period=7, fourier_order=3, prior_scale=0.1)


    df['nfl_sunday'] = df['ds'].apply(nfl_sunday)

    print(df)
    model.add_regressor('nfl_sunday')
    model.add_country_holidays(country_name='US')
    #save model
    filename = 'prophet_1.0.pkl'
    root = os.path.join(ROOT_DIR,'models')
    print(ROOT_DIR)
    path = os.path.join(root,filename)

    # with open(path, "wb") as f:
    #     pickle.dump(model, f)
    print(f"model saved at: {path}")

    model.fit(df)
    future = model.make_future_dataframe(periods=28)
    future['nfl_sunday'] = future['ds'].apply(nfl_sunday)
    forecast = model.predict(future)
    print(forecast[-28:])



if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    build_model()
    print('Memory Usage(MB):',process.memory_info()[0] / float(2 ** 20))