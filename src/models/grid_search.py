# github link: https://github.com/ds-praveenkumar/kaggle
# Author: ds-praveenkumar
# file: forcasting/grid_search.py/
# Created by ds-praveenkumar at 14-06-2020 10 52
# feature:


from sklearn.model_selection import ParameterGrid
from fbprophet import Prophet
import pandas as pd
import numpy as np
from fbprophet.diagnostics import cross_validation,performance_metrics
from multiprocessing import  Pool
import  os
import time

params_grid =  {
                'growth' : ('linear','logistic'),
                'changepoint_prior_scale' : [0.1,0.15,0.3,0.5,0.7],
                'seasonality_mode':('multiplicative','additive'),
                'yearly_seasonality':[10,20,30]
}

def set_parms():

    grid = ParameterGrid(params_grid)
    print([p for p in grid])
    return grid

def search_params(grid):
    df = pd.read_csv('H:\\forcasting\\data\\training\\11306.csv')
    df['y'] = np.log1p(df.y.astype(float) + 1)
    df['cap'] = df.y.max()
    df['floor'] = df.y.min()
    print(df)
    metric_dict = dict()
    for g in grid:

        model = Prophet(**g)
        model.add_country_holidays('US')
        model.fit(df)
        df_cv = cross_validation(model,initial='300 days',period='30 days',horizon='10 days')
        #print(df_cv)
        df_p = performance_metrics(df_cv)
        print('*'* 50)
        print('grid: ', g)
        print('rmse: ', df_p.rmse.mean())
        print('*' * 50)
        metric_dict[str(g)]=df_p.rmse.mean()
    return metric_dict


def main():
    grid = set_parms()
    start = time.time()
    # gs_res = search_params(grid)
    # for el in gs_res:
    #     for key in el.keys():
    #         print(f"{key}:{el[key]}", end='\n')
    # print('end time(mins):', float((time.time() -start)/60),end='\n')

    pool = Pool(processes=os.cpu_count() )
    result = pool.map(search_params, [grid] )
    for el in result:
        for key in el.keys():
            print(f"{key}:{el[key]}", end='\n')
    print('end time(mins):', float((time.time() - start) / 60), end='\n')

if __name__ == '__main__':
    main()






