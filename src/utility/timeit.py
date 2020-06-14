# github link: https://github.com/ds-praveenkumar/kaggle-m5-accuracy.git
# Author: ds-praveenkumar
# file: kaggle-m5-accuracy/timeit.py/
# Created by ds-praveenkumar at 06-06-2020 23 41
# feature: calculates time of methods

from functools import wraps
from time import time
import logging

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

def timeit(func):
    """
        calculates Execution time for functions
    :param func: used functions
    :return: time for execution of functions
    """
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time()
        result = func(*args,**kwargs)
        end = time()
        logging.info(f"Elapsed time for {func.__name__}: {round(end- start,5)}")
        return result
    return wrapper