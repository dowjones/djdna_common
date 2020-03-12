import pandas as pd
import numpy as np


def industries_hierarchy() -> pd.DataFrame:
    ret_ind = pd.read_csv('djdna_common/dicts/industries-hrchy.csv')
    ret_ind = ret_ind.replace(np.nan, '', regex=True)
    return ret_ind


def regions_hierarchy() -> pd.DataFrame:
    ret_reg = pd.read_csv('djdna_common/dicts/regions-hrchy.csv')
    ret_reg = ret_reg.replace(np.nan, '', regex=True)
    return ret_reg
