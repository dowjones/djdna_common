import pandas as pd
import numpy as np


def industries_visual_hierarchy() -> pd.DataFrame:
    ret_ind = pd.read_csv('djdna_common/dicts/industries-hrchy.csv')
    ret_ind = ret_ind.replace(np.nan, '', regex=True)
    ret_ind['name'] = '(' + ret_ind['ind_fcode'] + ') ' + ret_ind['name']
    return ret_ind


def regions_visual_hierarchy() -> pd.DataFrame:
    ret_reg = pd.read_csv('djdna_common/dicts/regions-hrchy.csv')
    ret_reg = ret_reg.replace(np.nan, '', regex=True)
    ret_reg['name'] = '(' + ret_reg['reg_fcode'] + ') ' + ret_reg['name']
    return ret_reg
