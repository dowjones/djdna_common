import os
import pandas as pd
import numpy as np

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
datadir = os.path.join(rootdir, 'djdna_common/dicts')
ind_hrchy_path = os.path.join(datadir, 'industries-hrchy.csv')
reg_hrchy_path = os.path.join(datadir, 'regions-hrchy.csv')


def industries_hierarchy() -> pd.DataFrame:
    ret_ind = pd.read_csv(ind_hrchy_path)
    ret_ind = ret_ind.replace(np.nan, '', regex=True)
    return ret_ind


def regions_hierarchy() -> pd.DataFrame:
    ret_reg = pd.read_csv(reg_hrchy_path)
    ret_reg = ret_reg.replace(np.nan, '', regex=True)
    return ret_reg
