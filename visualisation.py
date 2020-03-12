import pandas as pd
import numpy as np
import taxonomy as tx


def industries_visual_hierarchy() -> pd.DataFrame:
    ret_ind = tx.industries_hierarchy()
    ret_ind['name'] = '(' + ret_ind['ind_fcode'] + ') ' + ret_ind['name']
    return ret_ind


def regions_visual_hierarchy() -> pd.DataFrame:
    ret_reg = tx.regions_hierarchy()
    ret_reg['name'] = '(' + ret_reg['reg_fcode'] + ') ' + ret_reg['name']
    return ret_reg
