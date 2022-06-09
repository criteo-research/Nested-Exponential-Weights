import re
import os
import numpy as np
import pandas as pd


def get_regex_pattern(params_keys, metrics_keys):
    regex_pattern = 'algo:(.*)'
    if len(params_keys) > 1:
        regex_pattern += '\|' + '\|'.join(
            '{}:(.*)'.format(k) for k in (params_keys) if k not in ['algo', 'experiment'])
        regex_pattern += '' + '\|'.join('{}:(.*)'.format(k) for k in (metrics_keys))
    return regex_pattern

def get_col_df(params_keys, metrics_keys):
    columns = ['algo']
    columns += (params_keys) + (metrics_keys)
    return columns

def create_df(fname, regex_pattern, columns):
    rgx = re.compile(regex_pattern, flags=re.M)
    with open(fname, 'r') as f:
        lines = rgx.findall(f.read())
        df = pd.DataFrame(lines, columns=columns)

        columns_str = ['algo']
        columns_flt_int = [item for item in columns if item not in columns_str]
        for col in columns_flt_int:
            try:
                df[col] = df[col].astype(np.int32)
            except ValueError:
                df[col] = df[col].astype(np.float32, errors='ignore')
    return df