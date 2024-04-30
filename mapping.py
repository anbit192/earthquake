
import pandas as pd
import numpy as np

def _get_mapping_dict(string_list):
    dict = {}
    for i in range(string_list.shape[0]):
        current_value = string_list[i]
        dict[current_value] = i

    return dict

def map_str_values(df):
    for i in range(df.shape[1]):
        col_name = df.columns[i]

        if (df[col_name].dtype == "object"):
            uniques = np.unique(df[col_name])
            map_dict = _get_mapping_dict(uniques)
            df[col_name] = df[col_name].map(map_dict)

    return df


