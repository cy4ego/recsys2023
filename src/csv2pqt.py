from glob import glob 
import pandas as pd
import numpy as np

# DIRECTORY SETTING
base_dir = '/data/recsys/2023'
data_dir = f'{base_dir}/input/sharechat_recsys2023_data'
train_dir = f'{data_dir}/train'
test_dir = f'{data_dir}/test'

# COLUMNS
category_columns = [f'f_{n}' for n in range(2, 33)]
binary_columns = [f'f_{n}' for n in range(33, 42)]
numeric_columns = [f'f_{n}' for n in range(42, 80)]
label_columns = ['is_clicked', 'is_installed']

usecols = ['f_0', 'f_1'] + category_columns + binary_columns + numeric_columns + label_columns

dtypes = {column: 'int8' for column in binary_columns + label_columns}


# MAIN(csv to parquet)
filepaths = glob(f'{train_dir}/*.csv') + glob(f'{test_dir}/*.csv')
dfs = []
for filepath in filepaths:
    dfs.append(
        pd.read_csv(  # load to pandas.DataFrame
            filepath, sep='\t', usecols=usecols, dtype=dtypes
        ).replace(r'^\s*$', np.nan, regex=True)  # replace empty cell to NaN
    )
df = pd.concat(dfs)
del dfs


print('Save file')
df.to_parquet(f'{data_dir}/train_and_test.pqt', index=False)
