import os
import numpy as np
import pandas as pd

############
# CHANGE n_split & output_dir's version
############

n_split = 1

def get_submission():
    input_dir = "/data/recsys2023"
    path = f"{input_dir}/train_and_test.pqt"
    df = pd.read_parquet(path)
    return df[df['is_installed'].isna()][['f_0']]

output_dir = '/data/recsys2023/output/tabsurvey/v17'
models = [
    # 'LightGBM', 'XGBoost', 'CatBoost', 
    'SAINT',  # 'VIME', 'NODE', 'TabTransformer', 'TabNet', 
]
submission = get_submission()
submission['is_clicked'] = 0.0
submission.columns = ['row_id', 'is_clicked']
print('submission-shape:', submission.shape)


for model in models:
    data_dir = f'{output_dir}/{model}/RecSys2023/infers'
    submission_path = f'{data_dir}/ts_{model.lower()}_submission.csv'
    preds = []
    for index in range(n_split):
        preds.append(np.load(f'{data_dir}/r_{index}.npy'))
    print(np.shape(preds))
    preds = np.mean(preds, axis=0)[:, 1]
    submission['is_installed'] = preds
    print(submission.shape)
    submission.to_csv(submission_path, index=False, sep='\t')