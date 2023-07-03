import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import lightgbm as lgb


SEED = 21
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

N_COMPONENTS = 18  # For PCA
INPUT_DIR = "/data/recsys/2023/input/sharechat_recsys2023_data"
OUTPUT_DIR = '/data/recsys/2023/output/lightgbm'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


N_SPLITS = 10
PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt', 
    'seed': SEED,
    'num_leaves': 56,
    'learning_rate': 0.035,
    'feature_fraction': 0.4,
    'bagging_fraction': 1.0,
    'n_jobs': 20,
    'lambda_l2': 0.168,
    'lambda_l1': 1.8e-7,
    'verbose': 1,
    'min_data_in_leaf': 20,
    'max_bin': 255,
}


def train_lgbm(df:pd.DataFrame, x_cols:list, y_col:str, output_dir:str, ctg_cols:list):
    """ Train & Save lgbm model and Save feature_importance.csv for each fold """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f'label={y_col} fold[{fold}]', flush=True)
        tr_ds = lgb.Dataset(
            df.iloc[train_index][x_cols], 
            df.iloc[train_index][y_col],
            categorical_feature=ctg_cols
        )
        val_ds = lgb.Dataset(
            df.iloc[valid_index][x_cols], 
            df.iloc[valid_index][y_col],
            categorical_feature=ctg_cols
        )
        clf = lgb.train(
            params=PARAMS,
            train_set=tr_ds,
            num_boost_round=4000000,
            valid_sets=val_ds,
            # feval=calculate_log_loss,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
        )
        clf.save_model(
            f'{output_dir}/{y_col}_f{fold+1:02d}.lgb',
            num_iteration=clf.best_iteration, 
        )
        feature_imp = pd.DataFrame(
            {
                'Feature': x_cols,
                'Value': clf.feature_importance(),
            }
        )
        feature_imp.to_csv(
            f'{output_dir}/{y_col}_f{fold+1:02d}_feature_importance.csv',
            index=False,
        )


def infer_lgbm(df:pd.DataFrame, x_cols:list, y_col:str, output_dir:str):
    """ Inferring using df(test data) """
    ys = []
    for fold in range(N_SPLITS):
        print(fold, end=', ')
        model_file = f'{output_dir}/{y_col}_f{fold+1:02d}.lgb'
        if not os.path.exists(model_file): break 
        clf = lgb.Booster(model_file=model_file)
        y = clf.predict(df[x_cols], num_iteration=clf.best_iteration)
        ys.append(y)
    print()
    return np.mean(ys, axis=0)


def main():
    path = f"{INPUT_DIR}/train_and_test.pqt"
    df = pd.read_parquet(path)
    df['nan_count'] = (df.isna().sum(axis=1) - 2).astype('category')
    df.fillna(-1, inplace=True)
    label_col = 'is_installed'

    submission = df[df[label_col] == -1][['f_0']].copy()
    submission.columns = ['row_id']
    submission['is_clicked'] = 0.0

    num_ctg_ns = list(range(70, 80)) + list(range(60, 64)) + [57]
    num_ctg_cols = [f'f_{n}' for n in num_ctg_ns]
    num_cols = [f'f_{n}' for n in range(42, 80) if n not in num_ctg_ns]

    ctg_cols = [
        f'f_{n}' for n in range(2, 33) if n not in (7, 26, 27, 28, 29)
    ]
    bin_cols = [f'f_{n}' for n in range(33, 42)]

    for col in ctg_cols + bin_cols:
        df[col] = (df[col] + 1).astype('int16').astype('category')

    for col in num_ctg_cols:
        num2ctg = {n:i for i, n in enumerate(df[col].unique(), 1)}
        df[col] = df[col].map(num2ctg).astype('category')
    
    cat_cols = ctg_cols + bin_cols + num_ctg_cols + ['nan_count']

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    pca = PCA(n_components=N_COMPONENTS)
    pca_df = df.copy()
    pca_data = pca.fit_transform(pca_df[num_cols])
    print(sum(pca.explained_variance_ratio_))
    pca_cols = [f'pc_{n}' for n in range(N_COMPONENTS)]
    pca_df = pd.DataFrame(data=pca_data, columns=pca_cols)
    df = pd.concat([pca_df, df], axis=1)

    num_cols = pca_cols + num_cols

    te_df = df[df[label_col] == -1].copy()
    tr_df = df[df[label_col] != -1].copy()
    print('tr-shape:', tr_df.shape, 'te-shape:', te_df.shape)

    te_df.drop(columns=[label_col], inplace=True)

    x_cols = cat_cols + num_cols
    train_lgbm(tr_df, x_cols, label_col, OUTPUT_DIR, cat_cols)
    submission[label_col] = infer_lgbm(te_df, x_cols, label_col, OUTPUT_DIR)
    
    submission.to_csv(
        f'{OUTPUT_DIR}/lgbm_S{SEED}_N{N_COMPONENTS}_ff-{PARAMS["feature_fraction"]}.csv',
        header=True,
        sep='\t',
        index=False,
    )


if __name__ == '__main__':
    main()
