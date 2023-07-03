import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


# cy4ego, 2023.05.05, Added for RecSys2023
def fe(args):
    label_col = 'is_installed'
    num_ctg_ns = list(range(70, 80)) + list(range(60, 64)) + [57]
    num_ctg_cols = [f'f_{n}' for n in num_ctg_ns]
    num_cols = [f'f_{n}' for n in range(42, 80) if n not in num_ctg_ns]

    BASE_DIR = "/data/recsys/2023/input/sharechat_recsys2023_data"
    path = f"{BASE_DIR}/train_and_test.pqt"
    df = pd.read_parquet(path)
    df['nan_count'] = (df.isna().sum(axis=1) - 2).astype('int16')
    tr_df = df[~df[label_col].isna()].copy(deep=True)
    te_df = df[df[label_col].isna()].copy(deep=True).reset_index(drop=True)
    del df 
    tr_df.fillna(-1, inplace=True)
    te_df.fillna(-1, inplace=True)
    tr_df['nan_count'] = tr_df['nan_count'].astype('category')
    te_df['nan_count'] = te_df['nan_count'].astype('category')
    print(f'[1] tr-shape: {tr_df.shape} te-shape: {te_df.shape}')

    ctg_cols = [
        f'f_{n}' for n in range(2, 33) if n not in (7, 26, 27, 28, 29)
    ]
    bin_cols = [f'f_{n}' for n in range(33, 42)]

    for col in ctg_cols + bin_cols:
        tr_df[col] = (tr_df[col] + 1).astype('int16').astype('category')
        te_df[col] = (te_df[col] + 1).astype('int16').astype('category')

    num_ctg_dict = {}
    for col in num_ctg_cols:
        num2ctg = {n:i for i, n in enumerate(tr_df[col].unique(), 1)}
        tr_df[col] = tr_df[col].map(num2ctg).astype('category')
        num_ctg_dict[col] = num2ctg
    
    def _apply_num_ctg(num, ctg_dict):
        keys = np.array(sorted(list(ctg_dict.keys())))
        key = keys[np.argmin(abs(keys-num))]
        return ctg_dict[key]
    
    for col in num_ctg_cols:
        te_df[col] = te_df[col].apply(
            lambda x: _apply_num_ctg(x, num_ctg_dict[col])
        ).astype('category')

    if args.scale:
        scaler = StandardScaler()
        tr_df[num_cols] = scaler.fit_transform(tr_df[num_cols])
        te_df[num_cols] = scaler.transform(te_df[num_cols])
        print(f'[2] tr-shape: {tr_df.shape} te-shape: {te_df.shape}')

    if args.use_pca:
        N_COMPONENTS = 18
        pca = PCA(n_components=N_COMPONENTS)
        # For train data
        pca_data = pca.fit_transform(tr_df[num_cols])
        print(sum(pca.explained_variance_ratio_))
        pca_cols = [f'pc_{n}' for n in range(N_COMPONENTS)]
        pca_df = pd.DataFrame(data=pca_data, columns=pca_cols)
        tr_df = pd.concat([pca_df, tr_df], axis=1)
        del pca_data, pca_df 
        # For test data
        pca_data = pca.transform(te_df[num_cols])
        print(np.shape(pca_data))
        pca_df = pd.DataFrame(data=pca_data, columns=pca_cols)
        te_df = pd.concat([pca_df, te_df], axis=1)
        
        num_cols = pca_cols + num_cols

    cat_cols = ctg_cols + bin_cols + num_ctg_cols + ['nan_count']

    print('tr-shape:', tr_df.shape, 'te-shape:', te_df.shape)
    print('categories:', len(cat_cols), 'numerics:', len(num_cols))

    X = tr_df[num_cols + cat_cols].to_numpy()  # v01_01
    y = tr_df[label_col].to_numpy()
    te_X = te_df[num_cols + cat_cols].to_numpy()  # v01_01
    print(f'X-shape={X.shape} y-shape={y.shape}, te_X-shape={te_X.shape}')
    return X, y, te_X


def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q


def load_data(args):
    print("Loading dataset " + args.dataset + "...")

    if args.dataset == "CaliforniaHousing":  # Regression dataset
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    elif args.dataset == "Covertype":  # Multi-class classification dataset
        X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
        # X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset

    elif args.dataset == "KddCup99":  # Multi-class classification dataset with categorical data
        X, y = sklearn.datasets.fetch_kddcup99(return_X_y=True)
        X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset

        # filter out all target classes, that occur less than 1%
        target_counts = np.unique(y, return_counts=True)
        smaller1 = int(X.shape[0] * 0.01)
        small_idx = np.where(target_counts[1] < smaller1)
        small_tar = target_counts[0][small_idx]
        for tar in small_tar:
            idx = np.where(y == tar)
            y[idx] = b"others"

        # new_target_counts = np.unique(y, return_counts=True)
        # print(new_target_counts)

        '''
        # filter out all target classes, that occur less than 100
        target_counts = np.unique(y, return_counts=True)
        small_idx = np.where(target_counts[1] < 100)
        small_tar = target_counts[0][small_idx]
        for tar in small_tar:
            idx = np.where(y == tar)
            y[idx] = b"others"

        # new_target_counts = np.unique(y, return_counts=True)
        # print(new_target_counts)
        '''
    elif args.dataset == "Adult" or args.dataset == "AdultCat":  # Binary classification dataset with categorical data, if you pass AdultCat, the numerical columns will be discretized.
        url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

        features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        label = "income"
        columns = features + [label]
        df = pd.read_csv(url_data, names=columns)

        # Fill NaN with something better?
        df.fillna(0, inplace=True)
        if args.dataset == "AdultCat":
            columns_to_discr = [('age', 10), ('fnlwgt', 25), ('capital-gain', 10), ('capital-loss', 10),
                                ('hours-per-week', 10)]
            for clm, nvals in columns_to_discr:
                df[clm] = discretize_colum(df[clm], num_values=nvals)
                df[clm] = df[clm].astype(int).astype(str)
            df['education_num'] = df['education_num'].astype(int).astype(str)
            args.cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        X = df[features].to_numpy()
        y = df[label].to_numpy()

    elif args.dataset == "HIGGS":  # Binary classification dataset with one categorical feature
        path = "/opt/notebooks/data/HIGGS.csv.gz"
        df = pd.read_csv(path, header=None)
        df.columns = ['x' + str(i) for i in range(df.shape[1])]
        num_col = list(df.drop(['x0', 'x21'], 1).columns)
        cat_col = ['x21']
        label_col = 'x0'

        def fe(x):
            if x > 2:
                return 1
            elif x > 1:
                return 0
            else:
                return 2

        df.x21 = df.x21.apply(fe)

        # Fill NaN with something better?
        df.fillna(0, inplace=True)

        X = df[num_col + cat_col].to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Heloc":  # Binary classification dataset without categorical data
        path = "heloc_cleaned.csv"  # Missing values already filtered
        df = pd.read_csv(path)
        label_col = 'RiskPerformance'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    # cy4ego, 2023.05.05
    elif args.dataset == "RecSys2023":  # Binary classification dataset with categorical data
        X, y, te_X = fe(args)
        
        tot_X = np.vstack([X, te_X])

    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")
    print(X.shape)

    # Preprocess target
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Setting this if classification task
        if args.objective == "classification":
            args.num_classes = len(le.classes_)
            print("Having", args.num_classes, "classes as target.")

    num_idx = []
    args.cat_dims = []

    # Preprocess data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            le = LabelEncoder()
            tot_X[:, i] = le.fit_transform(tot_X[:, i])

            # Setting this?
            args.cat_dims.append(len(le.classes_))

        else:
            num_idx.append(i)

    if args.one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.cat_idx])
        new_x2 = X[:, num_idx]
        X = np.concatenate([new_x1, new_x2], axis=1)
        print("New Shape:", X.shape)

    return tot_X[:X.shape[0], :], y, tot_X[X.shape[0]:, :]
