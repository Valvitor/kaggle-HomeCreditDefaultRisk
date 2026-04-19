import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def agg_numeric(df, group_var, df_name):
    """
    Agrega features numéricas em relação à PK.
    Retorna o dataframe com as features agregadas com prefixo.
    """
    numeric_df = df.select_dtypes('number').drop(columns=[group_var], errors='ignore').copy()
    numeric_df[group_var] = df[group_var].values

    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Achata MultiIndex de colunas
    agg.columns = [
        group_var if col[0] == group_var
        else re.sub(r'[^A-Za-z0-9_]', '_', f'{df_name}_{col[0]}_{col[1]}')
        for col in agg.columns
    ]
    return agg


def agg_categorical(df, group_var, df_name):
    """
    Faz One-Hot Encoding nas colunas categóricas e agrega por grupo.
    Se não houver colunas categóricas, retorna apenas o group_var.
    """
    cat_cols = df.select_dtypes('object').columns.tolist()

    # Sem categorias: retorna só o ID para não quebrar o merge
    if not cat_cols:
        return df[[group_var]].drop_duplicates().reset_index(drop=True)

    categorical = pd.get_dummies(df[cat_cols]).copy()
    categorical[group_var] = df[group_var].values

    agg = categorical.groupby(group_var).agg(['sum', 'mean']).reset_index()

    # Achata MultiIndex de colunas
    agg.columns = [
        group_var if col[0] == group_var
        else re.sub(r'[^A-Za-z0-9_]', '_', f'{df_name}_{col[0]}_{col[1]}')
        for col in agg.columns
    ]
    return agg


def process_bureau(bureau_df, bureau_balance_df):
    """
    Pré-processa dados do bureau, garantindo que não haja data leakage temporal.
    """
    # 1. Agregação em bureau_balance (Nível SK_ID_BUREAU)
    bb_agg_cat = agg_categorical(bureau_balance_df, group_var='SK_ID_BUREAU', df_name='bureau_balance')
    bb_agg_num = agg_numeric(bureau_balance_df, group_var='SK_ID_BUREAU', df_name='bureau_balance')

    bureau_df = bureau_df.merge(bb_agg_cat, on='SK_ID_BUREAU', how='left')
    bureau_df = bureau_df.merge(bb_agg_num, on='SK_ID_BUREAU', how='left')

    # 2. Agregação na base do bureau (Nível SK_ID_CURR)
    bureau_agg_num = agg_numeric(bureau_df, group_var='SK_ID_CURR', df_name='bureau')
    bureau_agg_cat = agg_categorical(bureau_df, group_var='SK_ID_CURR', df_name='bureau')

    return bureau_agg_num, bureau_agg_cat


def kfold_target_encoding(train_df, test_df, cat_features, target_col, n_splits=5):
    """
    Target Encoding com KFold para evitar Data Leakage (fold a fold).
    """
    print("Aplicando K-Fold Target Encoding...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_encoded = train_df[cat_features].copy()
    test_encoded = test_df[cat_features].copy()

    for feature in cat_features:
        train_encoded[f'{feature}_TE'] = np.nan
        test_encoded[f'{feature}_TE'] = 0.0

        for train_idx, val_idx in kf.split(train_df):
            X_train_fold = train_df.iloc[train_idx]
            X_val_fold = train_df.iloc[val_idx]

            target_mean = X_train_fold.groupby(feature)[target_col].mean()
            train_encoded.loc[train_df.index[val_idx], f'{feature}_TE'] = (
                X_val_fold[feature].map(target_mean).values
            )

        full_target_mean = train_df.groupby(feature)[target_col].mean()
        test_encoded[f'{feature}_TE'] = test_df[feature].map(full_target_mean)

        global_mean = train_df[target_col].mean()
        train_encoded[f'{feature}_TE'].fillna(global_mean, inplace=True)
        test_encoded[f'{feature}_TE'].fillna(global_mean, inplace=True)

    return (
        train_encoded[[f'{f}_TE' for f in cat_features]],
        test_encoded[[f'{f}_TE' for f in cat_features]]
    )


if __name__ == "__main__":
    print("Módulo de feature engineering inicializado.")
