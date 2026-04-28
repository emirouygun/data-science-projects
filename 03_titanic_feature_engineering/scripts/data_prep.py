import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#############################################
# Aykiri deger fonksiyonlari (OUTLIER)
#############################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Verilen değişken için IQR yöntemi ile aykiri değer sinirlarini hesaplar.

    q1: 1. çeyrek (%25)
    q3: 3. çeyrek (%75)
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr

    return low_limit, up_limit 

def replace_with_thresholds(dataframe, variable):
    """
    Aykiri değerleri silmek yerine alt ve üst sinirlara eşitler (Winsorization).
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

def check_outlier(dataframe, col_name):
    """
    İlgili değişkende aykiri değer olup olmadiğini kontrol eder.
    True / False döner.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    return dataframe[
        (dataframe[col_name] < low_limit) |
        (dataframe[col_name] > up_limit)
    ].any(axis=None)

def grab_outliers(dataframe, col_name, index=False):
    """
    Aykiri değerleri listeler.
    index=True ise sadece index döner.
    """
    low, up = outlier_thresholds(dataframe, col_name)

    outliers = dataframe[
        (dataframe[col_name] < low) |
        (dataframe[col_name] > up)
    ]

    if index:
        return outliers.index
    else:
        return outliers
    
def remove_outlier(dataframe, col_name):
    """
    Aykiri değerleri veri setinden tamamen çikarir.
    """
    low, up = outlier_thresholds(dataframe, col_name)

    df_without_outliers = dataframe[
        ~((dataframe[col_name] < low) |
          (dataframe[col_name] > up))
    ]

    return df_without_outliers

#############################################
# Eksik deger fonksiyonlari (MISSING VALUES)
#############################################

def missing_values_table(dataframe, na_name=False):
    """
    Eksik değerlerin sayisini ve oranini gösterir.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])

    print(missing_df)

    if na_name:
        return na_columns
    
def missing_vs_target(dataframe, target, na_columns):
    """
    Eksik değerlerin target değişken ile ilişkisini inceler.
    """
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    for col in temp_df.columns:
        if "NA_FLAG" in col:
            print(col)
            print(temp_df.groupby(col)[target].mean(), "\n")

#############################################
# Encoding Fonksiyonlar
#############################################

def label_encoder(dataframe, binary_col):
    """
    2 sınıflı kategorik değişkenleri 0-1'e çevirir.
    """
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """
    Kategorik değişkenleri one-hot encoding yapar.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def titanic_data_prep(dataframe):
    """
    Titanic veri setini modellemeye hazır hale getirir.
    Tüm veri temizleme + feature engineering + encoding işlemlerini yapar.
    """

    df = dataframe.copy()

    ###################################################
    # Eksik değerler
    ###################################################
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if "Cabin" in df.columns:
        df.drop("Cabin", axis=1, inplace=True)

    ###################################################
    # Outlier işlemleri
    ###################################################
    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = [col for col in num_cols if col != "Survived"]

    #for col in num_cols:
    #    replace_with_thresholds(df, col)

    ###################################################
    # Gereksiz değişkenleri sil
    ###################################################
    df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True, errors="ignore")

    return df