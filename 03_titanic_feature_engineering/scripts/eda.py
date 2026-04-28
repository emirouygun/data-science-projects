import pandas as pd
import numpy as np

def check_df(dataframe, head=5):
    """
    DataFrame hakkinda bilgi verir.
    """
    print("########## SHAPE ##########")
    print(dataframe.shape)

    print("\n########## TYPES ##########")
    print(dataframe.dtypes)

    print("\n########## HEAD ##########")
    print(dataframe.head(head))

    print("\n########## TAIL ##########")
    print(dataframe.tail(head))

    print("\n########## NA ##########")
    print(dataframe.isnull().sum())

    print("\n########## DESCRIBE ##########")
    print(dataframe.describe().T)