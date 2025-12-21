import pandas as pd

def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    )
    df = df[["age", "fare", "sex", "survived"]].dropna()
    return df
