import pandas as pd

def prepare_features(df, model_features):
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Sex_male"] = (df["Sex"].str.lower() == "male").astype(int)
    df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
    df["Pclass_3"] = (df["Pclass"] == 3).astype(int)

    df = df.reindex(columns=model_features, fill_value=0)
    return df
