import pandas as pd
from preprocessing import encode_labels

def load_wine_data():
    df = pd.read_csv('/content/drive/MyDrive/ml/WineQT.csv')
    y = df['quality']
    X = df.drop(columns='quality')
    y_encoded = encode_labels(y)
    return X, y_encoded

def load_drug_data():
    df = pd.read_csv('/content/drive/MyDrive/ml/drug.csv')
    X = df.drop(columns='Drug')
    y = df['Drug']
    y_encoded = encode_labels(y)
    return X, y_encoded

def load_iris_data():
    df = pd.read_csv('/content/drive/MyDrive/ml/Iris.csv')
    y = df["Species"]
    y_encoded = encode_labels(y)
    X = df.drop(columns=["Species", "Id"])
    return X, y_encoded

def load_titanic_data():
    train = pd.read_csv('/content/drive/MyDrive/ml/train.csv')
    X_train = train.drop(columns=["Survived", "PassengerId"])
    y_train = train["Survived"]
    X_test = pd.read_csv('/content/drive/MyDrive/ml/test.csv').sort_values(by=["PassengerId"]).drop(columns=["PassengerId"])
    y_test = pd.read_csv('/content/drive/MyDrive/ml/gender_submission.csv').sort_values(by=["PassengerId"])["Survived"]
    return X_train, y_train, X_test, y_test
