import pandas as pd

def load_wine_data():
    df = pd.read_csv('WineQT.csv')
    y = df['quality']
    X = df.drop(columns='quality')
    return X, y

def load_drug_data():
    df = pd.read_csv('drug.csv')
    X = df.drop(columns='drug')
    y = df['drug']
    return X, y

def load_iris_data():
    df = pd.read_csv('Iris.csv')
    y = df["Species"]
    X = df.drop(columns=["Species", "Id"])
    return X, y

def load_titanic_data():
    train = pd.read_csv('train.csv')
    X_train = train.drop(columns=["Survived", "PassengerId"])
    y_train = train["Survived"]
    X_test = pd.read_csv('test.csv').sort_values(by=["PassengerId"]).drop(columns=["PassengerId"])
    y_test = pd.read_csv('gender_submission.csv').sort_values(by=["PassengerId"])["Survived"]
    return X_train, y_train, X_test, y_test
