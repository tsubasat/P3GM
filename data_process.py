import pandas as pd
import pyreadr
import tensorflow as tf
import os
import numpy as np
import copy
import json
import pathlib
import sklearn
import random
from sklearn.model_selection import train_test_split

filedir = pathlib.Path(__file__).resolve().parent
datasets_dir = filedir.parent / "dataset"
data_name = "data.csv"
test_name = "test.csv"
domain_name = "domain.json"

def save(dir, df, categorical_features):

    domain = {}
    for col in df.columns:
        if col in categorical_features:
            domain[col] = len(df[col].unique())
        else:
            domain[col] = 1

    train, test = train_test_split(df, test_size=0.1)
    train.to_csv(dir / data_name, index=None)
    test.to_csv(dir / test_name, index=None)
    with (dir / domain_name).open(mode="w") as f:
        json.dump(domain, f)

def preprocess(df, db):
    random.seed(0)
    
    obj_col = list(df.select_dtypes(include='object').columns)
    le = sklearn.preprocessing.LabelEncoder()

    if db == "adult":
        def adult_process(v):
            if v[-1] == ".":
                return v[:-1]
            else:
                return v
        df[14] = np.array([adult_process(v) for v in df[14]]).reshape(-1,1)
    elif db == "credit":
        zero_data = list(df[df[0] == "fraud"].index)
        one_data = list(df[df[0] != "fraud"].index)
        indice = random.sample(zero_data, 10000)
        indice.extend(one_data)
        indice = [int(v)-1 for v in indice]
        df_ = pd.DataFrame(np.array(df)[indice])
        df_.columns = df.columns
        df = df_

    for col in obj_col:
        df[col] = le.fit_transform(df[col])

    for col in df.columns:
        scaler = sklearn.preprocessing.MinMaxScaler()
        df[col] = scaler.fit_transform(np.array(df[col]).reshape(-1,1))

    return df


if __name__ == "__main__":
    print("process adult")
    dataset_dir = datasets_dir / "adult"
    dataset_dir.mkdir(exist_ok=True)
    label_column = 14
    df = pd.read_csv("/data/takagi/dataset/adult/adult.data", header=None)
    test_df = pd.read_csv("/data/takagi/dataset/adult/adult.test", header=None)
    df = pd.concat([df, test_df], ignore_index=True)
    df = preprocess(df, "adult")
    save(dataset_dir, df, [label_column])
    
    print("process esr")
    dataset_dir = datasets_dir / "esr"
    dataset_dir.mkdir(exist_ok=True)
    df = pd.read_csv("/data/takagi/dataset/esr/esr/esr.csv")
    save(dataset_dir, df, ["178"])
    
    print("process isolet")
    dataset_dir = datasets_dir / "isolet"
    dataset_dir.mkdir(exist_ok=True)
    isolet1234 = pd.read_csv("/data/takagi/dataset/isolet/isolet1+2+3+4.data", header=None)
    isolet5 = pd.read_csv("/data/takagi/dataset/isolet/isolet5.data", header=None)
    isolet = pd.concat([isolet1234, isolet5]).reset_index(drop=True)
    Y = isolet[617].to_numpy()
    isolet = isolet.drop(617, axis=1)
    for col in isolet.columns:
        isolet[col] = isolet[col].map(float)
    vowel = np.array([ord("a"), ord("i"), ord("u"), ord("e"), ord("o")]) - ord("a") + 1
    y_vo = []
    for y in Y:
        if y in vowel:
            y_vo.append("vowel")
        else:
            y_vo.append("cos")
    Y = np.array(y_vo).reshape(len(isolet), -1)
    Y = pd.DataFrame(Y)
    df = pd.concat([isolet, pd.DataFrame(Y)], axis=1, ignore_index=True)
    save(dataset_dir, df, [617])
    
    print("process credit")
    dataset_dir = datasets_dir / "credit"
    dataset_dir.mkdir(exist_ok=True)
    label_column = 0
    result = pyreadr.read_r("/data/takagi/dataset/credit/creditcard.Rdata")
    data = result["creditcard"]
    Y = data["Class"].to_numpy(dtype="float32")
    Y = list(map(lambda x:"fraud" if x==0 else "normal", Y))
    data = data.drop(["Time","Class"], axis=1)
    df = pd.concat([data, pd.DataFrame(Y, index=data.index)], axis=1)
    df = preprocess(df, "credit")
    save(dataset_dir, df, [label_column])
    
    print("process mnist")
    dataset_dir = datasets_dir / "mnist"
    dataset_dir.mkdir(exist_ok=True)
    (x_train, y_train), (x_test, y_test)  = tf.keras.datasets.mnist.load_data()
    X = np.concatenate([x_train, x_test]).reshape(len(x_train) + len(x_test), -1)
    Y = np.concatenate([y_train, y_test])
    X = pd.DataFrame(X.astype("float32") / 255.)
    Y = pd.DataFrame(Y)
    df = pd.concat([X,Y], axis=1, ignore_index=True)
    df[784] = df[784].map(lambda x:str(x))
    save(dataset_dir, df, [784])

    print("process fashion")
    dataset_dir = datasets_dir / "fashion"
    dataset_dir.mkdir(exist_ok=True)
    (x_train, y_train), (x_test, y_test)  = tf.keras.datasets.fashion_mnist.load_data()
    X = np.concatenate([x_train, x_test]).reshape(len(x_train) + len(x_test), -1)
    Y = np.concatenate([y_train, y_test])
    X = pd.DataFrame(X.astype("float32") / 255.)
    Y = pd.DataFrame(Y)
    df = pd.concat([X,Y], axis=1, ignore_index=True)
    df[784] = df[784].map(lambda x:str(x))
    save(dataset_dir, df, [784])
