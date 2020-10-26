import pandas as pd
import pyreadr
import tensorflow as tf
import os
import numpy as np
import copy
import json
import pathlib
import sklearn
from sklearn.model_selection import train_test_split

filedir = pathlib.Path(__file__).resolve().parent
datasets_dir = filedir.parent / "dataset"
data_name = "data.csv"
test_name = "test.csv"
domain_name = "domain.json"

def make_discretizer(series, n_bins, preprocess):
    if type(series[0]) is not str:
        if n_bins==1:
            if preprocess:
                scaler = sklearn.preprocessing.MinMaxScaler()
                scaler.fit(np.array(series).reshape(-1,1))
                def discretizer(value):
                    return scaler.transform([[value]])[0][0]
            else:
                def discretizer(value):
                    return value
            n_val = n_bins        
        else:
            max = series.max()
            min = series.min()
            def discretizer(value):
                bin_size = (max - min)/n_bins
                if bin_size == 0:
                    return 0
                if value == max:
                    return n_bins -1
                return int((value-min)/bin_size)
            n_val = n_bins
    else:
        value_set = sorted(list(set(series)))
        n_val = len(value_set)
        def discretizer(value):
            return value_set.index(value)
    return discretizer, n_val

def split_test(df):
    df = df.sample(frac=1, random_state=0)
    return train_test_split(df, test_size=0.1)
    

def discretize(df, n_bins, preprocess=True):
    domain = {}
    df_ = copy.deepcopy(df)
    for col in df.columns:
        discretizer, n_val = make_discretizer(df[col], n_bins, preprocess)
        df_[col] = df[col].map(discretizer)
        domain[col] = n_val
    return df_, domain

def save(dir, df, db, n_bins=10):
    if (db == "mnist") or (db == "fashion"):
        preprocess = False
    else:
        preprocess = True
    df, domain = discretize(df, n_bins, preprocess)
    if db == "credit":
        df = pd.DataFrame(np.clip(df, -1, 1))
        pos = df[df[29] == 1]
        neg = df[df[29] == 0]
        df = pd.concat([pos, neg]).reset_index(drop=True)
    train, test = split_test(df)
    train.to_csv(dir / data_name, index=None)
    test.to_csv(dir / test_name, index=None)
    with (dir / domain_name).open(mode="w") as f:
        json.dump(domain, f)
    

if __name__ == "__main__":
    print("process adult")
    dataset_dir = datasets_dir / "adult"
    df = pd.read_csv(dataset_dir / "adult.data", header=None)
    test_df = pd.read_csv(dataset_dir / "adult.test", header=None)
    df = pd.concat([df, test_df], ignore_index=True)
    save(dataset_dir, df, "adult", 1)
    
    print("process esr")
    dataset_dir = datasets_dir / "esr"
    df = pd.read_csv(dataset_dir / "esr.csv", index_col=0, header=None)
    dic = {i+1:i for i in range(len(df))}
    df = df.drop(df.index[0]).rename(columns=dic)
    for col in df.columns:
        df[col] = df[col].map(int)
    df[178] = df[178].map(lambda x:"t" if x==1 else "f")
    save(dataset_dir, df, "esr", 1)
    
    print("process isolet")
    dataset_dir = datasets_dir / "isolet"
    isolet1234 = pd.read_csv(dataset_dir / "isolet1+2+3+4.data", header=None)
    isolet5 = pd.read_csv(dataset_dir / "../../dataset/isolet/isolet5.data", header=None)
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
    save(dataset_dir, df, "isolet", 1)
    
    print("process credit")
    dataset_dir = datasets_dir / "credit"
    result = pyreadr.read_r(str(dataset_dir / "creditcard.Rdata"))
    data = result["creditcard"]
    Y = data["Class"].to_numpy(dtype="float32")
    Y = list(map(lambda x:"fraud" if x==0 else "normal", Y))
    data = data.drop(["Time","Class"], axis=1)
    df = pd.concat([data, pd.DataFrame(Y)], axis=1, ignore_index=True)
    save(dataset_dir, df, "credit", 1)
    
    print("process mnist")
    dataset_dir = datasets_dir / "mnist"
    (x_train, y_train), (x_test, y_test)  = tf.keras.datasets.mnist.load_data()
    X = np.concatenate([x_train, x_test]).reshape(len(x_train) + len(x_test), -1)
    Y = np.concatenate([y_train, y_test])
    X = pd.DataFrame(X.astype("float32") / 255.)
    Y = pd.DataFrame(Y)
    df = pd.concat([X,Y], axis=1, ignore_index=True)
    df[784] = df[784].map(lambda x:str(x))
    save(dataset_dir, df, "mnist", 1)

    print("process fashion")
    dataset_dir = datasets_dir / "fashion"
    (x_train, y_train), (x_test, y_test)  = tf.keras.datasets.fashion_mnist.load_data()
    X = np.concatenate([x_train, x_test]).reshape(len(x_train) + len(x_test), -1)
    Y = np.concatenate([y_train, y_test])
    X = pd.DataFrame(X.astype("float32") / 255.)
    Y = pd.DataFrame(Y)
    df = pd.concat([X,Y], axis=1, ignore_index=True)
    df[784] = df[784].map(lambda x:str(x))
    save(dataset_dir, df, "fashion", 1)
