import tabledata_classification
import mnist_classification
from sklearn.preprocessing import OneHotEncoder
import sklearn
import argparse
import pandas as pd
import json
import numpy as np
import pathlib
filedir = pathlib.Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description='ml task')
parser.add_argument('--db', type=str, default="adult", help='database name')
parser.add_argument('--alg', type=str, default="p3gm", help='database name')
parser.add_argument("--test", type=str, default="")
parser.add_argument("--param", type=str, default="")
args = parser.parse_args()

dataset_dir = filedir.parent.parent.parent / "dataset" / f"{args.db}"
syn_data_dir = filedir.parent / "synthetic_data" / f"{args.db}"
result_dir = filedir.parent / "result" / f"{args.db}"
result_dir.mkdir(parents=True, exist_ok=True)

# The method to encode the categorical value to one-hot vector
def to_one_hot(orig_df, syn_df):
    syn_data = []
    orig_data = []
    
    domain_dir = dataset_dir / "domain.json"
    
    with open(domain_dir, "r") as f:
        domain = json.load(f)
    
    for key, dim in zip(orig_df.columns, domain.values()):
        orig_attr = np.array(orig_df[key]).reshape(-1,1)
        syn_attr = np.array(syn_df[key]).reshape(-1,1)
        if dim == 1:
            pass
        else:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(orig_attr.reshape(-1,1))
            orig_attr = np.array(enc.transform(orig_attr).toarray())
            syn_attr = np.array(enc.transform(syn_attr).toarray())
        orig_data.append(orig_attr)
        syn_data.append(syn_attr)
    orig_data = np.concatenate(orig_data, axis=1)
    syn_data = np.concatenate(syn_data, axis=1)
    
    return orig_data, syn_data

# The mehtod to split data to data and label
def split(data):
    if args.db == "adult":
        return data[:, :-2], data[:, -2:].argmin(axis=1).ravel()
    elif args.db == "mnist" or args.db == "fashion":
        return data[:, :-10], data[:, -10:].argmax(axis=1)
    else:
        return data[:, :-2], data[:, -2:].argmax(axis=1).ravel()



if __name__ == "__main__":
    
    # for mnist and fashion, we use neural network classifier
    if args.db == "mnist" or args.db == "fashion":
        classify = mnist_classification.classify
    # for table data, we use four classifiers
    else:
        classify = tabledata_classification.classify
    
    # load test data
    test_df = pd.read_csv(dataset_dir / "test.csv")
    
    # find synthetic data direction
    files = syn_data_dir.glob("out_*")
    
    for i, file in enumerate(files):
        print(f"synthetic data {i}")
        syn_df = pd.read_csv(file)
        
        # one hot encoding
        test_data, syn_data = to_one_hot(test_df, syn_df)

        # split data to data and label
        test_data, test_label = split(test_data)
        syn_data, syn_label = split(syn_data)
        
        result = classify(syn_data, syn_label, test_data, test_label)
        with (result_dir / f"result_{i}.json").open("w") as f:
            json.dump(result, f)