import pandas as pd
import numpy as np
import sklearn.preprocessing
import collections
import torch
import tensorflow as tf
import pathlib
import sys
import math
import json
filedir = pathlib.Path(__file__).resolve().parent
dataset_dir = filedir.parent.parent / "dataset"

# The method to load dataset and encode to one-hot vectors for catecorical values
# n_bins desgnates the number of bins for descritization of continuous value (1 means doing nothing).
def load_dataset(db, n_bins=1):
    df = pd.read_csv(dataset_dir / db / "data.csv")
    domain_dir = dataset_dir / db / "domain.json"
    
    with open(domain_dir, "r") as f:
        domain = json.load(f)

    data = []
    encoders = []
    counter = 0
    
    # The class to do nothing instead of a one-hot encoder for continuous data
    class FakeEncoder():
        def __init__(self):
            self.categories_ = [[0]]
        
        def transform(self, value):
            return value
            
        def inverse_transform(self, value):
            return value
        
        def fit(self, a):
            pass
        
    for key, dim in zip(df.columns, domain.values()):
        
        # if n_bins > 1 and dim == 1, then the continuous value is descritized.
        if (n_bins > 1) and (dim == 1):
            max = df[key].max()
            min = df[key].min()
            def discretizer(value):
                bin_size = (max - min)/n_bins
                if bin_size == 0:
                    return 0
                if value == max:
                    return n_bins -1
                return int((value-min)/bin_size)
            df[key] = np.array(df[key].map(discretizer)).reshape(-1,1)
            
        # categorical (descritized) value is encoded to a one-hot vector.
        if ((n_bins > 1) and (dim == 1)) or (dim != 1):
            attr = np.array(df[key]).reshape(-1,1)
            enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
            enc.fit(attr.reshape(-1,1))
            attr = np.array(enc.transform(attr).toarray())
            
        # else, do nothing.
        else:
            attr = np.array(df[key]).reshape(-1,1)
            enc = FakeEncoder()
            enc.fit(attr.reshape(-1,1))
            enc.categories_ = [[0]]
            attr = np.array(enc.transform(attr))
        
        encoders.append(enc)
        data.append(attr)

    return np.concatenate(data, axis=1), encoders


# The method to inverse the one-hot encoded data to categorical data
# This method takes one-hot encoders as input
def inverse(encoded_data, encoders):
    inverse_data = []

    count = 0
    for encoder in encoders:
        dim = len(encoder.categories_[0])
        inversed = encoder.inverse_transform(encoded_data[:, count:count+dim].reshape(-1,dim))
        inverse_data.append(inversed)
        count += dim
        
    return np.concatenate(inverse_data, axis=1)

# The method to make data loader.
def make_dataloader(train, batch_size, random_state):
    
    # The class of a data loader.
    # This class is an iterator which randomly picks up a batch from dataset until the number of iteration reaches an epoch.
    class Generator(collections.abc.Iterator):
        def __init__(self, train_dataset, batch_size, random_state):
            self.random_state = random_state
            self.batch_size = batch_size
            shape = train_dataset.shape
            self.train = train_dataset[:int(shape[0] / batch_size) * batch_size].reshape(int(shape[0] / batch_size), batch_size, *shape[1:])
            self.n_iteration = self.get_n_batch()
            self.counter = 0
            
        def __next__(self):
            randint = self.random_state.randint(self.get_n_batch())
            tensor = self.train[randint]
            self.counter += 1
            if self.counter == self.n_iteration:
                self.counter = 0
                raise StopIteration
            return torch.tensor(tensor, dtype=torch.float32)
        
        def get_n_batch(self):
            return len(self.train)
        
        def get_dimensionality(self):
            return len(self.train[0][0])
        
        def get_data(self):
            return np.concatenate(self.train, axis=0)
        
        def get_batch_size(self):
            return self.batch_size
        
        def get_data_size(self):
            return len(self.get_data())
        
        def get_current_index(self):
            return self.counter
    
    return Generator(train, batch_size, random_state)
