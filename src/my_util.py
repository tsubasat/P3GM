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
def load_dataset(db):
    df = pd.read_csv(dataset_dir / db / "data.csv")
    
    with open(dataset_dir / db / "domain.json", "r") as f:
        domain = json.load(f)

    data = []
    encoders = {}
        
    for key, dim in domain.items():
        
        if dim != 1:
            attr = np.array(df[key]).reshape(-1,1)
            enc = sklearn.preprocessing.OneHotEncoder()
            enc.fit(attr)
            attr = np.array(enc.transform(attr).toarray())
            encoders[key] = enc
        else:
            attr = np.array(df[key]).reshape(-1,1)
            encoders[key] = None

        data.append(attr)

    return np.concatenate(data, axis=1), encoders, list(domain.keys()), list(domain.values())

def load_test_dataset(db):
    df = pd.read_csv(dataset_dir / db / "test.csv")
    
    with open(dataset_dir / db / "domain.json", "r") as f:
        domain = json.load(f)

    data = []
    encoders = {}
        
    for key, dim in domain.items():
        
        if dim != 1:
            attr = np.array(df[key]).reshape(-1,1)
            enc = sklearn.preprocessing.OneHotEncoder()
            enc.fit(attr)
            attr = np.array(enc.transform(attr).toarray())
            encoders[key] = enc
        else:
            attr = np.array(df[key]).reshape(-1,1)
            encoders[key] = None

        data.append(attr)

    return np.concatenate(data, axis=1), encoders, list(domain.keys()), list(domain.values())


# The method to inverse the one-hot encoded data to categorical data
# This method takes one-hot encoders as input
def inverse(encoded_data, encoders):
    inverse_data = []
    
    count = 0
    for encoder in encoders.values():
        if encoder:
            dim = len(encoder.categories_[0])
            inversed = encoder.inverse_transform(encoded_data[:, count:count+dim].reshape(-1,dim))
            inverse_data.append(inversed)
        else:
            dim = 1
            inversed = encoded_data[:, count:count+dim].reshape(-1,dim)
            inverse_data.append(inversed)
        count += dim
        
    return np.concatenate(inverse_data, axis=1)

# The method to make data loader.
def make_dataloader(train, batch_size, random_state, multi_dataset=False):
    
    # The class of a data loader.
    # This class is an iterator which randomly picks up a batch from dataset until the number of iteration reaches an epoch.
    class Generator(collections.abc.Iterator):
        def __init__(self, train_dataset, batch_size, random_state, multi_dataset=False):
            self.random_state = random_state
            self.batch_size = batch_size
            self.multi_dataset = multi_dataset
            self.train = []
            if multi_dataset:
                for dataset in train_dataset:
                    shape = dataset.shape
                    self.train.append(dataset[:int(shape[0] / batch_size) * batch_size].reshape(int(shape[0] / batch_size), batch_size, *shape[1:]))
            else:
                shape = train_dataset.shape
                self.train = [train_dataset[:int(shape[0] / batch_size) * batch_size].reshape(int(shape[0] / batch_size), batch_size, *shape[1:])]
            self.n_iteration = self.get_n_batch()
            self.counter = 0
            
        def __next__(self):
            randint = self.random_state.randint(self.get_n_batch())
            tensor = [torch.tensor(train[randint], dtype=torch.float32) for train in self.train]
            self.counter += 1
            if self.counter == self.n_iteration:
                self.counter = 0
                raise StopIteration
            if not self.multi_dataset:
                return tensor[0]
            return tensor
        
        def get_n_batch(self):
            return len(self.train[0])
        
        def get_dimensionality(self):
            return len(self.train[0][0][0])
        
        def get_data(self):
            return np.concatenate(self.train[0], axis=0)
        
        def get_batch_size(self):
            return self.batch_size
        
        def get_data_size(self):
            return len(self.get_data())
        
        def get_current_index(self):
            return self.counter
    
    return Generator(train, batch_size, random_state, multi_dataset)

def norm_clipping(X):
    clipped = np.zeros(X.shape)
    for i in range(X.shape[0]):
        clipped[i] = X[i] / max(1, np.linalg.norm(X[i]))
    return clipped