import os
import math
import numpy as np
import pandas as pd
import time
from tqdm.notebook import tqdm

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') 
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

# Pytoch
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data as data
print("Using torch", torch.__version__)
torch.manual_seed(42)


class MockDataset(data.Dataset):

    def __init__(self,
                 features=2,
                 pos_n=100,
                 neg_n=100, 
                 pos_mean=100,
                 pos_std=25,
                 neg_mean=150,
                 neg_std=12):
        """
        Inputs:
            features - Number of features we want to generate
            pos_n - Numer of samples for positive class
            neg_n - Numer of samples for negative class
            pos_mean - mean value for positive class features
            neg_mean - mean value for negative class features
            pos_std - standard deviation for positive class features
            neg_std - standard deviation for negative class features
        """
        super().__init__()
        self.features = features
        self.pos_n = pos_n
        self.pos_mean = pos_mean
        self.pos_std = pos_std
        self.neg_n = neg_n
        self.neg_mean = neg_mean
        self.neg_std = neg_std
        self.size = self.pos_n + self.neg_n
        self.generate_data()
        
    def generate_col(self, mean, std, n):
        data_cols = np.empty(n).T
        for col in range(1,self.features):
            temp = np.random.normal(mean, std, n)
            data_cols = np.vstack((data_cols,temp))
        return data_cols.T

    def generate_data(self):     
        pos_data = self.generate_col(self.pos_mean, self.pos_std, self.pos_n)
        neg_data = self.generate_col(self.neg_mean, self.neg_std, self.neg_n)
        data = np.vstack((pos_data,neg_data))
        self.data = torch.Tensor(data).to(torch.float64)
        
        self.pos_label = torch.ones(self.pos_n)
        self.neg_label = torch.zeros(self.neg_n)
        self.label = torch.cat((self.pos_label, self.neg_label), dim=0).to(torch.float64)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
    
    def to_df(self, row_ids=False, label=False):
        df = pd.DataFrame(self.data.numpy())
        df.columns = [f"col_{c}" for c in df.columns]
        if row_ids:
            df["row_ids"] = df.index.values.T
        if label:
            df["label"] = self.label
        return df
    
    @staticmethod
    def sort_to_sequence(df, key_col, exclude_cols):
        df = df.copy()  
        cols_space = (df.shape[1]-len(exclude_cols))
        np_seq = np.empty((0, cols_space))  
        for key in tqdm(df[key_col]):
            df_temp = df[df[key_col]==key].drop(exclude_cols, axis = 1). \
            T.sort_values(by=key).reset_index()
            np_temp = df_temp.T.values[0].reshape(1,-1)
            np_seq = np.concatenate([np_seq,np_temp],axis=0)
        return np_seq

    
class MyModule(nn.Module):
    
    def __init__(self, n_input):
        super().__init__()
        self.fc1 = nn.Linear(n_input, 8)
        self.b1 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 4)
        self.b2 = nn.BatchNorm1d(4)
        self.fc3 = nn.Linear(4,3)
        self.b3 = nn.BatchNorm1d(3)
        self.fc4 = nn.Linear(3,1)
    
    def swish(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        x = self.swish(self.fc1(x))
        x = self.b1(x)
        x = self.swish(self.fc2(x))
        x = self.b2(x)
        x = self.swish(self.fc3(x))
        x = self.b3(x)
        x = self.fc4(x)
        return x
    

def visualize_data(data, label, x1:int=1, x2:int=2):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    size = np.min([data_0.shape[0], data_1.shape[0]])
    
    plt.figure(figsize=(4,4))
    plt.scatter(data_0[:size,x1], data_0[:size,x2], edgecolor=".6", label="Negative Class")
    plt.scatter(data_1[:size,x1], data_1[:size,x2], edgecolor=".2", label="Positive Class")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    model.train() 
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            preds = model(data_inputs.float())
            preds = preds.squeeze(dim=1) 

            loss = loss_module(preds, data_labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            

def eval_model(model, data_loader):
    model.eval()
    true_preds, num_preds = 0., 0.
    
    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            
            preds = model(data_inputs.float())
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)
            pred_labels = preds.round()
            
            true_preds += (pred_labels.float() == data_labels.float()).sum().float()
            num_preds += data_labels.shape[0]
            
    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")
