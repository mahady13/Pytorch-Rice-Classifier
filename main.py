import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

df=pd.read_csv("riceClassification.csv")
df.drop(['id'],axis=1,inplace=True)

class MyModel(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, X):
        return self.net(X)

model = MyModel(input_dim=10)
model.load_state_dict(torch.load('rice_model.pth', map_location='cpu'))
model.eval()
