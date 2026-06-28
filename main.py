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

st.title("🌾 Pytorch Rice Classifier ")
st.write("This model predicts if the rice under test is either Jasmin or Gonen")
st.header("Used Dataset")
st.dataframe(df)

st.sidebar.title("Input Rice Properties")
col1, col2 = st.sidebar.columns(2)

with col1:
    area = st.number_input("Area", value=5000.0)
    major_axis = st.number_input("Major Axis Length", value=100.0)
    minor_axis = st.number_input("Minor Axis Length", value=40.0)
    eccentricity = st.number_input("Eccentricity", value=0.8)
    convex_area = st.number_input("Convex Area", value=5100.0)

with col2:
    equiv_dia = st.number_input("Equiv Diameter", value=80.0)
    extent = st.number_input("Extent", value=0.7)
    perimeter = st.number_input("Perimeter", value=300.0)
    roundness = st.number_input("Roundness", value=0.8)
    aspect_ratio = st.number_input("Aspect Ratio", value=2.0)

if st.sidebar.button("Predict Rice Type"):
    normalized_input = [
        area / 10210.0,
        major_axis / 183.21,
        minor_axis / 82.55,
        eccentricity / 0.96,
        convex_area / 11008.0,
        equiv_dia / 114.01,
        extent / 0.88,
        perimeter / 508.51,
        roundness / 0.90,
        aspect_ratio / 3.91
    ]

    input_tensor = torch.tensor([normalized_input], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob >= 0.5 else 0

    st.sidebar.subheader("Result:")
    if prediction == 1:
        st.sidebar.success(f"This is (Jasmine) Rice. Confidence: {prob * 100:.2f}%")
    else:
        st.sidebar.info(f"This is (Gonen) Rice। Confidence: {(1 - prob) * 100:.2f}%")
