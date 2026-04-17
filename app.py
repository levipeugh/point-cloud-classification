import streamlit as st
import numpy as np
from plyfile import PlyData
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("3D Point Cloud Classifier")

def load_ply(file):
    ply = PlyData.read(file)
    vertex = ply['vertex']
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    return np.vstack([x, y, z]).T

def extract_features(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    stds = points.std(axis=0)
    centroid = points.mean(axis=0)

    return np.array([
        len(points),
        spans[0], spans[1], spans[2],
        stds[0], stds[1], stds[2],
        centroid[0], centroid[1], centroid[2]
    ])

# dummy training (just to show pipeline works)
X_demo = np.random.rand(20,10)
y_demo = np.array([0]*10 + [1]*10)

scaler = StandardScaler()
X_demo = scaler.fit_transform(X_demo)

model = LogisticRegression()
model.fit(X_demo, y_demo)

uploaded_file = st.file_uploader("Upload a .ply file", type=["ply"])

if uploaded_file:
    points = load_ply(uploaded_file)
    feats = extract_features(points).reshape(1, -1)
    feats = scaler.transform(feats)
    pred = model.predict(feats)[0]

    st.write("Prediction:", "Feasible" if pred == 1 else "Infeasible")
