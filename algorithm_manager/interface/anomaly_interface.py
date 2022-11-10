import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor,KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.covariance import EllipticEnvelope


def anomaly_lof(data,method):
    all_need = data.columns.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_arr = scaler.fit_transform(data.values)
    if len(all_need)>6:
        pca = PCA(n_components=0.95)
        scaler_arr = pca.fit_transform(scaler_arr)
    if method == 'lof':
        clf = LocalOutlierFactor(n_neighbors=20)
        labels = clf.fit_predict(scaler_arr)
    elif method == 'isof':
        clf = IsolationForest().fit(scaler_arr)
        labels = clf.predict(scaler_arr)
    elif method == 'ellenv':
        clf = EllipticEnvelope().fit(scaler_arr)
        labels = clf.predict(scaler_arr)
    return labels

