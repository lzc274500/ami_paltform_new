import joblib
import requests
from flask import request, session, make_response, current_app

from sklearn import model_selection, neighbors, linear_model, svm, preprocessing, cluster, tree
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import json
import pandas as pd
import numpy as np
from common import tools
from algo import algo
import os
import time

from handler.predict_handler import regression_predict, get_threshold, regression_validation
from handler.predict_handler import sequence_predict, sequence_validation


def callback_test():
    json_data = request.get_json()
    print(json_data)
    re_dict = {"code": 200,
               "msg": "回调成功",
               "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
               "data": []}
    return re_dict


"""
线性回归训练的具体处理方法
split：划分的数据集的比例
测试集会保存在File文件夹下，用于模型评估阶段
训练完成后会在model文件夹下生成模型文件
"""


def lr_train():
    json_data = request.get_json()
    file_url = json_data['file_url']
    input = json_data['input']
    output = json_data['output']
    algorithm = json_data['algorithm']
    split = json_data['split']
    task = json_data['task']
    model_id = json_data['model_id']
    split_method = json_data['split_method']
    callback_url = json_data['callback_url']

    try:
        data = pd.read_csv(file_url, parse_dates=['Time'], infer_datetime_format=True)
    except Exception as e:
        re_dict = {"code": 404,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data": []}
        return re_dict
    data = pd.concat([data['Time'], data[input], data[output]], axis=1)
    if task == 'seq':
        sequence_length = json_data['sequence_length']
        data_all = np.array(data[data.columns[1:]]).astype(float)
        time_data = data[data.columns[0]].values.astype(str)
        datax = []
        label = []
        timeseq = []

        for i in range(len(data_all) - sequence_length):
            datax.append(data_all[i: i + sequence_length])
            # label.append(data_all[:,-1][i + sequence_length])
            label.append(data[data.columns[-1]].values[i + sequence_length])
            timeseq.append(time_data[i + sequence_length])
        x = np.array(datax).astype('float64')
        y = np.array(label).astype('float64')
    else:
        x = data[(data.columns[0:-1])]
        y = data[(data.columns[-1])]
    model = None
    if split_method == 'shuffle':
        # x = data[(data.columns[0:-1])]
        # y = data[(data.columns[-1])]
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=float(split))
    elif split_method == 'sort':
        print(data.columns[0:-1])
        # x = data[(data.columns[0:-1])]
        # y = data[(data.columns[-1])]
        split = int(len(x) * (1 - (float(split))))
        x_train, y_train = x[:split], y[:split]
        x_test, y_test = x[split:], y[split:]
        if task == 'seq':
            t_train, t_test = timeseq[split:], timeseq[:split]
    else:
        re_dict = {"code": 400,
                   "message": "请选择合适的划分方法",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data": []}
        return re_dict
    if task != "seq":
        test = pd.concat([x_test, y_test], axis=1)
        train = pd.concat([x_train, y_train], axis=1)

        # test.to_csv(os.path.join(tools.TempPath, model_id + '/test.csv'))
        print(x_train.columns)

        x_train = x_train[(x_train.columns[1:])]
    # 回归问题
    if task == 'reg' and algorithm in tools.reg_list:
        std = preprocessing.MinMaxScaler()
        x_train = std.fit_transform(x_train)
        # print(x_train)
        # y_train = std.fit_transform(y_train.values.reshape(-1,1))
        if algorithm == 'linear':
            fit_intercept = json_data['fit_intercept']
            model = linear_model.LinearRegression(fit_intercept=fit_intercept)
        elif algorithm == 'svr':
            kernel = json_data['kernel']  # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
            gamma = json_data['gamma']
            c = json_data['C']
            model = svm.SVR(kernel=kernel, gamma=gamma, C=c)
        elif algorithm == 'xgbreg':
            max_depth = json_data['max_depth']
            learning_rate = json_data['learning_rate']
            model = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate)
        if model is None:
            re_dict = {"code": 404,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": []}
            return re_dict
        else:
            model.fit(x_train, y_train)
    # 分类问题
    elif task == 'cls' and algorithm in tools.cls_list:

        std = preprocessing.StandardScaler()
        x_train = std.fit_transform(x_train)

        if algorithm == 'dtc':
            criterion = json_data['criterion']
            max_depth = json_data['max_depth']
            model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=None)
        elif algorithm == 'logistic':
            solver = json_data['solver']  # 优化求解方法 solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
            penalty = json_data['penalty']  # 正则化方法 penalty : {'l1', 'l2', 'elasticnet', 'none'}
            model = linear_model.LogisticRegression(solver=solver, penalty=penalty, C=1.0)
        if model is None:
            re_dict = {"code": 404,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": []}
            return re_dict
        else:
            model.fit(x_train, y_train)
    # 聚类问题
    elif task == 'clu' and algorithm in tools.clu_list:
        x = data[(data.columns[1:])]
        std = preprocessing.StandardScaler()
        x = std.fit_transform(x)

        if algorithm == 'kmeans':
            n_clusters = json_data['n_clusters']
            init = json_data['init']
            model = cluster.KMeans(n_clusters=n_clusters, init=init, random_state=0)

        elif algorithm == 'meanshift':
            quantile = json_data['quantile']
            n_samples = json_data['n_samples']
            bandwidth = cluster.estimate_bandwidth(x, quantile=quantile, n_samples=n_samples)
            model = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

        if model is None:
            re_dict = {"code": 404,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": []}
            return re_dict
        else:
            model.fit(x)
    # 序列问题
    elif task == 'seq' and algorithm in tools.seq_list:
        input_shape = x_train.shape[-2:]
        s0, s1, s2 = x_train.shape[0], x_train.shape[1], x_train.shape[2]
        loss = json_data['loss']
        optimizer = json_data['optimizer']
        x_train = x_train.reshape(s0 * s1, s2)
        scaler_x = preprocessing.MinMaxScaler()
        x_train = scaler_x.fit_transform(x_train)
        x_train = x_train.reshape(s0, s1, s2)
        scaler_y = preprocessing.MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
        if algorithm == 'lstm':
            model = algo.lstm_model(input_shape, loss, optimizer)

        if model is None:
            re_dict = {"code": 404,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": []}
            return re_dict
        else:
            model.fit(x_train, y_train, batch_size=512, epochs=1, validation_split=0.1)

    else:
        re_dict = {"code": 404,
                   "message": "请选择合适的算法",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data": []}
        return re_dict
    # modelname = algorithm + ".pkl"
    # path = os.path.join(tools.ModelPath, model_id+'.h5')
    if task == 'seq':
        path = os.path.join(tools.ModelPath, model_id + '.h5')
        path_scaler = os.path.join(tools.ModelPath, model_id + '.pkl')
        model.save(path)  # 模型保存
        joblib.dump(scaler_y, path_scaler)  # 归一化保存
    else:
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        joblib.dump(model, path)
    if task == 'reg':
        modelup, modeldown = get_threshold(train, model_id)
        bdict = regression_validation(test, model, modelup, modeldown)
    elif task == 'seq':
        y_predict = sequence_predict(x_test, model, scaler_y)
        bdict = sequence_validation(y_test, y_predict, t_test)
    bdict['modelId'] = model_id
    bdict['algorithm'] = algorithm

    re_dict = {"code": 200,
               "message": "训练完成",
               "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
               "data": []}
    # 'http://127.0.0.1:8880/callback'
    r = requests.post(callback_url, json=bdict)
    if r.status_code != 200:
        re_dict = {}
        re_dict["code"] = r.json()['code']
        re_dict["message"] = r.json()['msg']
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = []
        return re_dict
    return re_dict


def fit_train(task, callback_url, *args):
    if task == 'seq':
        model, model_id, x_train, y_train, x_test, y_test, scaler_y, t_test = args
        model.fit(x_train, y_train, batch_size=512, epochs=1, validation_split=0.1)
        path = os.path.join(tools.ModelPath, model_id + '.h5')
        path_scaler = os.path.join(tools.ModelPath, model_id + '.pkl')
        model.save(path)  # 模型保存
        joblib.dump(scaler_y, path_scaler)  # 归一化保存
        y_predict = sequence_predict(x_test, model, scaler_y)
        bdict = sequence_validation(y_test, y_predict, t_test)
    elif task == 'reg':
        model, model_id, x_train, y_train, x_test, y_test = args
        model.fit(x_train, y_train)
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        joblib.dump(model, path)
        bdict = regression_validation(test, model, modelup, modeldown)
    if task == 'reg':
        modelup, modeldown = get_threshold(train, model_id)
        bdict = regression_validation(test, model, modelup, modeldown)
    elif task == 'seq':
        y_predict = sequence_predict(x_test, model, scaler_y)
        bdict = sequence_validation(y_test, y_predict, t_test)

    r = requests.post(callback_url, json=bdict)
    if r.status_code != 200:
        re_dict = {}
        re_dict["code"] = r.json()['code']
        re_dict["message"] = r.json()['msg']
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = []
        return re_dict










