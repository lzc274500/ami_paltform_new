import logging
import os
import time

import numpy as np
import pandas as pd
from common import tools
from common.tools import asynch
import joblib
import requests
from sklearn import model_selection,preprocessing
from scipy.stats import pearsonr
from handler.predict_handler import regression_predict, get_threshold, regression_validation, classification_validation, \
    cluster_validation, classification_predict
from handler.predict_handler import sequence_predict,sequence_validation
from sklearn import model_selection,neighbors,linear_model,svm,preprocessing,cluster,tree
import xgboost as xgb
from algo import algo

# 特征变量提取
from interface.preprocess_interface import filter_data


def format_data(data):
    x = data[(data.columns[0:-1])]
    y = data[(data.columns[-1])]
    return x,y


# 分类特征提取
def format_clsdata(data):
    x = data[(data.columns[0:-2])]
    y = data[(data.columns[-2:])]
    return x,y


# 序列化特征提取
def format_seqdata(data,sequence_length,span):
    data_all = np.array(data[data.columns[1:]]).astype(float)
    # time_data = data['Time'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S')).values
    time_data = data['Time'].values.tolist()
    datax = []
    label = []
    timeseq = []

    for i in range(len(data_all) - sequence_length * span):
        datax.append(data_all[i: i + sequence_length * span:span])
        # label.append(data_all[:,-1][i + sequence_length])
        label.append(data[data.columns[-1]].values[i + sequence_length*span])
        timeseq.append(time_data[i + sequence_length * span])
    x = np.array(datax).astype('float64')
    y = np.array(label).astype('float64')
    return x,y,timeseq


# 分割数据
def split_data(split_method,split,x,y):
    if split_method == 'shuffle':
        # x = data[(data.columns[0:-1])]
        # y = data[(data.columns[-1])]
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=float(split))
    elif split_method == 'sort':
        # print(data.columns[0:-1])
        # x = data[(data.columns[0:-1])]
        # y = data[(data.columns[-1])]
        split = int(len(x) * (1 - (float(split))))
        x_train,y_train = x[:split],y[:split]
        x_test, y_test = x[split:], y[split:]
    return x_train, x_test, y_train, y_test


# 分割序列化数据
def split_seqdata(x,y,timeseq,split):
    split = int(len(x) * (1 - (float(split))))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]
    t_train, t_test = timeseq[:split], timeseq[split:]
    return x_train, x_test, y_train, y_test,t_train,t_test


def train_seq(data,sequence_length,split,model,epochs):
    x,y,timeseq = format_seqdata(data, sequence_length)
    x_train, x_test, y_train, y_test,t_train,t_test = split_seqdata(x, y, timeseq, split)
    s0, s1, s2 = x_train.shape[0], x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(s0 * s1, s2)
    scaler_x = preprocessing.MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_train = x_train.reshape(s0, s1, s2)
    scaler_y = preprocessing.MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    model.fit(x_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1)


@asynch
def train_callback(*args):
    logging.info('aaa')
    task,callback_url,data,status,username= args[0],args[1],args[2],args[3],args[4]
    logging.info(task)
    if task == 'seq':
        model,model_id,algorithm,sequence_length,split,epochs= args[5:]
        x, y, timeseq = format_seqdata(data, sequence_length)
        x_train, x_test, y_train, y_test, t_train, t_test = split_seqdata(x, y, timeseq, split)
        s0, s1, s2 = x_train.shape[0], x_train.shape[1], x_train.shape[2]
        x_train = x_train.reshape(s0 * s1, s2)
        scaler_x = preprocessing.StandardScaler()
        scaler_x.fit(x_train)
        x_train = scaler_x.transform(x_train)
        x_train = x_train.reshape(s0, s1, s2)
        scaler_y = preprocessing.StandardScaler()
        scaler_y.fit(y_train.reshape(-1, 1))
        y_train = scaler_y.transform(y_train.reshape(-1, 1))
        model.fit(x_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1)
        path = os.path.join(tools.ModelPath, model_id + '.h5')
        scalerx_path = os.path.join(tools.ModelPath, model_id + '.pkl')
        scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
        model.save(path)  # 模型保存
        # 归一化保存
        joblib.dump(scaler_x, scalerx_path)
        joblib.dump(scaler_y, scalery_path)
        y_predict = sequence_predict(x_test, model, scaler_x,scaler_y)
        bdict = sequence_validation(y_test, y_predict, t_test)
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm
    elif task == 'reg':
        model, model_id,algorithm, split_method,split= args[5:]
        x,y = format_data(data)
        x_train, x_test, y_train, y_test = split_data(split_method, split, x, y)
        test = pd.concat([x_test, y_test], axis=1)
        x_train = x_train[(x_train.columns[1:])]
        scaler_x = preprocessing.StandardScaler()
        scaler_x.fit(x_train.values)
        x_train = scaler_x.transform(x_train.values)
        # print(x_train)
        scaler_y = preprocessing.StandardScaler()
        scaler_y.fit(y_train.values.reshape(-1,1))
        y_train = scaler_y.transform(y_train.values.reshape(-1,1))
        model.fit(x_train,y_train)
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        scalerx_path = os.path.join(tools.ModelPath, model_id + '.feature')
        scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
        print("aaa")
        joblib.dump(model, path)
        joblib.dump(scaler_x, scalerx_path)
        joblib.dump(scaler_y, scalery_path)
        x_test = test[(test.columns[1:-1])]
        print(test.columns)
        y_predict = regression_predict(scaler_x,scaler_y,x_test.values,model)
        # y_test = test[[test.columns[-1]]]
        bdict = regression_validation(y_test,y_predict,test['Time'])
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm
    elif task == 'cls':
        model, model_id,algorithm, split_method,split= args[5:]
        x,y = format_data(data)
        x_train, x_test, y_train, y_test = split_data(split_method, split, x, y)
        test = pd.concat([x_test, y_test], axis=1)
        x_train = x_train[(x_train.columns[1:])]
        std = preprocessing.MinMaxScaler()
        x_train = std.fit_transform(x_train)
        print(x_train)
        model.fit(x_train,y_train)
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        print("aaa")
        joblib.dump(model, path)
        x_test = test[(test.columns[1:-1])]
        print(test.columns)
        y_predict = regression_predict(x_test.values,model)
        # y_test = test[[test.columns[-1]]]
        bdict = classification_validation(y_test,y_predict)
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm
    elif task == 'clu':
        model, model_id, algorithm = args[5:]
        x = data[(data.columns[1:])]
        std = preprocessing.StandardScaler()
        x = std.fit_transform(x)
        model.fit(x)
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        joblib.dump(model, path)
        test = data.copy(deep=True)
        x_test = test[(test.columns[1:])]
        y_predict = classification_predict(x_test.values, model)
        print(np.unique(y_predict))
        test['label'] = y_predict
        test1 = pd.DataFrame(data=None, columns=test.columns.values)
        for x in np.unique(y_predict):
            test1 = pd.concat([test1, test[test['label'] == x].sample(frac=0.1)])
        x_test = test1[(test1.columns[1:-1])]
        y_predict = test1[(test1.columns[-1])]
        bdict = cluster_validation(x_test,y_predict)
    bdict['username'] = username
    if status == 'offline':
        # 离线训练计算相关性
        input = data.columns[1:-1]

        output = data.columns[-1]
        print(output)

        alist = []
        for i in range(len(input)):
            blist = []
            pear = (pearsonr(data[input[i]], data[output])[0])
            pear = float("%.4f" % 0.0000) if pd.isna(pear) else float("%.4f" % pear)
            blist.append(input[i])
            blist.append(output)
            blist.append(pear)
            alist.append(blist)
        bdict['corr'] = alist
        bdict['split'] = split
    print(bdict['trainTime'][-5:])
    r = requests.post(callback_url, json=bdict)


@asynch
def train_callback2(json_data,status):
    file_url = json_data['file_url']
    algorithm = json_data['algorithm']
    task = json_data['task']
    model_id = json_data['model_id']
    callback_url = json_data['callback_url']
    username = json_data['username']

    try:
        data = pd.read_csv(file_url,parse_dates=['Time'],infer_datetime_format=True)
        data['Time'] = data['Time'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S'))
        print(data.columns)
    except Exception as e:
        print(e)
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id}]}
        requests.post(callback_url, json=re_dict)
        return
    if status == 'offline':
        start = data.columns.get_loc('Time')
        print(start)
        input = data.columns[start + 1:-1]
        output = data.columns[-1]
        split = 0.2
        split_method = 'sort'
    else:
        input = json_data['input']
        output = json_data['output']
        split = json_data['split']
        split_method = json_data['split_method']
    features = input + output
    if 'filterCalc' in json_data.keys():
        filterCalc = json_data['filterCalc']
        if json_data['filterCalc'] != []:
            data = filter_data(data,filterCalc,features)
            if data is None:
                re_dict = {"code": 400,
                           "message": "筛选规则剔除失败",
                           "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           "data":[{"username":username,"modelId":model_id}]}
                requests.post(callback_url, json=re_dict)
                return

    if np.any(data.isnull()):
        re_dict = {"code": 400,
                   "message": "数据不完整",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id}]}
        requests.post(callback_url, json=re_dict)
        return

    data = pd.concat([data['Time'], data[input], data[output]], axis=1)
    model = None
    # 回归问题
    if task == 'reg' and algorithm in tools.reg_list:
        if algorithm == 'linear':
            fit_intercept = json_data['fit_intercept']
            model = linear_model.LinearRegression(fit_intercept=fit_intercept)
        elif algorithm == 'svr':
            kernel = json_data['kernel']  # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
            gamma = json_data['gamma']
            c = json_data['C']
            model = svm.SVR(kernel=kernel,gamma=gamma,C=c)
        elif algorithm == 'xgbreg':
            max_depth = json_data['max_depth']
            learning_rate = json_data['learning_rate']
            model = xgb.XGBRegressor(max_depth=max_depth,learning_rate=learning_rate)
        if model is None:
            re_dict = {"code": 400,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": [{"username":username,"modelId":model_id}]}
            requests.post(callback_url, json=re_dict)
            return
        else:
            try:
                x, y = format_data(data)
                x_train, x_test, y_train, y_test = split_data(split_method, split, x, y)
                test = pd.concat([x_test, y_test], axis=1)
                x_train = x_train[(x_train.columns[1:])]
                scaler_x = preprocessing.StandardScaler()
                scaler_x.fit(x_train.values)
                x_train = scaler_x.transform(x_train.values)
                # print(x_train)
                scaler_y = preprocessing.StandardScaler()
                scaler_y.fit(y_train.values.reshape(-1, 1))
                y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
                model.fit(x_train, y_train)
                path = os.path.join(tools.ModelPath, model_id + '.pkl')
                scalerx_path = os.path.join(tools.ModelPath, model_id + '.feature')
                scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
                print("aaa")
                joblib.dump(model, path)
                joblib.dump(scaler_x, scalerx_path)
                joblib.dump(scaler_y, scalery_path)
                x_test = test[(test.columns[1:-1])]
                print(test.columns)
                y_predict = regression_predict(scaler_x, scaler_y, x_test.values, model)
                # y_test = test[[test.columns[-1]]]
                bdict = regression_validation(y_test, y_predict, test['Time'])
            except Exception as e:
                logging.info(e)
                re_dict = {"code": 400,
                           "message": "训练失败",
                           "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           "data": [{"username": username, "modelId": model_id}]}
                requests.post(callback_url, json=re_dict)
                return
            bdict['modelId'] = model_id
            bdict['algorithm'] = algorithm
    elif task == 'seq' and algorithm in tools.seq_list:
        try:
            loss = json_data['loss']
            optimizer = json_data['optimizer']
            epochs = int(json_data['epochs'])
            sequence_length = int(json_data['sequence_length'])
            interval = json_data['interval']
            future = json_data['future']
        except Exception as e:
            re_dict = {"code": 400,
                       "message": "缺少算法参数",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data":[{"username":username,"modelId":model_id}]}
            requests.post(callback_url, json=re_dict)
            return
        # if interval[-3:] != future[-3:] :
        #     re_dict = {"code": 400,
        #                "message": "请传入与采样间隔相同的时间单位",
        #                "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        #                "data":[{"username":username,"modelId":model_id}]}
        #     requests.post(callback_url, json=re_dict)
        #     return
        # else:
        try:
            interval = int(interval)
            future = int(future)
            if future == 0:
                future = interval
        except Exception as e:
            re_dict = {"code": 400,
                       "message": "请传入正确的间隔时间",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": [{"username": username, "modelId": model_id}]}
            requests.post(callback_url, json=re_dict)
            return
        span = future // interval
        print(span)
        if span <= 0 or future % interval != 0:
            re_dict = {"code": 400,
                       "message": "无法计算的间隔时间",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": [{"username": username, "modelId": model_id}]}
            requests.post(callback_url, json=re_dict)
            return
        input_shape = (sequence_length,len(input)+len(output))
        if algorithm == 'lstm':
            model = algo.lstm_model(input_shape,loss,optimizer)
        elif algorithm == 'gru':
            model = algo.lstm_model(input_shape, loss, optimizer)
        if model is None:
            re_dict = {"code": 400,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data":[{"username":username,"modelId":model_id}]}
            requests.post(callback_url, json=re_dict)
            return
        else:
            try:
                x, y, timeseq = format_seqdata(data, sequence_length,span-1)
                x_train, x_test, y_train, y_test, t_train, t_test = split_seqdata(x, y, timeseq, split)
                s0, s1, s2 = x_train.shape[0], x_train.shape[1], x_train.shape[2]
                x_train = x_train.reshape(s0 * s1, s2)
                scaler_x = preprocessing.StandardScaler()
                scaler_x.fit(x_train)
                x_train = scaler_x.transform(x_train)
                x_train = x_train.reshape(s0, s1, s2)
                scaler_y = preprocessing.StandardScaler()
                scaler_y.fit(y_train.reshape(-1, 1))
                y_train = scaler_y.transform(y_train.reshape(-1, 1))
                model.fit(x_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1)
                path = os.path.join(tools.ModelPath, model_id + '.h5')
                scalerx_path = os.path.join(tools.ModelPath, model_id + '.pkl')
                scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
                model.save(path)  # 模型保存
                # 归一化保存
                joblib.dump(scaler_x, scalerx_path)
                joblib.dump(scaler_y, scalery_path)
                y_predict = sequence_predict(x_test, model, scaler_x, scaler_y)
                bdict = sequence_validation(y_test, y_predict, t_test)
            except Exception as e:
                logging.info(e)
                re_dict = {"code": 400,
                           "message": "训练失败",
                           "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           "data": [{"username": username, "modelId": model_id}]}
                requests.post(callback_url, json=re_dict)
                return
            bdict['modelId'] = model_id
            bdict['algorithm'] = algorithm
    # 分类问题
    elif task == 'cls' and algorithm in tools.cls_list:
        if algorithm == 'dtc':
            criterion = json_data['criterion']
            max_depth = json_data['max_depth']
            model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,random_state=None)
        elif algorithm == 'logistic':
            solver = json_data['solver']  # 优化求解方法 solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
            penalty = json_data['penalty']  # 正则化方法 penalty : {'l1', 'l2', 'elasticnet', 'none'}
            model = linear_model.LogisticRegression(solver=solver, penalty=penalty, C = 1.0)
        if model is None:
            re_dict = {"code": 400,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": [{"username":username,"modelId":model_id}]}
            requests.post(callback_url, json=re_dict)
            return
        else:
            x, y = format_data(data)
            x_train, x_test, y_train, y_test = split_data(split_method, split, x, y)
            test = pd.concat([x_test, y_test], axis=1)
            x_train = x_train[(x_train.columns[1:])]
            scaler_x = preprocessing.StandardScaler()
            scaler_x.fit(x_train.values)
            x_train = scaler_x.transform(x_train.values)
            # print(x_train)
            scaler_y = preprocessing.StandardScaler()
            scaler_y.fit(y_train.values.reshape(-1, 1))
            y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
            model.fit(x_train, y_train)
            path = os.path.join(tools.ModelPath, model_id + '.pkl')
            scalerx_path = os.path.join(tools.ModelPath, model_id + '.feature')
            scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
            print("aaa")
            joblib.dump(model, path)
            joblib.dump(scaler_x, scalerx_path)
            joblib.dump(scaler_y, scalery_path)
            x_test = test[(test.columns[1:-1])]
            print(test.columns)
            y_predict = regression_predict(scaler_x, scaler_y, x_test.values, model)
            # y_test = test[[test.columns[-1]]]
            bdict = classification_validation(y_test, y_predict)
            bdict['modelId'] = model_id
            bdict['algorithm'] = algorithm
    # 聚类问题
    elif task == 'clu' and algorithm in tools.clu_list:
        x = data[(data.columns[1:])]
        std = preprocessing.StandardScaler()
        x = std.fit_transform(x)
        if algorithm == 'kmeans':
            n_clusters = json_data['n_clusters']
            init = json_data['init']
            model = cluster.KMeans(n_clusters=n_clusters,init=init,random_state=0)

        elif algorithm == 'meanshift':
            quantile = json_data['quantile']
            n_samples = json_data['n_samples']
            bandwidth = cluster.estimate_bandwidth(x, quantile=quantile, n_samples=n_samples)
            model = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

        if model is None:
            re_dict = {"code": 400,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data":[{"username":username,"modelId":model_id}]}
            requests.post(callback_url, json=re_dict)
            return
        else:
            model.fit(x)
            path = os.path.join(tools.ModelPath, model_id + '.pkl')
            joblib.dump(model, path)
            test = data.copy(deep=True)
            x_test = test[(test.columns[1:])]
            y_predict = classification_predict(x_test.values, model)
            print(np.unique(y_predict))
            test['label'] = y_predict
            test1 = pd.DataFrame(data=None, columns=test.columns.values)
            for x in np.unique(y_predict):
                test1 = pd.concat([test1, test[test['label'] == x].sample(frac=0.1)])
            x_test = test1[(test1.columns[1:-1])]
            y_predict = test1[(test1.columns[-1])]
            bdict = cluster_validation(x_test, y_predict)
    if status == 'offline':
        # 离线训练计算相关性
        input = data.columns[1:-1]

        output = data.columns[-1]
        print(output)
        alist = []
        for i in range(len(input)):
            blist = []
            pear = (pearsonr(data[input[i]], data[output])[0])
            pear = float("%.4f" % 0.0000) if pd.isna(pear) else float("%.4f" % pear)
            blist.append(input[i])
            blist.append(output)
            blist.append(pear)
            alist.append(blist)
        bdict['corr'] = alist
        bdict['split'] = split
    print(bdict['trainTime'][-5:])
    bdict['username'] = username
    alist = []
    alist.append(bdict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    requests.post(callback_url, json=re_dict)
