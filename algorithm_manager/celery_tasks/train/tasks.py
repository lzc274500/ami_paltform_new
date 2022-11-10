import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_path)
import time
import numpy as np
import pandas as pd
import requests
from common.tools import logger
from celery_tasks.main import celery_app
from interface.preprocess_interface import filter_data
from interface.train_interface import format_data, format_seqdata, split_seqdata, split_data, format_clsdata
from common import tools
import xgboost as xgb
from algo import algo
import joblib
from sklearn import model_selection,neighbors,linear_model,svm,preprocessing,cluster,tree,ensemble,neural_network
from scipy.stats import pearsonr
from handler.predict_handler import regression_predict, regression_validation, classification_validation, \
    cluster_validation, classification_predict,sequence_predict,sequence_validation

"""
异步训练任务
"""
@celery_app.task(name='train_callback2')
def train_callback2(json_data,status):
    file_url = json_data['file_url']
    algorithm = json_data['algorithm']
    task = json_data['task']
    model_id = json_data['model_id']
    callback_url = json_data['callback_url']
    username = json_data['username']
    pathcall = json_data['path']

    try:
        data = pd.read_csv(file_url,parse_dates=['Time'],infer_datetime_format=True)
        data['Time'] = data['Time'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        logger.error(e)
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
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
                           "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
                requests.post(callback_url, json=re_dict)
                return
    data_length = len(data.index.values.tolist())
    if np.any(data.isnull()) or data_length<100:
        re_dict = {"code": 400,
                   "message": "数据不完整",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
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
        elif algorithm == 'rfreg':
            n_estimators = json_data['n_estimators']
            max_depth = json_data['max_depth']
            model = ensemble.RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
        elif algorithm == 'ridge':
            fit_intercept = json_data['fit_intercept']
            alpha = json_data['alpha']
            model = linear_model.Ridge(alpha=alpha,fit_intercept=fit_intercept)
        elif algorithm == 'mlpreg':
            activation = json_data['activation']
            max_iter = json_data['max_iter']
            model = neural_network.MLPRegressor(activation=activation,max_iter=max_iter)
        if model is None:
            re_dict = {"code": 400,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": [{"username":username,"modelId":model_id,"path":pathcall}]}
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
                joblib.dump(model, path)
                joblib.dump(scaler_x, scalerx_path)
                joblib.dump(scaler_y, scalery_path)
                x_test = test[(test.columns[1:-1])]
                print(test.columns)
                y_predict = regression_predict(scaler_x, scaler_y, x_test.values, model)
                # y_test = test[[test.columns[-1]]]
                bdict = regression_validation(y_test, y_predict, test['Time'])
            except Exception as e:
                logger.error(e)
                re_dict = {"code": 400,
                           "message": "训练失败",
                           "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           "data": [{"username": username, "modelId": model_id,"path":pathcall}]}
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
                       "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
        try:
            interval = int(interval)
            future = int(future)
            if future == 0:
                future = interval
        except Exception as e:
            re_dict = {"code": 400,
                       "message": "请传入正确的间隔时间",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": [{"username": username, "modelId": model_id,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
        span = future // interval
        print(span)
        if span <= 0 or future % interval != 0:
            re_dict = {"code": 400,
                       "message": "无法计算的间隔时间",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": [{"username": username, "modelId": model_id,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
        input_shape = (sequence_length,len(input)+len(output))
        if algorithm == 'lstm':
            model = algo.lstm_model(input_shape,loss,optimizer)
        elif algorithm == 'gru':
            model = algo.gru_model(input_shape, loss, optimizer)
        elif algorithm == 'rnn':
            model = algo.rnn_model(input_shape, loss, optimizer)
        if model is None:
            re_dict = {"code": 400,
                       "message": "请传入正确的算法名称",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
        else:
            try:
                x, y, timeseq = format_seqdata(data, sequence_length,span)
                x_train, x_test, y_train, y_test, t_train, t_test = split_seqdata(x, y, timeseq, split)
                s0, s1, s2 = x_train.shape[0], x_train.shape[1], x_train.shape[2]
                x_train = x_train.reshape(s0 * s1, s2)
                scaler_x = preprocessing.MinMaxScaler()
                scaler_x.fit(x_train)
                x_train = scaler_x.transform(x_train)
                x_train = x_train.reshape(s0, s1, s2)
                scaler_y = preprocessing.MinMaxScaler()
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
                logger.error(e)
                re_dict = {"code": 400,
                           "message": "训练失败",
                           "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           "data": [{"username": username, "modelId": model_id,"path":pathcall}]}
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
                       "data": [{"username":username,"modelId":model_id,"path":pathcall}]}
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
                joblib.dump(model, path)
                joblib.dump(scaler_x, scalerx_path)
                joblib.dump(scaler_y, scalery_path)
                x_test = test[(test.columns[1:-1])]
                y_predict = regression_predict(scaler_x, scaler_y, x_test.values, model)
                bdict = classification_validation(y_test, y_predict)
            except Exception as e:
                logger.error(e)
                re_dict = {"code": 400,
                           "message": "训练失败",
                           "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           "data": [{"username": username, "modelId": model_id,"path":pathcall}]}
                requests.post(callback_url, json=re_dict)
                return
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
                       "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
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
    bdict['path'] = pathcall
    alist = []
    alist.append(bdict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    requests.post(callback_url, json=re_dict)
    return


@celery_app.task(name='traincls_callback')
def traincls_callback(json_data,status):
    file_url = json_data['file_url']
    algorithm = json_data['algorithm']
    task = json_data['task']
    model_id = json_data['model_id']
    callback_url = json_data['callback_url']
    username = json_data['username']
    pathcall = json_data['path']

    try:
        data = pd.read_csv(file_url,parse_dates=['Time'],infer_datetime_format=True)
        data['Time'] = data['Time'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        logger.error(e)
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
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
    labels_cls = np.unique(data[output].values)
    print(labels_cls)
    if len(labels_cls)>20 or (len(labels_cls)>2 and algorithm =='logistic'):
        re_dict = {"code": 400,
                   "message": "无法分类的数据",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data": [{"username": username, "modelId": model_id, "path": pathcall}]}
        requests.post(callback_url, json=re_dict)
        return
    features = input + output
    if 'filterCalc' in json_data.keys():
        filterCalc = json_data['filterCalc']
        if json_data['filterCalc'] != []:
            data = filter_data(data,filterCalc,features)
            if data is None:
                re_dict = {"code": 400,
                           "message": "筛选规则剔除失败",
                           "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
                requests.post(callback_url, json=re_dict)
                return
    data_length = len(data.index.values.tolist())
    if np.any(data.isnull()) or data_length<100:
        re_dict = {"code": 400,
                   "message": "数据不完整",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
        requests.post(callback_url, json=re_dict)
        return

    encoder = preprocessing.LabelEncoder()
    encoder.fit(data[output].values)
    y_encoder = encoder.transform(data[output].values)
    data['encoder'] = y_encoder
    data = pd.concat([data['Time'], data[input], data[output],data['encoder']], axis=1)
    model = None
    if task == 'cls' and algorithm in tools.cls_list:
        if algorithm == 'dtc':
            criterion = json_data['criterion']
            max_depth = json_data['max_depth']
            model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,random_state=None)
        elif algorithm == 'logistic':
            solver = json_data['solver']  # 优化求解方法 solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
            penalty = json_data['penalty']  # 正则化方法 penalty : {'l1', 'l2', 'elasticnet', 'none'}
            model = linear_model.LogisticRegression(solver=solver, penalty=penalty, C = 1.0)
        elif algorithm == 'svc':
            kernel = json_data['kernel']  # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
            gamma = json_data['gamma']
            c = json_data['C']
            model = svm.SVC(kernel=kernel,gamma=gamma,C=c)
        elif algorithm == 'knn':
            n_neighbors = json_data['n_neighbors']  # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
            compute_method = json_data['compute_method']
            model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=compute_method)
        try:
            x, y = format_clsdata(data)
            x_train, x_test, y_train, y_test = split_data(split_method, split, x, y)
            test = pd.concat([x_test, y_test], axis=1)
            x_train = x_train[(x_train.columns[1:])]
            scaler_x = preprocessing.StandardScaler()
            scaler_x.fit(x_train.values)
            x_train = scaler_x.transform(x_train.values)
            # print(type(y_train))
            model.fit(x_train, y_train['encoder'].values)
            path = os.path.join(tools.ModelPath, model_id + '.pkl')
            scalerx_path = os.path.join(tools.ModelPath, model_id + '.feature')
            scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
            joblib.dump(model, path)
            joblib.dump(scaler_x, scalerx_path)
            joblib.dump(encoder, scalery_path)
            x_test = test[(test.columns[1:-2])]
            y_predict = regression_predict(scaler_x, encoder, x_test.values, model)
            bdict = classification_validation(y_test[output].values, y_predict,labels_cls)
        except Exception as e:
            logger.error(e)
            re_dict = {"code": 400,
                       "message": "训练失败",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data": [{"username": username, "modelId": model_id,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm
        bdict['username'] = username
        bdict['path'] = pathcall
        alist = []
        alist.append(bdict)
        re_dict = {}
        re_dict["code"] = 200
        re_dict["message"] = "请求成功"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = alist
        requests.post(callback_url, json=re_dict)
        return
    else:
        re_dict = {"code": 400,
                   "message": "请传入正确的算法名称",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data": [{"username": username, "modelId": model_id, "path": pathcall}]}
        requests.post(callback_url, json=re_dict)
        return


@celery_app.task(name='trainclu_callback')
def trainclu_callback(json_data,status):
    file_url = json_data['file_url']
    algorithm = json_data['algorithm']
    task = json_data['task']
    model_id = json_data['model_id']
    callback_url = json_data['callback_url']
    username = json_data['username']
    pathcall = json_data['path']

    try:
        data = pd.read_csv(file_url,parse_dates=['Time'],infer_datetime_format=True)
        data['Time'] = data['Time'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        logger.error(e)
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
        requests.post(callback_url, json=re_dict)
        return
    if status == 'offline':
        start = data.columns.get_loc('Time')
        input = data.columns[start + 1:]
    else:
        input = json_data['input']

    features = input
    if 'filterCalc' in json_data.keys():
        filterCalc = json_data['filterCalc']
        if json_data['filterCalc'] != []:
            data = filter_data(data,filterCalc,features)
            if data is None:
                re_dict = {"code": 400,
                           "message": "筛选规则剔除失败",
                           "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
                requests.post(callback_url, json=re_dict)
                return
    data_length = len(data.index.values.tolist())
    if np.any(data.isnull()) or data_length<100:
        re_dict = {"code": 400,
                   "message": "数据不完整",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
        requests.post(callback_url, json=re_dict)
        return

    data = pd.concat([data['Time'], data[input]], axis=1)
    model = None
    if task == 'clu' and algorithm in tools.clu_list:
        x = data[(data.columns[1:])]
        scaler_x = preprocessing.StandardScaler()
        scaler_x.fit(x)
        x = scaler_x.transform(x)
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
                       "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
        else:
            model.fit(x)
            path = os.path.join(tools.ModelPath, model_id + '.pkl')
            scaler_path = os.path.join(tools.ModelPath, model_id + '.feature')
            joblib.dump(model, path)
            joblib.dump(scaler_x, scaler_path)
            test = data.copy(deep=True)
            x_test = test[(test.columns[1:])]
            y_predict = classification_predict(x_test.values, model,scaler_x)
            print(np.unique(y_predict))
            # test['label'] = y_predict
            # test1 = pd.DataFrame(data=None, columns=test.columns.values)
            # for x in np.unique(y_predict):
            #     test1 = pd.concat([test1, test[test['label'] == x].sample(frac=0.1)])
            # x_test = test1[(test1.columns[1:-1])]
            # y_predict = test1[(test1.columns[-1])]
            # bdict = cluster_validation(x_test, y_predict)
            bdict ={}
            bdict['label'] = y_predict.flatten().tolist()
            bdict['modelId'] = model_id
            bdict['algorithm'] = algorithm
            bdict['username'] = username
            bdict['path'] = pathcall
            alist = []
            alist.append(bdict)
            re_dict = {}
            re_dict["code"] = 200
            re_dict["message"] = "请求成功"
            re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            re_dict["data"] = alist
            requests.post(callback_url, json=re_dict)
            return