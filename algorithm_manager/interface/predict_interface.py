import os
import numpy as np
import pandas as pd
import requests
from sklearn import metrics,preprocessing
import logging
import joblib
from common import tools
from common.tools import asynch
from keras.models import load_model
import time


# 数字孪生预测
def regression_predict(scaler_x,scaler_y,x_test,model):
    x_test = scaler_x.transform(x_test)
    y_predict = (model.predict(x_test)).reshape(-1,1)
    y_predict = scaler_y.inverse_transform(y_predict)
    logging.info(type(y_predict))
    return y_predict


# 趋势预测算法预测
def sequence_predict(x_test,model,scaler_x,scaler_y):
    s0, s1, s2 = x_test.shape[0], x_test.shape[1], x_test.shape[2]
    x_test = x_test.reshape(s0*s1, s2)
    x_test = scaler_x.transform(x_test)
    x_test = x_test.reshape(s0, s1, s2)
    y_predict = (model.predict(x_test))
    y_predict = scaler_y.inverse_transform(y_predict)
    return y_predict


# 趋势预测算法评估
def sequence_validation(y_true,y_predict,tlabel):
    mse = metrics.mean_squared_error(y_true, y_predict)
    mae = metrics.mean_absolute_error(y_true,y_predict)
    r2_score = metrics.r2_score(y_true, y_predict)
    bdict = {}
    bdict['trainTime'] = np.array(tlabel).astype(str).tolist()
    bdict['trueValue'] = [round(i,2) for i in y_true.tolist()]
    # count_list = []
    # for i in bdict['trueValue'][-10:]:
    #     count = len(str(i).split(".")[1])
    #     count_list.append(count)
    # places = max(count_list)
    bdict['predictValue'] = [round(i,2) for i in y_predict.flatten().tolist()]
    bdict['mse'] = mse
    bdict['mae'] = mae
    bdict['r2Score'] = r2_score
    return bdict


# 数字孪生评估
def regression_validation(y_test,y_predict,tlabel):
    mse = metrics.mean_squared_error(y_test, y_predict)
    mae = metrics.mean_absolute_error(y_test,y_predict)
    r2_score = metrics.r2_score(y_test,y_predict)
    bdict = {}
    bdict['trainTime'] = tlabel.astype(str).values.tolist()
    bdict['trueValue'] = [round(i,2) for i in y_test.values.flatten().tolist()]
    # count_list = []
    # for i in bdict['trueValue'][-10:]:
    #     count = len(str(i).split(".")[1])
    #     count_list.append(count)
    # places = max(count_list)
    # print(places)
    # print("小数位数为：%s" % places)
    bdict['predictValue'] = [round(i,2) for i in y_predict.flatten().tolist()]
    bdict['mse'] = mse
    bdict['mae'] = mae
    bdict['r2Score'] = r2_score

    return bdict


def classification_predict(test,model):
    std = preprocessing.StandardScaler()
    x_test = std.fit_transform(test)
    y_predict = model.predict(x_test)
    return y_predict


def classification_validation(y_test,y_predict):
    report = metrics.classification_report(y_test, y_predict, labels=[0,1],output_dict=True)
    bdict = {}
    bdict = report
    return bdict


def cluster_validation(x_test,y_predict):

    score = metrics.silhouette_score(x_test, y_predict)
    bdict = {}
    bdict['轮廓系数'] = score
    return bdict


@asynch
def validate_callback(json_data,filepath,model_id,algorithm,input,output,callback_url,username):
    path = os.path.join(tools.ModelPath, model_id+'.pkl')
    try:
        model = joblib.load(path)

    except Exception as e:
        re_dict = {"code": 400,
                   "message": "模型文件不存在",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id}]}
        requests.post(callback_url, json=re_dict)
        return
    try:
        data = pd.read_csv(filepath,parse_dates=['Time'],infer_datetime_format=True)
        data['Time'] = data['Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
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
    test = pd.concat([data['Time'], data[input], data[output]], axis=1)
    alist = []
    bdict = {}
    if algorithm in tools.reg_list:
        scalerx_path = os.path.join(tools.ModelPath, model_id + '.feature')
        scaler_x = joblib.load(scalerx_path)
        scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
        scaler_y = joblib.load(scalery_path)
        x_test = test[(test.columns[1:-1])]
        print(test.columns)
        y_predict = regression_predict(scaler_x,scaler_y,x_test.values,model)
        y_test = test[[test.columns[-1]]]
        bdict = regression_validation(y_test,y_predict,test['Time'])
    elif algorithm in tools.cls_list:
        x_test = test[(test.columns[1:-1])]
        print(test.columns)
        y_predict = regression_predict(x_test.values,model)
        y_test = test[[test.columns[-1]]]
        bdict = classification_validation(y_test,y_predict)
    elif algorithm in tools.clu_list:
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
    elif algorithm in tools.seq_list:
        scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
        scaler_y = joblib.load(scalery_path)
        path = os.path.join(tools.ModelPath, model_id + '.h5')
        model1 = load_model(path)
        sequence_length = json_data['sequence_length']
        interval = json_data['interval']
        future = json_data['future']
        try:
            interval = int(interval)
            future = int(future)
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
        span = span-1
        data_all = np.array(test[test.columns[1:]]).astype(float)
        datax = []
        label = []
        timeseq = []

        for i in range(len(data_all) - sequence_length-span):
            datax.append(data_all[i: i + sequence_length])
            label.append(test[test.columns[-1]].values[i + sequence_length+span])
            timeseq.append(test[test.columns[0]].values[i + sequence_length+span])
        x = np.array(datax).astype('float64')
        y = np.array(label).astype('float64')
        y_predict = sequence_predict(x,model1,model,scaler_y)
        bdict = sequence_validation(y,y_predict,timeseq)
    bdict['modelId'] = model_id
    bdict['algorithm'] = algorithm
    bdict['username'] = username
    alist.append(bdict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    requests.post(callback_url, json=re_dict)