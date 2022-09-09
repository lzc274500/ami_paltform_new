import joblib
import keras
import tensorflow as tf
from docx.shared import Inches
from flask import request, session, make_response
from sklearn import metrics,preprocessing
from keras.models import load_model
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from common import tools
from common.tools import logger
import time
from interface import predict_interface
from docx import Document
from cachetools import cached,LRUCache
from celery_tasks.validate_task import tasks

# 阈值
def get_threshold(test1,model_id):
    test = pd.DataFrame.copy(test1, deep=True)
    features = test.columns
    test['x'] = test[features[2]].rolling(100, 100).mean()

    test['yup'] = test[features[-1]].rolling(100,100).mean()+1.96*test[features[-1]].rolling(100,100).std()
    test['ydown'] = test[features[-1]].rolling(100,100).mean()-1.96*test[features[-1]].rolling(100,100).std()

    df = test.loc[test.index[0]+101::100]

    fup = np.polyfit(df['x'].values,df['yup'].values,6)
    pup = np.poly1d(fup)
    fdown = np.polyfit(df['x'].values,df['ydown'].values,6)
    pdown = np.poly1d(fdown)
    path1 = os.path.join(tools.ModelPath, model_id + '.u')
    joblib.dump(pup, path1)
    # print(pup(18))
    path2 = os.path.join(tools.ModelPath, model_id + '.d')
    joblib.dump(pdown, path2)
    return pup,pdown


def regression_predict(scaler_x,scaler_y,x_test,model):
    x_test = scaler_x.transform(x_test)
    y_predict = (model.predict(x_test)).reshape(-1,1)
    y_predict = scaler_y.inverse_transform(y_predict)
    logger.info(type(y_predict))
    return y_predict


def sequence_predict(x_test,model,scaler_x,scaler_y):
    s0, s1, s2 = x_test.shape[0], x_test.shape[1], x_test.shape[2]
    x_test = x_test.reshape(s0*s1, s2)
    x_test = scaler_x.transform(x_test)
    x_test = x_test.reshape(s0, s1, s2)
    y_predict = (model.predict(x_test))
    y_predict = scaler_y.inverse_transform(y_predict)
    return y_predict


def sequence_predict_intime(x_test,model,scaler_x,scaler_y):
    s0, s1, s2 = x_test.shape[0], x_test.shape[1], x_test.shape[2]
    x_test = x_test.reshape(s0*s1, s2)
    x_test = scaler_x.transform(x_test)
    x_test = x_test.reshape(s0, s1, s2)
    y_predict = model(x_test)
    y_predict = scaler_y.inverse_transform(y_predict)
    return y_predict


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

"""
评估模型的具体处理方法
path:模型路径
返回：损失函数mse
"""
def model_validate():
    json_data = request.get_json()
    try:
        filepath = json_data['filepath']
        model_id = json_data['model_id']
        algorithm = json_data['algorithm']
        input = json_data['input']
        output = json_data['output']
        callback_url = json_data['callback_url']
        username = json_data['username']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "请传入请求参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.validate_callback.delay(json_data,filepath,model_id,algorithm,input,output,callback_url,username)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "验证中"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = []
    return re_dict


@cached(cache=LRUCache(maxsize=256))
def get_model(model_id):
    path = os.path.join(tools.ModelPath, model_id + '.h5')
    model1 = load_model(path)
    return model1


def model_predict():
    try:
        json_data = request.get_json()
        logger.info(json_data)
        model_id = json_data['model_id']
        algorithm = json_data['algorithm']
        input = json_data['input']
    except Exception as e:
        logger.error(e)
        re_dict = {"code": 400,
                   "message": "参数有误",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    logger.info(type(input))

    path = os.path.join(tools.ModelPath, model_id+'.pkl')

    try:
        model = joblib.load(path)

    except Exception as e:
        re_dict = {"code": 400,
                   "message": "模型文件不存在",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict

    alist = []
    bdict = {}
    test = np.array(input)
    if algorithm in tools.reg_list:
        scalerx_path = os.path.join(tools.ModelPath, model_id + '.feature')
        scaler_x = joblib.load(scalerx_path)
        scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
        scaler_y = joblib.load(scalery_path)
        y_predict = regression_predict(scaler_x,scaler_y,test,model)
        bdict['predictValue'] = [round(i,4) for i in y_predict.flatten().tolist()]
        print(bdict['predictValue'])
    elif algorithm in tools.cls_list:
        y_predict = classification_predict(test,model)
        bdict['predictValue'] = y_predict.tolist()
    elif algorithm in tools.clu_list:
        y_predict = classification_predict(test,model)
        bdict['predictValue'] = y_predict.tolist()
    elif algorithm in tools.seq_list:
        test = np.array([test])
        tf.keras.backend.clear_session()

        # path = os.path.join(tools.ModelPath, model_id + '.h5')
        # model1 = load_model(path)
        print(time.time())
        model1 = get_model(model_id)
        print(time.time())
        scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
        scaler_y = joblib.load(scalery_path)
        print(time.time())
        y_predict = sequence_predict_intime(test,model1,model,scaler_y)
        print(time.time())
        bdict['predictValue'] = [round(i,4) for i in y_predict.flatten().tolist()]

    else:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "无相关算法"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = alist
        return re_dict
    alist.append(bdict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = bdict['predictValue'][0]
    return re_dict

