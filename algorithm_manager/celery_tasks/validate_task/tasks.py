import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_path)
import time
import joblib
import numpy as np
import pandas as pd
import requests
from celery_tasks.main import celery_app
from common import tools
from keras.models import load_model
from interface.predict_interface import sequence_predict,sequence_validation,regression_predict,\
    regression_validation,classification_predict,classification_validation,cluster_validation


@celery_app.task(name='validate_callback')
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