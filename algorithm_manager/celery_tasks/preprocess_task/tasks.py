import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_path)
import time
import requests
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from collections import OrderedDict
from celery_tasks.main import celery_app
from interface.preprocess_interface import filter_data


@celery_app.task(name='callback_corr')
def callback_corr(json_data):
    file_url = json_data['file_url']
    top_num = json_data['top_num']
    callback_url = json_data['callback_url']
    username = json_data['username']
    pathcall = json_data['path']
    if 'method' in json_data:
        method = json_data['method']
    else:
        method = 'pearson'
    try:
        data = pd.read_csv(file_url,parse_dates = ['Time'],infer_datetime_format=True)
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":{"username":username,"path":pathcall}}
        requests.post(callback_url, json=re_dict)
        return

    feature_length = len(data.columns.values.tolist())
    data_length = len(data.index.values.tolist())
    print(data_length)
    if data_length<100 or np.any(data.isnull()) or feature_length<2:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "数据不足"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = {"username":username,"path":pathcall}
        requests.post(callback_url, json=re_dict)
        return

    data = data[(data.columns[1:])]

    df = data.corr(method=method)
    bdict = {}
    for j in df.columns.values:
        bdict[j] = OrderedDict()
        df1 = df[[j]].sort_values(by=j, ascending=False).drop(labels=j,axis=0).head(top_num)
        for i in df1.index.values:
            bdict[j][i] = float("%.4f"%0.0000) if pd.isna(df1.loc[i, j]) else float("%.4f"%df1.loc[i, j])
    bdict['username'] = username
    bdict['path'] = pathcall
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = bdict
    requests.post(callback_url, json=re_dict)


@celery_app.task(name='preprocess_callback')
def preprocess_callback(filename,input,output,callback_url,model_id,username,filterCalc,pathcall):
    try:
        print(time.localtime())
        data = pd.read_csv(filename,parse_dates= ['Time'],infer_datetime_format=True)
        print(time.localtime())
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "文件不存在",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
        requests.post(callback_url, json=re_dict)
        return
    features = input + output
    if filterCalc != []:
        data = filter_data(data,filterCalc,features)
        if data is None:
            re_dict = {"code": 400,
                       "message": "筛选规则剔除失败",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data":[{"username":username,"modelId":model_id,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return

    feature_length = len(features)
    data_length = len(data.index.values.tolist())
    if data_length<100 or np.any(data.isnull()) or feature_length<2:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "数据不足"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = [{"username":username,"modelId":model_id,"path":pathcall}]
        requests.post(callback_url, json=re_dict)
        return

    alist = []
    bdict = {}
    bdict[output[0]] = {}
    for i in range(len(input)):
        pear = (pearsonr(data[input[i]], data[output[0]])[0])
        bdict[output[0]][input[i]] = float("%.4f" % 0.0000) if pd.isna(pear) else float("%.4f" % pear)
    bdict['modelId'] = model_id
    bdict['username'] = username
    bdict['surplusCount'] = data_length
    bdict['path'] = pathcall
    alist.append(bdict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    requests.post(callback_url, json=re_dict)
    return