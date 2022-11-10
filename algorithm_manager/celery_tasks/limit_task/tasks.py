import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_path)
import time
import requests
import numpy as np
import pandas as pd
from celery_tasks.main import celery_app
from interface.preprocess_interface import filter_data


@celery_app.task(name='callback_limit')
def callback_limit(json_data):
    file_url = json_data['file_url']
    input = json_data['input']
    callback_url = json_data['callback_url']
    username = json_data['username']
    filterCalc = json_data['filterCalc']
    pathcall = json_data['path']
    try:
        data = pd.read_csv(file_url)
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"path":pathcall}]}
        requests.post(callback_url, json=re_dict)
        return
    start = data.columns.get_loc('Time')
    features = data.columns[start+1:]
    if filterCalc != []:
        data = filter_data(data,filterCalc,features)
        if data is None:
            re_dict = {"code": 400,
                       "message": "筛选规则剔除失败",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data":[{"username":username,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
    feature_length = len(features)
    data_length = len(data.index.values.tolist())
    if data_length<2 or np.any(data.isnull()) or feature_length<1:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "数据不足"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = [{"username":username,"path":pathcall}]
        requests.post(callback_url, json=re_dict)
        return
    rate_dict = {}
    for i in input:
        rate_list = []
        mean = data[i].mean()
        std = data[i].std()
        rate_list.append([round(mean-std,4),round(mean+std,4)])
        rate_list.append([round(mean-2*std,4),round(mean+2*std,4)])
        rate_list.append([round(mean-3*std,4),round(mean+3*std,4)])
        rate_dict[i] = rate_list
    alist = []
    rate_dict["username"] = username
    rate_dict["filterCalc"] = filterCalc
    rate_dict["path"] = pathcall
    alist.append(rate_dict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    requests.post(callback_url, json=re_dict)
    return


@celery_app.task(name='callback_changerate')
def callback_changerate(json_data):
    file_url = json_data['file_url']
    input = json_data['input']
    callback_url = json_data['callback_url']
    username = json_data['username']
    filterCalc = json_data['filterCalc']
    pathcall = json_data['path']
    requireResult = json_data['requireResult']
    try:
        data = pd.read_csv(file_url)
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"path":pathcall}]}
        requests.post(callback_url, json=re_dict)
        return
    start = data.columns.get_loc('Time')
    features = data.columns[start+1:]
    if filterCalc != []:
        data = filter_data(data,filterCalc,features)
        if data is None:
            re_dict = {"code": 400,
                       "message": "筛选规则剔除失败",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data":[{"username":username,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
    feature_length = len(features)
    data_length = len(data.index.values.tolist())
    print(np.any(data.isnull()))
    if data_length<2 or np.any(data.isnull()) or feature_length<1:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "数据不足"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = [{"username":username,"path":pathcall}]
        requests.post(callback_url, json=re_dict)
        return
    rate_dict = {}
    if "UL" in requireResult:
        featureLimit = json_data['featureLimit']
        for i in input:
            rate_dict[i] = {}
            mean = data[i].mean()
            std = data[i].std()
            rate_dict[i]['limit1'] = [round(mean - std, 4), round(mean + std, 4)]
            rate_dict[i]['limit2'] = [round(mean - 2 * std, 4), round(mean + 2 * std, 4)]
            rate_dict[i]['limit3'] = [round(mean - 3 * std, 4), round(mean + 3 * std, 4)]
        rate_dict['featureLimit'] = featureLimit
    if "CH" in requireResult:
        data_span = json_data['data_span']
        time_span = json_data['time_span']
        changeRate = json_data['changeRate']
        span = int(time_span / data_span)
        print(span)
        for i in input:
            if i not in rate_dict:
                rate_dict[i] = {}
            df = (data[i]-data[i].shift(span))/time_span
            df = df.dropna().values
            rate_dict[i]['rate'] = [round(np.min(df),4),round(np.max(df),4)]
        rate_dict['changeRate'] = changeRate
    alist = []
    rate_dict['username'] = username
    rate_dict['filterCalc'] = filterCalc
    rate_dict['path'] = pathcall
    alist.append(rate_dict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    requests.post(callback_url, json=re_dict)
    return