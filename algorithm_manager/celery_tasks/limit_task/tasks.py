import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_path)
import time
import requests
import numpy as np
import pandas as pd
from celery_tasks.main import celery_app


@celery_app.task(name='callback_limit')
def callback_limit(json_data):
    file_url = json_data['file_url']
    callback_url = json_data['callback_url']
    username = json_data['username']
    try:
        data = pd.read_csv(file_url)
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username}]}
        requests.post(callback_url, json=re_dict)
        return
    start = data.columns.get_loc('Time')
    rate_dict = {}
    for i in data.columns[start+1:]:
        rate_list = []
        mean = data[i].mean()
        std = data[i].std()
        rate_list.append([round(mean-std,4),round(mean+std,4)])
        rate_list.append([round(mean-2*std,4),round(mean+2*std,4)])
        rate_list.append([round(mean-3*std,4),round(mean+3*std,4)])
        rate_dict[i] = rate_list
    alist = []
    rate_dict["username"] = username
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
    data_span = json_data['data_span']
    time_span = json_data['time_span']
    file_url = json_data['file_url']
    callback_url = json_data['callback_url']
    username = json_data['username']
    changeRate = json_data['changeRate']
    try:
        data = pd.read_csv(file_url)
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username}]}
        requests.post(callback_url, json=re_dict)
        return

    span = int(time_span / data_span)
    print(span)
    start = data.columns.get_loc('Time')
    rate_dict = {}
    for i in data.columns[start+1:]:
        df = (data[i]-data[i].shift(span))/time_span
        df = df.dropna().values
        rate_dict[i] = [round(np.min(df),4),round(np.max(df),4)]

    alist = []
    rate_dict['username'] = username
    rate_dict['changeRate'] = changeRate
    alist.append(rate_dict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    requests.post(callback_url, json=re_dict)
    return