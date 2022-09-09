import logging

import joblib
import requests
from flask import request, session, make_response,current_app
from scipy.stats import pearsonr
import sklearn
from sklearn import metrics,preprocessing
import json
import pandas as pd
import numpy as np
from common import tools
from common.tools import asynch
import os
import time
from celery_tasks.limit_task import tasks


@asynch
def callback_limit(json_data):
    file_url = json_data['file_url']
    callback_url = json_data['callback_url']
    username = json_data['username']
    try:
        data = pd.read_csv(file_url)
    except Exception as e:
        logging.info(e)
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


def get_limit():
    json_data = request.get_json()
    logging.info('a')
    try:
        file_url = json_data['file_url']
        callback_url = json_data['callback_url']
        username = json_data['username']
    except Exception as e:
        logging.info(e)
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "请传入参数"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = []
        return re_dict
    tasks.callback_limit.delay(json_data)
    alist = []
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    return re_dict


@asynch
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
        logging.info(e)
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


def get_changerate():
    json_data = request.get_json()
    try:
        data_span = json_data['data_span']
        time_span = json_data['time_span']
        file_url = json_data['file_url']
        callback_url = json_data['callback_url']
        username = json_data['username']
        changeRate = json_data['changeRate']
    except Exception as e:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "参数有误"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = []
        return re_dict
    remainder = time_span % data_span
    if remainder != 0:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "无法计算时间间隔"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = []
        return re_dict
    tasks.callback_changerate.delay(json_data)
    alist = []
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    return re_dict


