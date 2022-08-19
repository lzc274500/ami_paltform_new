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
import os
import time
import seaborn as sns
from matplotlib import pyplot as plt
from collections import OrderedDict


def get_limit():
    json_data = request.get_json()
    logging.info('a')
    try:
        file_url = json_data['file_url']
    except Exception as e:
        logging.info(e)
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "数据格式错误"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = []
        return re_dict
    try:
        data = pd.read_csv(file_url)
    except Exception as e:
        logging.info(e)
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    start = data.columns.get_loc('Time')
    limit1 = np.array([-1,1]).reshape(-1,1)
    limit2 = np.array([-2,2]).reshape(-1,1)
    limit3 = np.array([-3,3]).reshape(-1,1)
    rate_dict = {}
    for i in data.columns[start+1:]:
        rate_list = []
        std = preprocessing.StandardScaler()
        std.fit_transform(data[i].values.reshape(-1, 1))
        limit1 = std.inverse_transform(limit1)
        limit2 = std.inverse_transform(limit2)
        limit3 = std.inverse_transform(limit3)
        rate_list.append([round(i,4) for i in limit1.flatten().tolist()])
        rate_list.append([round(i,4) for i in limit2.flatten().tolist()])
        rate_list.append([round(i,4) for i in limit3.flatten().tolist()])
        rate_dict[i] = rate_list
    alist = []
    alist.append(rate_dict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    return re_dict


def get_changerate():
    json_data = request.get_json()
    try:
        data_span = json_data['data_span']
        time_span = json_data['time_span']
        input = json_data['input']
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
    span = int(time_span / data_span)
    print(span)
    data = np.array(input).astype(float)
    df = pd.DataFrame(data=data,columns=['sample'])
    print(df)
    df['rate'] = (df['sample'].shift(span)-df['sample'])
    print(df)
    df = df.drop(df.index[:span])
    upLimit = np.max(df['rate'].values)
    downLimit = np.min(df['rate'].values)
    alist = []
    bdict = {}
    bdict['upLimit'] = upLimit
    bdict['downLimit'] = downLimit
    alist.append(bdict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    return re_dict


