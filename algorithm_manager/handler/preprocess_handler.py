import joblib
import requests
from flask import request
import pandas as pd
import numpy as np
from common import tools
import os
import time

from collections import OrderedDict
from interface import preprocess_interface
from celery_tasks.preprocess_task import tasks


"""
数据相关
"""
def data_corr():
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        top_num = json_data['top_num']
        callback_url = json_data['callback_url']
        username = json_data['username']
        pathcall = json_data['path']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "请传入请求参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.callback_corr.delay(json_data)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = []
    return re_dict


def callback_corr1(*args):
    data,top_num,callback_url = args
    data = data[(data.columns[1:])]
    df = data.corr()
    alist = []
    bdict = {}
    for j in df.columns.values:
        bdict[j] = OrderedDict()
        df1 = df[[j]].sort_values(by=j, ascending=False).drop(labels=j,axis=0).head(top_num)
        for i in df1.index.values:
            bdict[j][i] = float("%.4f"%0.0000) if pd.isna(df1.loc[i, j]) else float("%.4f"%df1.loc[i, j])
    r = requests.post(callback_url, json=bdict)



"""
数据预处理的具体处理方法
input:传入的特征值，各个特征值以逗号分隔
output:传入的目标值
filename:传入的数据集名称
请求成功后返回两两特征间的皮尔逊相关相关系数
并在File文件夹下生成preprocess.csv
"""
def preprocess():
    json_data = request.get_json()
    try:
        input = json_data['input']
        filename = json_data['filename']
        output = json_data['output']
        model_id = json_data['model_id']
        username = json_data['username']
        callback_url = json_data['callback_url']
        filterCalc = json_data['filterCalc']
        pathcall = json_data['path']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "请传入请求参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.preprocess_callback.delay(filename,input,output,callback_url,model_id,username,filterCalc,pathcall)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "验证中"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = []
    return re_dict



