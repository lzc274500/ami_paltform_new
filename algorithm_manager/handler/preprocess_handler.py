import joblib
import requests
from flask import request, session, make_response,current_app
from scipy.stats import pearsonr
import sklearn
import json
import pandas as pd
import numpy as np
from common import tools
import os
import time
import seaborn as sns
from matplotlib import pyplot as plt
from collections import OrderedDict
from interface import preprocess_interface



"""
数据相关
"""
def data_corr():
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        top_num = json_data['top_num']
        callback_url = json_data['callback_url']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "请传入请求参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    try:
        data = pd.read_csv(file_url,parse_dates = ['Time'],infer_datetime_format=True)
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict

    feature_length = len(data.columns.values.tolist())
    data_length = len(data.index.values.tolist())
    print(data_length)
    if data_length<100 or np.any(data.isnull()) or feature_length<2:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "数据不足"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = []
        return re_dict
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = []
    return re_dict,data,top_num,callback_url


def callback_corr(*args):
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
    # callback_url = json_data['callback_url']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "请传入请求参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    # path = os.path.join(tools.FilePath, filename)
    try:
        print(time.localtime())
        data = pd.read_csv(filename,parse_dates= ['Time'],infer_datetime_format=True)
        print(time.localtime())
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "文件不存在",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    temppath = os.path.join(tools.TempPath, model_id)
    if not os.path.exists(temppath):
        os.mkdir(temppath)
    if output != '' or output is not None:
        output = json_data['output']
        # test = pd.concat([data['Time'],data[input], data[output]], axis=1)

    else:
        pass

    features = input + output
    # df = data[features].corr()
    # df = df[output].drop(labels=output)
    # plt.barh(df.index.values, df[output[0]].values, left=0)
    # plt.xticks(np.arange(-1.0, 1.0, 0.1), rotation=-90)
    # plt.savefig(os.path.join(temppath, '相关性柱状图.png'),bbox_inches='tight')
    # plt.close()
    # 抽样
    frac = 1
    feature_length = len(features)
    data_length = len(data.index.values.tolist())
    print(data_length)
    if data_length<100 or np.any(data.isnull()):
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "数据不足"
        re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        re_dict["data"] = []
        return re_dict
    if feature_length > 20:
        if data_length > 200000:
            frac = round(200000/data_length,1)
    elif feature_length > 10 and feature_length <= 20:
        if data_length > 400000:
            frac = round(400000/data_length,1)
    elif feature_length > 2 and feature_length <= 10:
        if data_length > 600000:
            frac = round(600000/data_length,1)
    print(frac)
    print(time.localtime())
    data = data.sample(frac=frac)
    print(len(data.index.values.tolist()))
    print(time.localtime())
    alist = []
    bdict = {}
    bdict[output[0]] = {}
    for i in range(len(input)):
        pear = (pearsonr(data[input[i]], data[output[0]])[0])
        bdict[output[0]][input[i]] = float("%.4f"%0.0000) if pd.isna(pear) else float("%.4f"%pear)
    bdict['modelId'] = model_id
    alist.append(bdict)
    # preprocess_interface.preprocess_callback(data,input,output,callback_url,model_id)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    return re_dict



