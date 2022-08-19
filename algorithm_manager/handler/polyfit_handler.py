import joblib
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
import time


def polyfit_reg():
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        independ = json_data['independ']
        depend = json_data['depend']
        multiple = json_data['multiple']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "缺少参数",
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
    if np.any(data.isnull()):
        re_dict = {"code": 400,
                   "message": "数据不完整",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    data_x = data[independ].values
    data_y = data[depend].values
    # 中线拟合
    polyfit_mid = np.polyfit(data_x,data_y,multiple)
    poly_mid = np.poly1d(polyfit_mid)
    y_mid = poly_mid(data_x)

    describe_y = data[depend].describe()
    min = describe_y['min']
    quartile = describe_y['25%']

    # 下线拟合
    polyfit_down = np.polyfit(data_x,data_y-(quartile-min),multiple)
    poly_down = np.poly1d(polyfit_down)
    y_down = poly_down(data_x)

    # 上线拟合
    polyfit_up = np.polyfit(data_x,data_y+(quartile-min),multiple)
    poly_up = np.poly1d(polyfit_up)
    y_up = poly_up(data_x)

    bdict = {}
    bdict['indepent'] = data_x.tolist()[-5:]
    bdict['depent'] = data_y.tolist()[-5:]
    bdict['mid'] = y_mid.tolist()[-5:]
    bdict['up'] = y_up.tolist()[-5:]
    bdict['down'] = y_down.tolist()[-5:]
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = [bdict]
    return re_dict

