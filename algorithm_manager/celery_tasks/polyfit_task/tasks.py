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


# 方程转字符串
def get_formula(polyfit_up,multiple):
    formula = ''
    for i in range(multiple+1):
        if polyfit_up[i]>0 and i>0:
            a = "+"+str(polyfit_up[i])
        else:
            a = str(polyfit_up[i])
        # formula+=a+"*x^"+str(multiple-i)
        formula+=a+"*Math.pow(x,"+str(multiple-i)+")"
    return formula


# 劣化度计算
def get_badness(intime,mid,up,down):
    if intime >= up or intime <= down:
        badness = 1
    elif intime > mid:
        badness = (intime -mid)/(up-mid)
    elif intime < mid:
        badness = (intime-mid)/(down-mid)
    else:
        badness = 0
    return badness


@celery_app.task(name='polyfit_callback')
def polyfit_callback(json_data):
    file_url = json_data['file_url']
    independ = json_data['independ']
    depend = json_data['depend']
    multiples = json_data['multiple']
    callback_url = json_data['callback_url']
    username = json_data['username']
    analysisConfigId = json_data['analysisConfigId']
    filterCalc = json_data['filterCalc']
    try:
        data = pd.read_csv(file_url,parse_dates = ['Time'],infer_datetime_format=True)
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username}]}
        requests.post(callback_url, json=re_dict)
        return

    if filterCalc != []:
        features = depend + [independ]
        filterCalc = json_data['filterCalc']
        data = filter_data(data,filterCalc,features)
        if data is None:
            re_dict = {"code": 400,
                       "message": "筛选规则剔除失败",
                       "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       "data":[{"username":username,"analysisConfigId":analysisConfigId}]}
            requests.post(callback_url, json=re_dict)
            return
    data_length = len(data.index.values.tolist())
    if data_length<10 or np.any(data.isnull()):
        re_dict = {"code": 400,
                   "message": "数据不完整",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"analysisConfigId":analysisConfigId}]}
        requests.post(callback_url, json=re_dict)
        return
    data_x = data[independ].values
    bdict = {}
    for i in range(len(multiples)):
        bdict[depend[i]] = {}
        multiple = multiples[i]
        data_y = data[depend[i]].values
        # 中线拟合
        polyfit_mid = np.polyfit(data_x,data_y,multiple)
        poly_mid = np.poly1d(polyfit_mid)
        y_mid = poly_mid(data_x)

        describe_y = data[depend[i]].describe()
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

        # 计算对应的劣化率
        badnesses = []
        for j in range(len(y_mid)):
            badness = get_badness(data_y[j],y_mid[j],y_up[j],y_down[j])
            badnesses.append(badness)
        # bdict[depend[i]]['true'] = [round(i, 4) for i in data_y.tolist()]
        bdict[depend[i]]['mid'] = [round(i,4) for i in y_mid.tolist()]
        bdict[depend[i]]['up'] = [round(i,4) for i in y_up.tolist()]
        bdict[depend[i]]['down'] = [round(i,4) for i in y_down.tolist()]
        bdict[depend[i]]['badness'] = [round(i,4) for i in badnesses]
        bdict[depend[i]]['formula_up'] = get_formula(polyfit_up, multiple)
        bdict[depend[i]]['formula_mid'] = get_formula(polyfit_mid, multiple)
        bdict[depend[i]]['formula_down'] = get_formula(polyfit_down, multiple)
    bdict["independ"] = [round(i, 4) for i in data_x.tolist()]
    bdict["username"] = username
    bdict["analysisConfigId"] = analysisConfigId
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = [bdict]
    requests.post(callback_url, json=re_dict)
    return
