import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_path)
import time
import requests
import numpy as np
import pandas as pd
from celery_tasks.main import celery_app
from interface.anomaly_interface import anomaly_lof


@celery_app.task(name='callback_anomaly')
def callback_anomaly(json_data):
    file_url = json_data['file_url']
    algorithm = json_data['algorithm']
    callback_url = json_data['callback_url']
    username = json_data['username']
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
    labels = anomaly_lof(data[features],algorithm)
    alist = []
    anom_dict = {}
    anom_dict['label'] = labels.tolist()
    anom_dict["username"] = username
    anom_dict["path"] = pathcall
    alist.append(anom_dict)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = alist
    requests.post(callback_url, json=re_dict)
    return
