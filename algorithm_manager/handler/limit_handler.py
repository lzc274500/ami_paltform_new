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


def get_limit():
    json_data = request.get_json()
    logging.info('a')
    try:
        file_url = json_data['file_url']
        input = json_data['input']
        callback_url = json_data['callback_url']
        username = json_data['username']
        filterCalc = json_data['filterCalc']
        pathcall = json_data['path']
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


def get_changerate():
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        input = json_data['input']
        callback_url = json_data['callback_url']
        username = json_data['username']
        filterCalc = json_data['filterCalc']
        pathcall = json_data['path']
        requireResult = json_data['requireResult']
        if 'UL' in requireResult:
            featureLimit = json_data['featureLimit']
        if 'CH' in requireResult:
            data_span = json_data['data_span']
            time_span = json_data['time_span']
            changeRate = json_data['changeRate']
            remainder = time_span % data_span
            if remainder != 0:
                re_dict = {}
                re_dict["code"] = 400
                re_dict["message"] = "无法计算时间间隔"
                re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                re_dict["data"] = []
                return re_dict
    except Exception as e:
        re_dict = {}
        re_dict["code"] = 400
        re_dict["message"] = "参数有误"
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


