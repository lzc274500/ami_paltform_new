from flask import request, session, make_response
import time
from celery_tasks.polyfit_task import tasks


def polyfit_reg():
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        independ = json_data['independ']
        depend = json_data['depend']
        multiple = json_data['multiple']
        callback_url = json_data['callback_url']
        username = json_data['username']
        analysisConfigId = json_data['analysisConfigId']
        filterCalc = json_data['filterCalc']
        pathcall = json_data['path']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "缺少参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.polyfit_callback.delay(json_data)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = []
    return re_dict

def multi_polyfit():
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        independ = json_data['independ']
        depend = json_data['depend']
        callback_url = json_data['callback_url']
        username = json_data['username']
        analysisConfigId = json_data['analysisConfigId']
        filterCalc = json_data['filterCalc']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "缺少参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.polyfit_callback2.delay(json_data)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = []
    return re_dict