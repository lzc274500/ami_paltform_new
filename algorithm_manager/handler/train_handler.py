import joblib
import requests
from flask import request
from common import tools
from algo import algo
import os
import time
from celery_tasks.train import tasks
from handler.predict_handler import regression_predict, get_threshold, regression_validation
from handler.predict_handler import sequence_predict,sequence_validation


def callback_test(status):
    json_data = request.get_json()
    if status == 'train':
        # print(json_data)
        json_data = json_data['data'][0]
        trainTime = json_data['trainTime']
        trueValue = json_data['trueValue']
        predictValue = json_data['predictValue']
        modelId = json_data['modelId']
        algorithm = json_data['algorithm']

        print(trainTime[-5:])
        print(trueValue[-5:])
        print(predictValue[-5:])
        print(modelId)
        print(algorithm)
    else:
        print(json_data)

    re_dict = {"code": 200,
               "msg": "回调成功",
               "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
               "data":[]}
    return re_dict

"""
训练的路由处理方法
split：划分的数据集的比例
训练完成后会在model文件夹下生成模型文件
"""
def lr_train(status):
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        algorithm = json_data['algorithm']
        task = json_data['task']
        model_id = json_data['model_id']
        callback_url = json_data['callback_url']
        username = json_data['username']
        pathcall = json_data['path']
        if status == "":
            input = json_data['input']
            output = json_data['output']
            split = json_data['split']
            split_method = json_data['split_method']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "缺少参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.train_callback2.delay(json_data,status)
    re_dict = {"code": 200,
               "message": "开始训练",
               "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
               "data": []}
    return re_dict


def cls_train(status):
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        algorithm = json_data['algorithm']
        task = json_data['task']
        model_id = json_data['model_id']
        callback_url = json_data['callback_url']
        username = json_data['username']
        pathcall = json_data['path']
        if status == "":
            input = json_data['input']
            output = json_data['output']
            split = json_data['split']
            split_method = json_data['split_method']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "缺少参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.traincls_callback.delay(json_data,status)
    re_dict = {"code": 200,
               "message": "开始训练",
               "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
               "data": []}
    return re_dict


def clu_train(status):
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        algorithm = json_data['algorithm']
        task = json_data['task']
        model_id = json_data['model_id']
        callback_url = json_data['callback_url']
        username = json_data['username']
        pathcall = json_data['path']
        if status == "":
            input = json_data['input']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "缺少参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.trainclu_callback.delay(json_data,status)
    re_dict = {"code": 200,
               "message": "开始训练",
               "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
               "data": []}
    return re_dict











