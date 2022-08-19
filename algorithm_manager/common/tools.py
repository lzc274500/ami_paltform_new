#!/usr/bin/python
#coding:utf-8
import platform
import uuid
import time
from datetime import datetime
from threading import Thread

# FilePath = '/home/algorithm_manager/File'
# ModelPath = '/home/algorithm_manager/model'
# TempPath = '/home/algorithm_manager/temp'

# windows测试
FilePath = 'E:/PycharmProjects/nodiot-algorithm/algorithm_manager/File'
ModelPath = 'E:/PycharmProjects/nodiot-algorithm/algorithm_manager/model'
TempPath = 'E:/PycharmProjects/nodiot-algorithm/algorithm_manager/temp'

reg_list = ['linear','svr','xgbreg']
cls_list = ['dtc','logistic']
clu_list = ['kmeans','meanshift']
seq_list = ['lstm','gru']


def get_id_bytype():
    uid_code = uuid.uuid1()
    return str(uid_code)


def asynch(fun):
    """
    多线程实现异步执行
    :param f:
    :return:
    """
    def wrapper(*args, **kwargs):
        thr = Thread(target=fun, args=args, kwargs=kwargs)
        thr.start()
    return wrapper