#!/usr/bin/python
#coding:utf-8
import uuid
from threading import Thread
import logging


logger = logging.getLogger('gunicorn.error')
FilePath = '/home/algorithm_manager/File'
ModelPath = '/home/Model'
TempPath = '/home/algorithm_manager/temp'

# windows测试
# FilePath = 'E:/PycharmProjects/nodiot-algorithm/algorithm_manager/File'
# ModelPath = 'E:/PycharmProjects/nodiot-algorithm/algorithm_manager/model'
# TempPath = 'E:/PycharmProjects/nodiot-algorithm/algorithm_manager/temp'

reg_list = ['linear','svr','xgbreg','rfreg','ridge','mlpreg']
cls_list = ['dtc','logistic','svc','knn']
clu_list = ['kmeans','meanshift']
seq_list = ['lstm','gru','rnn']


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

