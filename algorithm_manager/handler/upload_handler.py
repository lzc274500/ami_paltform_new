import joblib
from flask import request, session, make_response
import sklearn
import json
import pandas as pd
import os
from common import tools
import time


'''
具体上传文件处理方法
上传成功的文件会保存在File文件夹下
'''
def upload():
    if 'file' not in request.files:
        re_dict = {"code": 400,
                   "message": "请传入文件",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data": []}
        return re_dict
    file = request.files.get('file')
    filename = file.filename
    path = os.path.join(tools.FilePath, filename)
    file.save(path)
    re_dict = {"code": 200,
               "message": "上传成功",
               "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
               "data": []}
    return re_dict

