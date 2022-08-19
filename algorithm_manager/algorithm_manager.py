#!/usr/bin/python
#coding:utf-8
import time
from datetime import timedelta
import flask
# import requests
from flask import Flask, render_template, session, request
from flask import jsonify
from flask import make_response
# from urllib3 import response
from handler import upload_handler, train_handler, preprocess_handler, predict_handler, limit_handler, polyfit_handler
#from flask_mongoengine import MongoEngine
from flask import send_file
import os
import json
import logging
import functools
from concurrent.futures import ThreadPoolExecutor
from common import tools


# 创建flask app对象
app = Flask('am',static_url_path='')

# mongodb数据库配置，暂无用到
app.config['MONGODB_SETTINGS'] = {
    'db':   'am',
    'host': '127.0.0.1',
    'port': 27018
}

# jsonfy中文乱码问题
app.config['JSON_AS_ASCII'] = False
logging.basicConfig(level=logging.DEBUG)
# session存储时间，暂无用到
# app.config['SECRET_KEY'] = '123456'
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

# db = MongoEngine(app)

# from flask_cors import CORS
# CORS(app, supports_credentials=True)
executor = ThreadPoolExecutor()

# 统一返回json格式文本
def response_return(re_msg):
    response = make_response(json.dumps(re_msg,ensure_ascii=False),re_msg['code'])
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    response.mimetype = 'application/json'
    return response


# cookie设置，暂无用到
def response_return_cookie(re_msg,user_id):
    response = make_response(jsonify(data=re_msg))
    response.set_cookie("user_id", user_id, max_age=3600*24*7)
    response.headers['Access-Control-Allow-Origin'] = ''
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response


# 上传文件路由
@app.route('/upload',methods=['POST'])
def upload():
    re_msg = upload_handler.upload()
    return response_return(re_msg)


# 文件下载
@app.route('/download/<string:id>/<string:filename>',methods=['POST'])
def sending_file(id,filename):
    if filename:
        path = os.path.join(tools.TempPath, id+'/'+filename)
        return send_file(path, download_name='file.pkl')


# 数据相关路由
@app.route('/datacorr',methods=['POST'])
def corr_data():
    args = preprocess_handler.data_corr()
    if type(args) is dict:
        re_msg = args
    else:
        re_msg,args = args[0],args[1:]
        if re_msg['code'] == 200:
            executor.submit(lambda p: preprocess_handler.callback_corr(*p), args)

    return response_return(re_msg)


# 预处理路由
@app.route('/preprocess',methods=['POST'])
def preprocess_data():
    re_msg = preprocess_handler.preprocess()

    return response_return(re_msg)


# 训练路由
# @app.route('/train',methods=['POST'])
# def train_lr():
#     args= train_handler.lr_train()
#     if type(args) is dict:
#         re_msg = args
#     else:
#         re_msg,args = args[0],args[1:]
#         if re_msg['code'] == 200:
#             executor.submit(lambda p: train_handler.fit_train(*p), args)
#
#     return response_return(re_msg)
@app.route('/train',defaults={'status': ''}, methods=['POST'])
@app.route('/train/<string:status>',methods=['POST'])
def train_lr(status):
    re_msg = train_handler.lr_train(status)

    return response_return(re_msg)


# 预测路由
@app.route('/predict',methods=['POST'])
def predict_model():
    re_msg = predict_handler.model_predict()
    return response_return(re_msg)


# 评估路由
@app.route('/validate',methods=['POST'])
def validate_model():
    re_msg = predict_handler.model_validate()
    return response_return(re_msg)


# 阈值三限路由
@app.route('/limit',methods=['POST'])
def limit_get():
    re_msg = limit_handler.get_limit()
    return response_return(re_msg)


# 变化率阈值路由
@app.route('/ratelimit',methods=['POST'])
def limit_rate():
    re_msg = limit_handler.get_changerate()
    return response_return(re_msg)


# 一元多项式回归路由
@app.route('/ployfit',methods=['POST'])
def fit_ploy():
    re_msg = polyfit_handler.polyfit_reg()
    return response_return(re_msg)


# 测试回调路由
@app.route('/callback/<string:status>',methods=['POST'])
def callback_test(status):
    re_msg = train_handler.callback_test(status)
    return response_return(re_msg)


# 程序主入口
if __name__ == '__main__':
    # app.run(host='192.168.1.72',port=8880)  # 服务器地址，端口号
    app.run(host='0.0.0.0',port=8880)
    # http_server = WSGIServer(("0.0.0.0", 8880), app)
    # print('启动成功')
    # http_server.serve_forever()