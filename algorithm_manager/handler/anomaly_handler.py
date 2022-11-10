from flask import request, session, make_response
import time
from celery_tasks.anomaly_task import tasks


def anomaly_detect():
    json_data = request.get_json()
    try:
        file_url = json_data['file_url']
        algorithm = json_data['algorithm']
        callback_url = json_data['callback_url']
        username = json_data['username']
        pathcall = json_data['path']
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "请传入请求参数",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[]}
        return re_dict
    tasks.callback_anomaly.delay(json_data)
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = []
    return re_dict