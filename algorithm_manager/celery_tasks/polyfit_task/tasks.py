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
from scipy.stats import pearsonr
from sklearn import preprocessing, linear_model
from sklearn.feature_selection import SelectKBest, f_regression


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


def get_progress(callback_url,username,analysisConfigId,progressBar,pathcall):
    re_dict = {"code": 200,
               "message": "进度信息",
               "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
               "data":[{"username":username,"analysisConfigId":analysisConfigId,"progressBar":progressBar,"path":pathcall}]}
    requests.post(callback_url, json=re_dict)
    time.sleep(5)

"""
一元多项式回归异步任务
"""
@celery_app.task(name='polyfit_callback')
def polyfit_callback(json_data):
    begin_time = time.time()
    file_url = json_data['file_url']
    independ = json_data['independ']
    depend = json_data['depend']
    multiples = json_data['multiple']
    callback_url = json_data['callback_url']
    username = json_data['username']
    analysisConfigId = json_data['analysisConfigId']
    filterCalc = json_data['filterCalc']
    pathcall = json_data['path']
    try:
        data = pd.read_csv(file_url,parse_dates = ['Time'],infer_datetime_format=True)
    except Exception as e:
        re_dict = {"code": 400,
                   "message": "读取文件失败",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"analysisConfigId":analysisConfigId,"path":pathcall}]}
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
                       "data":[{"username":username,"analysisConfigId":analysisConfigId,"path":pathcall}]}
            requests.post(callback_url, json=re_dict)
            return
    data_length = len(data.index.values.tolist())
    if data_length<10 or np.any(data.isnull()):
        re_dict = {"code": 400,
                   "message": "数据不完整",
                   "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                   "data":[{"username":username,"analysisConfigId":analysisConfigId,"path":pathcall}]}
        requests.post(callback_url, json=re_dict)
        return
    if time.time() - begin_time >= 5*60:
        get_progress(callback_url, username, analysisConfigId, "50",pathcall)
    data = data.sort_values(by=independ, axis=0, ascending=True, ignore_index=True)
    data_x = data[independ].values
    max_x = data[independ].max()
    min_x = data[independ].min()
    # 根据自变量范围划分100个区间
    cuts = pd.cut(data[independ], bins=100)
    groups = data.groupby(cuts)
    mean = groups.mean()
    std = groups.std()
    null_index = (groups.std()[groups.std().isnull().T.any()]).index
    mean = mean.drop(index=null_index)
    std = std.drop(index=null_index)
    mean[independ].values[0] = min_x
    mean[independ].values[-1] = max_x
    down_limit = mean - 1.96 * std
    up_limit = mean + 1.96 * std

    bdict = {}
    for i in range(len(multiples)):
        bdict[depend[i]] = {}
        multiple = multiples[i]
        data_y = data[depend[i]].values
        # 中线拟合
        polyfit_mid = np.polyfit(data_x,data_y,multiple)
        poly_mid = np.poly1d(polyfit_mid)
        y_mid = poly_mid(data_x)

        # describe_y = data[depend[i]].describe()
        # min = describe_y['min']
        # quartile = describe_y['25%']

        # 下线拟合
        polyfit_down = np.polyfit(mean[independ].values,down_limit[depend[i]].values,multiple)
        poly_down = np.poly1d(polyfit_down)
        y_down = poly_down(data_x)

        # 上线拟合
        polyfit_up = np.polyfit(mean[independ].values,up_limit[depend[i]].values,multiple)
        poly_up = np.poly1d(polyfit_up)
        y_up = poly_up(data_x)

        # 计算对应的劣化率
        badnesses = []
        for j in range(len(y_mid)):
            badness = get_badness(data_y[j],y_mid[j],y_up[j],y_down[j])
            badnesses.append(badness)
        bdict[depend[i]]['true'] = [round(i, 4) for i in data_y.tolist()]
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
    bdict["path"] = pathcall
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = [bdict]
    requests.post(callback_url, json=re_dict)
    return


def get_multiformula(coef_array,columns,intercept):
    formula = ''
    for i in range(len(coef_array)):
        if coef_array[i]>0 and i>0:
            a = "+"+str(coef_array[i])
        else:
            a = str(coef_array[i])
        formula+=a+"*"+columns[i]
    if intercept > 0:
        formula += "+"+str(intercept)
    else:
        formula += str(intercept)
    return formula

"""
多元多项式回归异步任务
"""
@celery_app.task(name='polyfit_callback2')
def polyfit_callback2(json_data):
    file_url = json_data['file_url']
    independ = json_data['independ']
    depend = json_data['depend']
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
        features = depend + independ
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
    # 计算与因变量最相关的自变量
    x_new = SelectKBest(f_regression, k=1)
    x_new.fit(data[independ], data[depend])
    index = x_new.get_support(indices=True)
    print(index)
    x_a = []  # 筛选后的自变量
    x_a.append(independ[index[0]])
    independ.pop(index[0])
    for i in range(len(independ)):
        pear = (pearsonr(data[independ[i]], data[x_a[0]])[0])
        if pear < 0.5:
            x_a.append(independ[i])
    print(x_a)
    x_b = x_a.copy()
    if len(x_b) > 1:
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly.fit(data[x_b].values)
        print(poly.powers_)
        x_train = poly.transform(data[x_b].values)
        for arr in poly.powers_:
            arr1 = np.where(arr == 1)[0]
            if len(arr1) > 1:
                x_b.append(x_b[arr1[0]] + "*" + x_b[arr1[1]])
    else:
        x_train = data[x_b].values

    df = pd.DataFrame(data=x_train, columns=x_b)
    df[depend] = data[depend]
    print(df.columns)
    model = linear_model.LinearRegression()
    model.fit(x_train,data[depend].values)
    #
    coef = model.coef_
    coef = coef.flatten().tolist()
    intercept_ = model.intercept_
    formula = get_multiformula(coef,df.columns[:-1],intercept_[0])
    print(formula)
    bdict = {}
    bdict['needIndepend'] = x_a
    bdict['formula'] = formula
    bdict["username"] = username
    bdict["analysisConfigId"] = analysisConfigId
    re_dict = {}
    re_dict["code"] = 200
    re_dict["message"] = "请求成功"
    re_dict["return_time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    re_dict["data"] = [bdict]
    requests.post(callback_url, json=re_dict)
    return

