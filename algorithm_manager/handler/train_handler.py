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
线性回归训练的具体处理方法
split：划分的数据集的比例
测试集会保存在File文件夹下，用于模型评估阶段
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
    #
    # try:
    #     data = pd.read_csv(file_url,parse_dates=['Time'],infer_datetime_format=True)
    #     data['Time'] = data['Time'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S'))
    #     print(data.columns)
    # except Exception as e:
    #     print(e)
    #     re_dict = {"code": 400,
    #                "message": "读取文件失败",
    #                "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                "data":[]}
    #     return re_dict
    # if np.any(data.isnull()):
    #     re_dict = {"code": 400,
    #                "message": "数据不完整",
    #                "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                "data":[]}
    #     return re_dict
    # if status == 'offline':
    #     start = data.columns.get_loc('Time')
    #     print(start)
    #     input = data.columns[start+1:-1]
    #     output = data.columns[-1]
    #     split = 0.2
    #     split_method = 'sort'
    # else:
    #     try:
    #         input = json_data['input']
    #         output = json_data['output']
    #         split = json_data['split']
    #         split_method = json_data['split_method']
    #     except Exception as e:
    #         re_dict = {"code": 400,
    #                    "message": "缺少参数",
    #                    "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                    "data": []}
    #         return re_dict
    #
    # data = pd.concat([data['Time'], data[input], data[output]], axis=1)
    # model = None
    # # 回归问题
    # if task == 'reg' and algorithm in tools.reg_list:
    #     if algorithm == 'linear':
    #         fit_intercept = json_data['fit_intercept']
    #         model = linear_model.LinearRegression(fit_intercept=fit_intercept)
    #     elif algorithm == 'svr':
    #         kernel = json_data['kernel']  # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
    #         gamma = json_data['gamma']
    #         c = json_data['C']
    #         model = svm.SVR(kernel=kernel,gamma=gamma,C=c)
    #     elif algorithm == 'xgbreg':
    #         max_depth = json_data['max_depth']
    #         learning_rate = json_data['learning_rate']
    #         model = xgb.XGBRegressor(max_depth=max_depth,learning_rate=learning_rate)
    #     if model is None:
    #         re_dict = {"code": 404,
    #                    "message": "请传入正确的算法名称",
    #                    "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                    "data": []}
    #         return re_dict
    #     else:
    #         re_dict = {"code": 200,
    #                    "message": "开始训练",
    #                    "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                    "data": []}
    #         # model.fit(x_train, y_train)
    #
    #         train_interface.train_callback(task,callback_url,data,status,username,model, model_id, algorithm, split_method, split)
    #         return re_dict
    # # 分类问题
    # elif task == 'cls' and algorithm in tools.cls_list:
    #
    #
    #     if algorithm == 'dtc':
    #         criterion = json_data['criterion']
    #         max_depth = json_data['max_depth']
    #         model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,random_state=None)
    #     elif algorithm == 'logistic':
    #         solver = json_data['solver']  # 优化求解方法 solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
    #         penalty = json_data['penalty']  # 正则化方法 penalty : {'l1', 'l2', 'elasticnet', 'none'}
    #         model = linear_model.LogisticRegression(solver=solver, penalty=penalty, C = 1.0)
    #     if model is None:
    #         re_dict = {"code": 404,
    #                    "message": "请传入正确的算法名称",
    #                    "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                    "data": []}
    #         return re_dict
    #     else:
    #         # model.fit(x_train, y_train)
    #         train_interface.train_callback(task,callback_url,data,status,username,model, model_id, algorithm,split_method,split)
    # # 聚类问题
    # elif task == 'clu' and algorithm in tools.clu_list:
    #     x = data[(data.columns[1:])]
    #     std = preprocessing.StandardScaler()
    #     x = std.fit_transform(x)
    #
    #     if algorithm == 'kmeans':
    #         n_clusters = json_data['n_clusters']
    #         init = json_data['init']
    #         model = cluster.KMeans(n_clusters=n_clusters,init=init,random_state=0)
    #
    #     elif algorithm == 'meanshift':
    #         quantile = json_data['quantile']
    #         n_samples = json_data['n_samples']
    #         bandwidth = cluster.estimate_bandwidth(x, quantile=quantile, n_samples=n_samples)
    #         model = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #
    #     if model is None:
    #         re_dict = {"code": 404,
    #                    "message": "请传入正确的算法名称",
    #                    "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                    "data":[]}
    #         return re_dict
    #     else:
    #         train_interface.train_callback(task,callback_url,data,status,username,model, model_id, algorithm)
    # # 序列问题
    # elif task == 'seq' and algorithm in tools.seq_list:
    #     loss = json_data['loss']
    #     optimizer = json_data['optimizer']
    #     epochs = int(json_data['epochs'])
    #     sequence_length = int(json_data['sequence_length'])
    #     input_shape = (sequence_length,len(input)+len(output))
    #
    #     if algorithm == 'lstm':
    #         model = algo.lstm_model(input_shape,loss,optimizer)
    #     elif algorithm == 'gru':
    #         model = algo.lstm_model(input_shape, loss, optimizer)
    #     if model is None:
    #         re_dict = {"code": 400,
    #                    "message": "请传入正确的算法名称",
    #                    "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                    "data":[]}
    #         return re_dict
    #     else:
    #         # model.fit(x_train, y_train, batch_size=512, epochs=1, validation_split=0.1)
    #         re_dict = {"code": 200,
    #                    "message": "开始训练",
    #                    "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                    "data": []}
    #         # model.fit(x_train, y_train)
    #         train_interface.train_callback(task, callback_url, data, status,username,model,model_id,algorithm,sequence_length,split,epochs)
    #         return re_dict
    #
    # else:
    #     re_dict = {"code": 400,
    #                "message": "请选择合适的算法",
    #                "return_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
    #                "data": []}
    #     return re_dict


def fit_train(*args):
    print('aaa')
    task,callback_url= args[0],args[1]
    print(task)
    if task == 'seq':
        model, model_id,algorithm, x_train, y_train, x_test, y_test,scaler_y,t_test,epochs= args[2:]
        model.fit(x_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1)
        path = os.path.join(tools.ModelPath, model_id + '.h5')
        path_scaler = os.path.join(tools.ModelPath, model_id + '.pkl')
        model.save(path)  # 模型保存
        joblib.dump(scaler_y, path_scaler)  # 归一化保存
        y_predict = sequence_predict(x_test, model, scaler_y)
        bdict = sequence_validation(y_test, y_predict, t_test)
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm
    elif task == 'reg':
        model, model_id,algorithm, x_train, y_train, test = args[2:]
        print(x_train)
        model.fit(x_train,y_train)
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        print("aaa")
        joblib.dump(model, path)
        x_test = test[(test.columns[1:-1])]
        print(test.columns)
        y_predict = regression_predict(x_test.values,model)
        y_test = test[[test.columns[-1]]]
        bdict = regression_validation(y_test,y_predict,test['Time'])
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm

    r = requests.post(callback_url, json=bdict)











