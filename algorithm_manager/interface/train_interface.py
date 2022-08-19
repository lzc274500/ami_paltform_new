import os
import numpy as np
import pandas as pd
from common import tools
from common.tools import asynch
import joblib
import requests
from sklearn import model_selection,preprocessing
from scipy.stats import pearsonr
from handler.predict_handler import regression_predict, get_threshold, regression_validation, classification_validation, \
    cluster_validation, classification_predict
from handler.predict_handler import sequence_predict,sequence_validation

# 特征变量提取
def format_data(data):
    x = data[(data.columns[0:-1])]
    y = data[(data.columns[-1])]
    return x,y


# 序列化特征提取
def format_seqdata(data,sequence_length):
    data_all = np.array(data[data.columns[1:]]).astype(float)
    time_data = data['Time'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S')).values
    print(time_data)
    datax = []
    label = []
    timeseq = []

    for i in range(len(data_all) - sequence_length):
        datax.append(data_all[i: i + sequence_length])
        # label.append(data_all[:,-1][i + sequence_length])
        label.append(data[data.columns[-1]].values[i + sequence_length])
        timeseq.append(time_data[i + sequence_length])
    x = np.array(datax).astype('float64')
    y = np.array(label).astype('float64')
    return x,y,timeseq


# 分割数据
def split_data(split_method,split,x,y):
    if split_method == 'shuffle':
        # x = data[(data.columns[0:-1])]
        # y = data[(data.columns[-1])]
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=float(split))
    elif split_method == 'sort':
        # print(data.columns[0:-1])
        # x = data[(data.columns[0:-1])]
        # y = data[(data.columns[-1])]
        split = int(len(x) * (1 - (float(split))))
        x_train,y_train = x[:split],y[:split]
        x_test, y_test = x[split:], y[split:]
    return x_train, x_test, y_train, y_test


# 分割序列化数据
def split_seqdata(x,y,timeseq,split):
    split = int(len(x) * (1 - (float(split))))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]
    t_train, t_test = timeseq[split:], timeseq[:split]
    return x_train, x_test, y_train, y_test,t_train,t_test


def train_seq(data,sequence_length,split,model,epochs):
    x,y,timeseq = format_seqdata(data, sequence_length)
    x_train, x_test, y_train, y_test,t_train,t_test = split_seqdata(x, y, timeseq, split)
    s0, s1, s2 = x_train.shape[0], x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(s0 * s1, s2)
    scaler_x = preprocessing.MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_train = x_train.reshape(s0, s1, s2)
    scaler_y = preprocessing.MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    model.fit(x_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1)


@asynch
def train_callback(*args):
    print('aaa')
    task,callback_url,data,status,username= args[0],args[1],args[2],args[3],args[4]
    print(task)

    if task == 'seq':
        model,model_id,algorithm,sequence_length,split,epochs= args[5:]
        x, y, timeseq = format_seqdata(data, sequence_length)
        x_train, x_test, y_train, y_test, t_train, t_test = split_seqdata(x, y, timeseq, split)
        s0, s1, s2 = x_train.shape[0], x_train.shape[1], x_train.shape[2]
        x_train = x_train.reshape(s0 * s1, s2)
        scaler_x = preprocessing.StandardScaler()
        scaler_x.fit(x_train)
        x_train = scaler_x.transform(x_train)
        x_train = x_train.reshape(s0, s1, s2)
        scaler_y = preprocessing.StandardScaler()
        scaler_y.fit(y_train.reshape(-1, 1))
        y_train = scaler_y.transform(y_train.reshape(-1, 1))
        model.fit(x_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1)
        path = os.path.join(tools.ModelPath, model_id + '.h5')
        scalerx_path = os.path.join(tools.ModelPath, model_id + '.pkl')
        scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
        model.save(path)  # 模型保存
        # 归一化保存
        joblib.dump(scaler_x, scalerx_path)
        joblib.dump(scaler_y, scalery_path)
        y_predict = sequence_predict(x_test, model, scaler_x,scaler_y)
        bdict = sequence_validation(y_test, y_predict, t_test)
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm
    elif task == 'reg':
        model, model_id,algorithm, split_method,split= args[5:]
        x,y = format_data(data)
        x_train, x_test, y_train, y_test = split_data(split_method, split, x, y)
        test = pd.concat([x_test, y_test], axis=1)
        x_train = x_train[(x_train.columns[1:])]
        scaler_x = preprocessing.StandardScaler()
        scaler_x.fit(x_train)
        x_train = scaler_x.transform(x_train)
        # print(x_train)
        scaler_y = preprocessing.StandardScaler()
        scaler_y.fit(y_train.values.reshape(-1,1))
        y_train = scaler_y.transform(y_train.values.reshape(-1,1))
        model.fit(x_train,y_train)
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        scalerx_path = os.path.join(tools.ModelPath, model_id + '.feature')
        scalery_path = os.path.join(tools.ModelPath, model_id + '.label')
        print("aaa")
        joblib.dump(model, path)
        joblib.dump(scaler_x, scalerx_path)
        joblib.dump(scaler_y, scalery_path)
        x_test = test[(test.columns[1:-1])]
        print(test.columns)
        y_predict = regression_predict(scaler_x,scaler_y,x_test.values,model)
        # y_test = test[[test.columns[-1]]]
        bdict = regression_validation(y_test,y_predict,test['Time'])
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm
    elif task == 'cls':
        model, model_id,algorithm, split_method,split= args[5:]
        x,y = format_data(data)
        x_train, x_test, y_train, y_test = split_data(split_method, split, x, y)
        test = pd.concat([x_test, y_test], axis=1)
        x_train = x_train[(x_train.columns[1:])]
        std = preprocessing.MinMaxScaler()
        x_train = std.fit_transform(x_train)
        print(x_train)
        model.fit(x_train,y_train)
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        print("aaa")
        joblib.dump(model, path)
        x_test = test[(test.columns[1:-1])]
        print(test.columns)
        y_predict = regression_predict(x_test.values,model)
        # y_test = test[[test.columns[-1]]]
        bdict = classification_validation(y_test,y_predict)
        bdict['modelId'] = model_id
        bdict['algorithm'] = algorithm
    elif task == 'clu':
        model, model_id, algorithm = args[5:]
        x = data[(data.columns[1:])]
        std = preprocessing.StandardScaler()
        x = std.fit_transform(x)
        model.fit(x)
        path = os.path.join(tools.ModelPath, model_id + '.pkl')
        joblib.dump(model, path)
        test = data.copy(deep=True)
        x_test = test[(test.columns[1:])]
        y_predict = classification_predict(x_test.values, model)
        print(np.unique(y_predict))
        test['label'] = y_predict
        test1 = pd.DataFrame(data=None, columns=test.columns.values)
        for x in np.unique(y_predict):
            test1 = pd.concat([test1, test[test['label'] == x].sample(frac=0.1)])
        x_test = test1[(test1.columns[1:-1])]
        y_predict = test1[(test1.columns[-1])]
        bdict = cluster_validation(x_test,y_predict)
    bdict['username'] = username
    if status == 'offline':
        # 离线训练计算相关性
        input = data.columns[1:-1]

        output = data.columns[-1]
        print(output)

        alist = []
        for i in range(len(input)):
            blist = []
            pear = (pearsonr(data[input[i]], data[output])[0])
            pear = float("%.4f" % 0.0000) if pd.isna(pear) else float("%.4f" % pear)
            blist.append(input[i])
            blist.append(output)
            blist.append(pear)
            alist.append(blist)
        bdict['corr'] = alist
        bdict['split'] = split

    r = requests.post(callback_url, json=bdict)