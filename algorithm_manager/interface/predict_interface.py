from sklearn import metrics,preprocessing


def regression_predict(x_test,model):
    std = preprocessing.MinMaxScaler()
    x_test = std.fit_transform(x_test)
    y_predict = (model.predict(x_test))
    return y_predict


def sequence_predict(x_test,model,model_scaler):
    s0, s1, s2 = x_test.shape[0], x_test.shape[1], x_test.shape[2]
    x_test = x_test.reshape(s0*s1, s2)
    std = preprocessing.MinMaxScaler()
    x_test = std.fit_transform(x_test)
    x_test = x_test.reshape(s0, s1, s2)
    y_predict = (model.predict(x_test))
    y_predict = model_scaler.inverse_transform(y_predict)
    return y_predict


def sequence_validation(y_true,y_predict,tlabel):
    mse = metrics.mean_squared_error(y_true, y_predict)
    mae = metrics.mean_absolute_error(y_true,y_predict)
    bdict = {}
    bdict['trainTime'] = tlabel
    bdict['trueValue'] = y_true.tolist()
    bdict['predictValue'] = y_predict.flatten().tolist()
    bdict['均方误差'] = mse
    bdict['平均绝对误差'] = mae
    return bdict


def regression_validation(y_test,y_predict,tlabel):
    mse = metrics.mean_squared_error(y_test, y_predict)
    mae = metrics.mean_absolute_error(y_test,y_predict)
    r2_score = metrics.r2_score(y_test,y_predict)
    bdict = {}
    bdict['trainTime'] = tlabel.astype(str).values.tolist()
    bdict['trueValue'] = y_test.values.flatten().tolist()
    bdict['predictValue'] = y_predict.tolist()

    bdict['mse'] = mse
    bdict['mae'] = mae
    bdict['r2Score'] = r2_score

    return bdict


def classification_validation(y_test,y_predict):
    # x_test = test[(test.columns[1:-1])]
    # y_test = test[(test.columns[-1])]
    # y_predict = regression_predict(x_test.values,model)

    report = metrics.classification_report(y_test, y_predict, labels=[0,1],output_dict=True)

    bdict = {}
    bdict = report
    return bdict


def cluster_validation(x_test,y_predict):

    score = metrics.silhouette_score(x_test, y_predict)
    bdict = {}
    bdict['轮廓系数'] = score
    return bdict