import pandas as pd
from scipy.stats import pearsonr
import requests
from common.tools import asynch
from collections import OrderedDict


@asynch
def callback_corr(data,top_num,callback_url ):

    data = data[(data.columns[1:])]
    df = data.corr()
    bdict = {}
    for j in df.columns.values:
        bdict[j] = OrderedDict()
        df1 = df[[j]].sort_values(by=j, ascending=False).drop(labels=j,axis=0).head(top_num)
        for i in df1.index.values:
            bdict[j][i] = float("%.4f"%0.0000) if pd.isna(df1.loc[i, j]) else float("%.4f"%df1.loc[i, j])
    r = requests.post(callback_url, json=bdict)


@asynch
def preprocess_callback(data,input,output,callback_url,model_id):
    # datadict = {}
    # alist = []
    bdict = {}
    bdict[output[0]] = {}
    for i in range(len(input)):
        pear = (pearsonr(data[input[i]], data[output[0]])[0])
        bdict[output[0]][input[i]] = float("%.4f"%0.0000) if pd.isna(pear) else float("%.4f"%pear)
    bdict['modelId'] = model_id
    # alist.append(bdict)
    # datadict['data'] = alist
    r = requests.post(callback_url, json=bdict)