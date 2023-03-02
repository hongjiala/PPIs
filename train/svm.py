import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
import torch
import torch.nn as nn #引入神经网络模型
import torch.optim as optim #引入优化器
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.utils import shuffle #随机打乱顺序
from sklearn.neural_network import MLPRegressor #多元感知器
from sklearn.metrics import mean_squared_error, r2_score #定义误差
from sklearn.svm import SVR #向量支持机回归
from sklearn.ensemble import RandomForestRegressor #随机森林回归
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font',family='Times New Roman')
plt.rcParams['mathtext.fontset']='stix'
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from lmfit import Minimizer, Parameters, report_fit

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from model_protein import get_model
import numpy as np

from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
K=10
import warnings

warnings.filterwarnings("ignore")


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')




names = ["DENV","Hepatitis","Herpes","HIV","Influenza","Papilloma","SARS2","ZIKV"]

for name in names:
    train=np.load('data_new/%s_train.npz'%name)
    cross_valid = np.load('data_new/%s_cross_valid.npz'%name)
    #test=np.load(Data_dir+'%s_test.npz'%name)
    X_hu_tra,X_vi_tra,y_tra=train['human'],train['virus'],train['label']
    X_hu_cross, X_vi_cross, y_cross = cross_valid['human'], cross_valid['virus'], cross_valid['label']
    print('Training %s protein specific model ...............................' % name)
    for i in range(K):
        t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        X_hu_val = np.vstack(
            (X_hu_tra[-1 * (i + 1) * (X_hu_cross.shape[0]) - 1:-1 * i * (X_hu_cross.shape[0]) - 1, :], X_hu_cross))
        X_vi_val = np.vstack(
            (X_vi_tra[-1 * (i + 1) * (X_vi_cross.shape[0]) - 1:-1 * i * (X_vi_cross.shape[0]) - 1, :], X_vi_cross))
        y_val = np.vstack((y_tra[-1 * (i + 1) * (y_cross.shape[0]) - 1:-1 * i * (y_cross.shape[0]) - 1, :], y_cross))

        X_hu_tra = np.vstack(
            (X_hu_tra[:-1 * (i + 1) * (X_hu_cross.shape[0]) - 2, :], X_hu_tra[-1 * i * (X_hu_cross.shape[0]) - 1:, :]))
        X_vi_tra = np.vstack(
            (X_vi_tra[:-1 * (i + 1) * (X_vi_cross.shape[0]) - 2, :], X_vi_tra[-1 * i * (X_vi_cross.shape[0]) - 1:, :]))
        y_tra = np.vstack(
            (y_tra[:-1 * (i + 1) * (y_cross.shape[0]) - 2, :], y_tra[-1 * i * (y_cross.shape[0]) - 1:, :]))



        X_hu_tra = np.reshape(X_hu_tra, (np.shape(X_hu_tra)[0], np.shape(X_hu_tra)[1], 1))
        X_vi_tra = np.reshape(X_vi_tra, (np.shape(X_vi_tra)[0], np.shape(X_vi_tra)[1], 1))
        X_hu_val = np.reshape(X_hu_val, (np.shape(X_hu_val)[0], np.shape(X_hu_val)[1], 1))
        X_vi_val = np.reshape(X_vi_val, (np.shape(X_vi_val)[0], np.shape(X_vi_val)[1], 1))
        svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)



        print("第%d次交叉测试-------------------------------------------------------"%i)


        X_hu_tra=np.reshape(X_hu_tra,(X_hu_tra.shape[0],X_hu_tra.shape[1]))
        X_vi_tra = np.reshape(X_vi_tra,(X_vi_tra.shape[0],X_vi_tra.shape[1]))
        X_hu_val =np.reshape(X_hu_val,(X_hu_val.shape[0],X_hu_val.shape[1]))
        X_vi_val =np.reshape(X_vi_val,(X_vi_val.shape[0],X_vi_val.shape[1]))
        X_trains = np.append(X_hu_tra, X_vi_tra, axis=1)
        X_tests= np.append(X_hu_val,X_vi_val, axis=1)


        history = svr.fit(X_trains, y_tra)
        Y_pred = svr.predict(X_tests)

        print("AUPRC:",average_precision_score(y_val, Y_pred))
        wd="SVM_model/"
        dill.dump(svr, open(wd + str(name)+str(i)+".obj", "wb"))


        t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print("开始时间:"+t1+"结束时间："+t2)
