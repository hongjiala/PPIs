import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
from model_protein import get_model
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix
import pandas as pd


def sen(Y_test, Y_pred, n):  # n为分类数

    sen = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen[0]


def spe(Y_test, Y_pred, n):
    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe[0]


models = ["DENV","Hepatitis","Herpes","HIV","Influenza","Papilloma","SARS2","ZIKV"]
dic = {}
AUC = {}
AUPRC = {}
F1={}
ACCU={}
PRECISION = {}
RECALL={}
SENSITIVITY={}
SPECIFITY ={}
for name in models:
    train = np.load('data_new/%s_train.npz' % name)
    cross_valid = np.load('data_new/%s_cross_valid.npz' % name)
    # test=np.load(Data_dir+'%s_test.npz'%name)
    X_hu_tra, X_vi_tra, y_tra = train['human'], train['virus'], train['label']
    X_hu_cross, X_vi_cross, y_cross = cross_valid['human'], cross_valid['virus'], cross_valid['label']
    print('Training %s protein specific model ...............................' % name)

    temp_auc=[]
    temp_auprc = []
    temp_f1 = []
    temp_accu = []
    temp_precision = []
    temp_recall = []
    temp_sensitivity = []
    temp_specificity = []
    for i in range(10):
        model=None
        model=get_model()

        model.load_weights("model/%s/%s_Model_final%d_cross.h5"%(name,name,i))
        # test=np.load(Data_dir+'%s_test.npz'%name)

        X_hu_val = np.vstack(
            (X_hu_tra[-1 * (i + 1) * (X_hu_cross.shape[0]) - 1:-1 * i * (X_hu_cross.shape[0]) - 1, :], X_hu_cross))
        X_vi_val = np.vstack(
            (X_vi_tra[-1 * (i + 1) * (X_vi_cross.shape[0]) - 1:-1 * i * (X_vi_cross.shape[0]) - 1, :], X_vi_cross))
        y_val = np.vstack((y_tra[-1 * (i + 1) * (y_cross.shape[0]) - 1:-1 * i * (y_cross.shape[0]) - 1, :], y_cross))

        X_hu_val = np.reshape(X_hu_val, (np.shape(X_hu_val)[0], np.shape(X_hu_val)[1], 1))
        X_vi_val = np.reshape(X_vi_val, (np.shape(X_vi_val)[0], np.shape(X_vi_val)[1], 1))

        y_pred = model.predict([X_hu_val,X_vi_val])

        auc=roc_auc_score(y_val, y_pred)
        aupr=average_precision_score(y_val, y_pred)
        y_pred = np.round(y_pred)
        f1 = f1_score(y_val, y_pred)
        accu = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        sn = sen(y_val, y_pred,2)
        sp =spe(y_val, y_pred,2)
        #print("recall  ="+str(recall))
        #print("precision = "+str(precision))
        #print("fscore = "+str(fscore))
        #print("accu = "+str(accu))
        #print("auc = "+str(auc))
        #print("aupr = " + str(aupr))
        #print("SN = " + str(sn))

        temp_auc.append(auc)
        temp_auprc.append(aupr)
        temp_f1.append(f1)
        temp_accu.append(accu)
        temp_precision.append(precision)
        temp_recall.append(recall)
        temp_sensitivity.append(sn)
        temp_specificity.append(sp)

    AUPRC[name] = temp_auprc
    AUC[name] = temp_auc
    F1[name] = temp_f1
    ACCU[name] = temp_accu
    PRECISION[name] = temp_precision
    RECALL[name] = temp_recall
    SENSITIVITY[name]=temp_sensitivity
    SPECIFITY[name]=temp_specificity


for model_trained in models:
    print("AUPRC in %s:"%model_trained+str(sum(AUPRC[model_trained])/10))
    print("AUC in %s:"%model_trained+str(sum(AUC[model_trained])/10))
    print("F1 in %s:"%model_trained+str(sum(F1[model_trained])/10))
    print("ACCU in %s:"%model_trained+str(sum(ACCU[model_trained])/10))
    print("PRECISION in %s:"%model_trained+str(sum(PRECISION[model_trained])/10))
    print("SENSITIVITY in %s:" % model_trained + str(sum(SENSITIVITY[model_trained]) / 10))
    print("SPECIFICITY in %s:" % model_trained + str(sum(SPECIFITY[model_trained]) / 10))
    print("在%s中表现最好的AUPRC是%lf,是第%d个模型" % (model_trained,max(AUPRC[model_trained]), AUPRC[model_trained].index(max(AUPRC[model_trained]))))
    print("在%s中表现最好的AUC是%lf,是第%d个模型" % (model_trained,max(AUC[model_trained]), AUC[model_trained].index(max(AUC[model_trained]))))
    print("在%s中表现最好的F1是%lf,是第%d个模型" % (model_trained, max(F1[model_trained]), F1[model_trained].index(max(F1[model_trained]))))
    print("在%s中表现最好的ACCU是%lf,是第%d个模型" % (model_trained, max(ACCU[model_trained]), ACCU[model_trained].index(max(ACCU[model_trained]))))
    print("在%s中表现最好的PRECISION是%lf,是第%d个模型\n" % (model_trained, max(PRECISION[model_trained]), PRECISION[model_trained].index(max(PRECISION[model_trained]))))





