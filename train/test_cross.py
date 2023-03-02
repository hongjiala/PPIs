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
for model_trained in models:

    print("****************Testing %s cell line specific model on %s cell line****************" % (model_trained, model_trained))
    temp_auc=[]
    temp_auprc = []
    temp_f1 = []
    temp_accu = []
    temp_precision = []
    temp_recall = []
    temp_sensitivity = []
    temp_specificity = []
    for model_tested in models:
        model=None
        model=get_model()

        model.load_weights("model/%s/%s_Model_final1_cross.h5"%(model_trained,model_trained))

        test=np.load('data_new/%s_test.npz'%model_tested)
        human_test,virus_test,y_tes=test['human'],test['virus'],test['label']
        human_test=np.reshape(human_test,(human_test.shape[0],human_test.shape[1],1))
        virus_test = np.reshape(virus_test,(virus_test.shape[0],virus_test.shape[1],1))
        y_pred = model.predict([human_test,virus_test])

        #auc=roc_auc_score(y_tes, y_pred)
        aupr=average_precision_score(y_tes, y_pred)
        #y_pred = np.round(y_pred)
        #f1 = f1_score(y_tes, y_pred)
        #accu = accuracy_score(y_tes, y_pred)
        #precision = precision_score(y_tes, y_pred)
        #recall = recall_score(y_tes, y_pred)
        #fscore = (2 * precision * recall) / (precision + recall)
        #sn = sen(y_tes, y_pred,2)
        #sp =spe(y_tes, y_pred,2)
        #print("recall  ="+str(recall))
        #print("precision = "+str(precision))
        #print("fscore = "+str(fscore))
        #print("accu = "+str(accu))
        #print("auc = "+str(auc))
        #print("aupr = " + str(aupr))
        #print("SN = " + str(sn))

        #temp_auc.append(auc)

        #temp_f1.append(f1)
        #temp_accu.append(accu)
        #temp_precision.append(precision)
        #temp_recall.append(recall)
        #temp_sensitivity.append(sn)
        #temp_specificity.append(sp)
        print("%s'sAUPRC in %s's test is %lf:" % (model_trained,model_tested,aupr))


