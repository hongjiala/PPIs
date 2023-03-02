
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from model_protein import get_model
import numpy as np

from keras.utils import plot_model
from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
K=10
import warnings

warnings.filterwarnings("ignore")
class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.hu = val_data[0]
        self.vi = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.hu,self.vi])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)

        self.model.save_weights("model/%s/%sModel%d.h5" % (self.name,self.name,epoch))


        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


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

        #test=np.load(Data_dir+'%s_test.npz'%name)

        t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        X_hu_val = np.vstack((X_hu_tra[-1*(i+1)*(X_hu_cross.shape[0])-1:-1*i*(X_hu_cross.shape[0])-1,:],X_hu_cross))
        X_vi_val = np.vstack((X_vi_tra[-1*(i+1)*(X_vi_cross.shape[0])-1:-1*i*(X_vi_cross.shape[0])-1,:],X_vi_cross))
        y_val = np.vstack((y_tra[-1*(i+1)*(y_cross.shape[0])-1:-1*i*(y_cross.shape[0])-1,:],y_cross))

        X_hu_tra = np.vstack((X_hu_tra[:-1*(i+1)*(X_hu_cross.shape[0])-2,:],X_hu_tra[-1*i*(X_hu_cross.shape[0])-1:,:]))
        X_vi_tra =np.vstack((X_vi_tra[:-1*(i+1)*(X_vi_cross.shape[0])-2,:],X_vi_tra[-1*i*(X_vi_cross.shape[0])-1:,:]))
        y_tra =np.vstack((y_tra[:-1*(i+1)*(y_cross.shape[0])-2,:],y_tra[-1*i*(y_cross.shape[0])-1:,:]))

        model=None
        model=get_model()


        model.summary()


        print("第%d次交叉测试-------------------------------------------------------"%i)
        print(np.shape(X_hu_tra))

        X_hu_tra=np.reshape(X_hu_tra,(np.shape(X_hu_tra)[0],np.shape(X_hu_tra)[1],1))
        X_vi_tra = np.reshape(X_vi_tra,(np.shape(X_vi_tra)[0],np.shape(X_vi_tra)[1],1))
        X_hu_val =np.reshape(X_hu_val,(np.shape(X_hu_val)[0],np.shape(X_hu_val)[1],1))
        X_vi_val =np.reshape(X_vi_val,(np.shape(X_vi_val)[0],np.shape(X_vi_val)[1],1))
        back = roc_callback(val_data=[X_hu_val, X_vi_val, y_val], name=name)


        history = model.fit([X_hu_tra, X_vi_tra], y_tra, validation_data=([X_hu_val, X_vi_val], y_val), epochs=25, batch_size=32,callbacks=[back])


        t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print("开始时间:"+t1+"结束时间："+t2)













