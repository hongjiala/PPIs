import xlrd
import numpy as np
import xlwt
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pandas as pd
names = ["DENV","Hepatitis","Herpes","HIV","Influenza","Papilloma","SARS2","ZIKV"]
for name in names:
    # 读取句向量
    data = np.load('./vectors/%s_vector.npy'%name)
    initial_data = pd.read_excel('./initialdata/%s_data.xlsx'%name, sheet_name=0)
    rownum = initial_data.shape[0]
    human_data_array=[]
    virus_data_array=[]
    label_data_array=[]
    # 分别写入human,virus,label的数据
    print(data.shape)
    for i in range(rownum):
        human_data_array.append(data[int(initial_data.iloc[i,0]-1)])
        virus_data_array.append(data[int(initial_data.iloc[i,1]-1)])
        label_data_array.append((int(initial_data.iloc[i,2])))
    np.savez('./data_new/%s_data.npz'%name,
             human=human_data_array,
             virus=virus_data_array,
             label=label_data_array)