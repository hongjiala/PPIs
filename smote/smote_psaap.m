%% 清空环境变量
close all;
clear;
clc;
format compact;
%% 下载数据
%n_data = xlsread("negative_data.xlsx");
%n_labels=xlsread("negtive_label.xlsx");
%p_data= xlsread("positive_data.xlsx");
%p_labels=xlsread("positive_label.xlsx");
%Cost=CostMatrix(2,5);
names = ["DENV","Hepatitis","Herpes","HIV","Influenza","Papilloma","SARS2","ZIKV"];
for i = 1:length(names)
load_name = names(i)+".mat";
load(load_name);
C=[1,0.543];%convert cost matrix to cost vector
data = [human virus];
p_data = data(1:sum(label),:);
n_data = data(sum(label)+1:end,:);
p_labels = ones(sum(label),1);
n_labels = zeros(length(label)-sum(label),1);
% 产生正类训练集和测试集
Train_p = p_data(1:floor(size(p_data,1)*3/5),:);
Test_p = p_data(floor(size(p_data,1)*3/5)+1:floor(size(p_data,1)*4/5),:);
cross_valid_p = p_data(floor(size(p_data,1)*4/5)+1:end,:);
% 相应的正类训练集和测试集的标签
Train_p_labels =p_labels(1:floor(size(p_labels,1)*3/5),:);
Test_p_labels = p_labels(floor(size(p_labels,1)*3/5)+1:floor(size(p_data,1)*4/5),:);
cross_valid_p_labels =  p_labels(floor(size(p_data,1)*4/5)+1:end,:);
% 产生负类训练集和测试集
Train_n = n_data(1:floor(size(n_data,1)*4/5),:);
Test_n = n_data(floor(size(n_data,1)*4/5)+1:floor(size(n_data,1)*4/5)+1+size(Test_p  ,1),:);

%相应的负类训练集和测试集的标签

Train_n_labels =n_labels(1:floor(size(n_data,1)*4/5),:);
Test_n_labels = n_labels(floor(size(n_data,1)*4/5)+1:floor(size(n_data,1)*4/5)+1+size(Test_p  ,1),:);

AttVector = zeros(1,6000);
ClassType=[0,1];
%训练数据和测试数据及相应标签
Train=[Train_p;Train_n];
Train_labels=[Train_p_labels;Train_n_labels];
Test=[Test_p;Test_n];
Test_labels=[Test_p_labels;Test_n_labels];
Train=Train';
Train_labels=Train_labels';
[newTrain,newTrainLabel]=SmoteOverSampling(Train,Train_labels ,ClassType,C,AttVector,8,'numeric');
newTrainTarget=LabelFormatConvertion(newTrainLabel,ClassType);% change to 2-value format
newTrain=newTrain';
newTrainLabel=newTrainLabel';
newtrain_Positive=[newTrain((newTrainLabel==1),:)];
newtrain_Positive_labels=ones(size(newtrain_Positive,1),1);
%组成平衡的新样本
data_new = [newtrain_Positive;Train_n(1:size(newtrain_Positive,1),:)];
label_new = [ones(size(newtrain_Positive,1),1);zeros(size(Train_n(1:size(newtrain_Positive,1),:),1),1)];
human_new=data_new(:,1:3000);
virus_new=data_new(:,3001:end);
human_test = Test(:,1:3000);
virus_test = Test(:,3001:end);
label_test = Test_labels;
human_cross = cross_valid_p(:,1:3000);
virus_cross = cross_valid_p(:,3001:end);
label_cross =cross_valid_p_labels ;
% %取4/5为训练集，1/5为测试集
% human_test = [human_new(1:floor(size(newtrain_Positive,1)/5),:);
%     human_new(size(newtrain_Positive,1)+1:size(newtrain_Positive,1)+1+floor(size(newtrain_Positive,1)/5),:)];
% virus_test = [virus_new(1:floor(size(newtrain_Positive,1)/5),:);
%     virus_new(size(newtrain_Positive,1)+1:size(newtrain_Positive,1)+1+floor(size(newtrain_Positive,1)/5),:)];
% label_test = [label_new(1:floor(size(newtrain_Positive,1)/5),:);
%     label_new(size(newtrain_Positive,1)+1:size(newtrain_Positive,1)+1+floor(size(newtrain_Positive,1)/5),:)];
% 
% human_train = [human_new(floor(size(newtrain_Positive,1)/5)+1:size(newtrain_Positive,1),:);
%     human_new(size(newtrain_Positive,1)+2+floor(size(newtrain_Positive,1)/5):end,:)];
% virus_train = [virus_new(floor(size(newtrain_Positive,1)/5)+1:size(newtrain_Positive,1),:);
%     virus_new(size(newtrain_Positive,1)+2+floor(size(newtrain_Positive,1)/5):end,:)];
% label_train = [label_new(floor(size(newtrain_Positive,1)/5)+1:size(newtrain_Positive,1),:);
%     label_new(size(newtrain_Positive,1)+2+floor(size(newtrain_Positive,1)/5):end,:)];
train_name = names(i)+"_train.mat";
test_name = names(i)+"_test.mat";
cross_valid_name = names(i)+"_cross_valid.mat";
save (train_name,'human_new', 'virus_new', 'label_new');
save (test_name,'human_test', 'virus_test', 'label_test');
save(cross_valid_name,'human_cross', 'virus_cross','label_cross');
end