clc
clear
pos = xlsread('pos_hsbe.csv');
T = pos';
N = 1059;
k = 5;
type = "numeric";
AttVector = 0;
sample = SMOTE(T,N,k,type,AttVector);
T = T';
newsample = sample';
Pos_hsbe_sos = [newsample; T];
save Pos_hsbe_sos