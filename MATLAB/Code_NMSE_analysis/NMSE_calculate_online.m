
%% NMSE_calculate
% 计算网络预测输出结果的误差，并保存文件
% 只需改文件名

clear;
clc;
addpath(genpath('tensorlab_2016-03-28'));

%% 文件索引
sigma = 2;%dB
RR = 8;
snr = 20;
method_char = 'NMFCPD';  %DWCPD/ICPD  方法选择
test_char = 'sigma';    %sigma/trace/其他  ;对应实验参数设置是不同的sigma还是不同的轨迹trace
Sigma_index = sigma;   

trace = 2;
trace_index = trace;

%%
time_step = 10;
input_window = 20;
dw_active = time_step+input_window+15;

%% 导入X-ground truth   （路径需改）
%file_path_cpd = 'E:\hqy\Sigma_icpd_retry\MATLAB\';
file_path_cpd = ['E:\hqy\Mat_R',num2str(RR),'_SNR',num2str(snr),'_sinc2C\'];
%file_nameX = ['DWCPD_X_sigma',num2str(sigma),'.mat'];
file_nameX = ['Xtrue_',test_char,num2str(sigma),'_R',num2str(RR),'.mat'];


X_batch = load([file_path_cpd,file_nameX]);
Xtrue = X_batch.XtrueAll;
T = size(Xtrue,3)-dw_active+1;
Xtrue_active = Xtrue(:,:,dw_active:end);%45~1014,一共970个数据，网络从第44个数据开始online训练与预测

%% 导入AB  
% 注意A_batch{Sigma_index}，如果更改了要观察的参数种类（sigma/trace），需改index  
file_nameA = [method_char,'_S_A_',test_char,num2str(sigma),'_R',num2str(RR),'.mat'];
file_nameB = [method_char,'_S_B_',test_char,num2str(sigma),'_R',num2str(RR),'.mat'];

A_batch = load([file_path_cpd,file_nameA]);
A_cpd = A_batch.Ai_R_T;
A_cpd_active = A_cpd(:,dw_active:end);

B_batch = load([file_path_cpd,file_nameB]);
B_cpd = B_batch.Bi_R_T;
B_cpd_active = B_cpd(:,dw_active:end);

I = size(A_cpd_active{1},1);
J = size(B_cpd_active{1},1);

%% 导入PSD
file_namePSD =[method_char,'_PSD_',test_char,num2str(sigma),'_R',num2str(RR),'.mat'];
Cest = load([file_path_cpd,file_namePSD]);
Cest = Cest.Cest;

%% 导入C  （路径需改）
K=64;
epochs = 2;
%file_path_net = 'E:\hqy\Sigma_icpd_retry\NET\';
file_path_net = [file_path_cpd,'NEToutput\'];

test_char_C = [test_char,num2str(sigma)];

file_nameC_lstm = [method_char,'_LSTM_',test_char_C,'_lr=1e-4_online'];

%% 对比
Ts = size(Xtrue_active,3);
NMSE_lstm_tt = zeros(Ts,epochs);
for tt = 1:size(Xtrue_active,3)
    Xtrue_tt_mat = squeeze(Xtrue_active(:,:,tt));
    Xtrue_tt_tens = mat2tens(Xtrue_tt_mat,[I,J,K],[],3);
    for ee = 1:epochs
        Xhat_nmfcpd_lstm = zeros(I,J,K);
        for rr = 1:RR
            C_temp = load([file_path_net,file_nameC_lstm,'_',num2str(rr),'.mat']);
            C_temp = C_temp.C_pred;
            C_temp_tt = squeeze(C_temp(tt,ee,:));
            Shat_lstm_rr = A_cpd_active{rr,tt}*diag(C_temp_tt)*B_cpd_active{rr,tt}';
            
            Xhat_nmfcpd_lstm = Xhat_nmfcpd_lstm + outprod(Shat_lstm_rr,Cest(:,rr));
        end
        NMSE_lstm_tt(tt,ee) = frob(Xtrue_tt_tens - Xhat_nmfcpd_lstm).^2/frob(Xtrue_tt_tens).^2;
    end
end

%% 对比

%save_file_path = 'E:\hqy\Sigma_icpd_retry\NMSE\';
NMSE_time{1} = NMSE_lstm_tt;
save_file_path = [file_path_cpd,'result\'];
if ~exist(save_file_path,'dir')
    mkdir(save_file_path);
end
save_file_name = ['NMSE_',method_char,'_sigma',num2str(sigma),'_R',num2str(RR),'.mat'];
save([save_file_path,save_file_name],"NMSE_time");


