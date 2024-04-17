
%% NMSE_calculate
% 计算网络预测输出结果的误差，并保存文件
% 只需改文件名

clear;
clc;
addpath(genpath('tensorlab_2016-03-28'));

%% 文件索引
sigma = 6;%dB

method_char = 'DWCPD';  %DWCPD/ICPD  方法选择
test_char = 'sigma';    %sigma/trace/其他  ;对应实验参数设置是不同的sigma还是不同的轨迹trace
Sigma_index = sigma;   

trace = 2;
trace_index = trace;


T = 8000;
train_size = 0.9;
T_train = train_size*T;
batch_size = 50;
time_step = 20;

%%
dw_active = 15;

%% 导入X   （路径需改）
%file_path_cpd = 'E:\hqy\Sigma_icpd_retry\MATLAB\';
file_path_cpd = 'F:\黄清扬\Sigma_cpd\MATLAB\';
%file_nameX = ['DWCPD_X_sigma',num2str(sigma),'.mat'];
file_nameX = [method_char,'_X_',test_char,'.mat'];


X_batch = load([file_path_cpd,file_nameX]);
X_batch = X_batch.X_batch;
Xtrue = X_batch{Sigma_index};
Xtrue_active = Xtrue(:,:,dw_active:dw_active+T_train-1);

%% 导入AB  
% 注意A_batch{Sigma_index}，如果更改了要观察的参数种类（sigma/trace），需改index  
file_nameA = [method_char,'_A_',test_char,'.mat'];
file_nameB = [method_char,'_B_',test_char,'.mat'];

A_batch = load([file_path_cpd,file_nameA]);
A_batch = A_batch.A_batch;
A_dwcpd = A_batch{Sigma_index};
A_dwcpd_active = A_dwcpd(dw_active:dw_active+T_train-1);
B_batch = load([file_path_cpd,file_nameB]);
B_batch = B_batch.B_batch;
B_dwcpd = B_batch{Sigma_index};
B_dwcpd_active = B_dwcpd(dw_active:dw_active+T_train-1);

%% 导入C  （路径需改）
%file_path_net = 'E:\hqy\Sigma_icpd_retry\NET\';
file_path_net = 'F:\黄清扬\Sigma_cpd\NET\';


test_char_C = [test_char,num2str(sigma)];
% file_nameC_lstm = ['LSTM_',test_char,num2str(sigma),'_lr=1e-4.mat'];
% file_nameC_trans = ['Trans_',test_char,num2str(sigma),'_lr=1e-4.mat'];
file_nameC_lstm = [method_char,'_LSTM_',test_char_C,'_lr=1e-4_epoch=50.mat'];
file_nameC_trans = [method_char,'_Trans_',test_char_C,'_lr=1e-4_epoch=50.mat'];
C_lstm = load([file_path_net,file_nameC_lstm]);
C_trans = load([file_path_net,file_nameC_trans]);
C_trans = C_trans.C_pred;
C_lstm = C_lstm.C_pred;
[epochs,batches,batch_size,time_step,Rcp] = size(C_lstm);

%% 分time_step窗口
Xtrue_timestep = get_timestep_label(Xtrue_active,time_step);
Num_timestep = T_train-time_step;
Adwcpd_timestep = cell(1,Num_timestep);
Bdwcpd_timestep = cell(1,Num_timestep);
Adwcpd_timestep = A_dwcpd_active(time_step+1:end);
Bdwcpd_timestep = B_dwcpd_active(time_step+1:end);

%% 对比
% NMSE_lstm_tt = zeros(epochs,batches,batch_size);
% NMSE_trans_tt = zeros(epochs,batches,batch_size);
NMSE_lstm_tt = cell(epochs,batches);
NMSE_trans_tt = cell(epochs,batches);
NMSE_trans_time = zeros(epochs,Num_timestep);
NMSE_lstm_time = zeros(epochs,Num_timestep);
batch_flag = 1;
for ee = 1:epochs
    NMSE_trans_time_temp = [];
    NMSE_lstm_time_temp = [];
    for bb = 1:batches
        %% 分batch
        if batch_flag+batch_size-1>=Num_timestep
            Adwcpd_batch = Adwcpd_timestep(batch_flag:end);
            Bdwcpd_batch = Bdwcpd_timestep(batch_flag:end);
            Xtrue_batch = Xtrue_timestep(batch_flag:end,:,:);          
        else
            Adwcpd_batch = Adwcpd_timestep(batch_flag:batch_flag+batch_size-1);
            Bdwcpd_batch = Bdwcpd_timestep(batch_flag:batch_flag+batch_size-1);
            Xtrue_batch = Xtrue_timestep(batch_flag:batch_flag+batch_size-1,:,:);
        end
        NMSE_trans_tt_temp = [];
        NMSE_lstm_tt_temp = [];
        for tt = 1:length(Adwcpd_batch)
            C_lstm_tt = squeeze(C_lstm(ee,bb,tt,end,:));
            C_trans_tt = squeeze(C_trans(ee,bb,tt,end,:));
            Xtrue_tt = squeeze(Xtrue_batch(tt,:,:));
            Xhat_lstm_tt = Adwcpd_batch{tt}*diag(C_lstm_tt)*Bdwcpd_batch{tt}';
            Xhat_trans_tt = Adwcpd_batch{tt}*diag(C_trans_tt)*Bdwcpd_batch{tt}';
            
            %% 计算误差
            NMSE_trans_tt_temp(tt) = frob(Xtrue_tt-Xhat_trans_tt)^2/ frob(Xtrue_tt)^2;
            NMSE_lstm_tt_temp(tt) = frob(Xtrue_tt-Xhat_lstm_tt)^2/ frob(Xtrue_tt)^2;

        end
        NMSE_trans_tt{ee,bb} = NMSE_trans_tt_temp;
        NMSE_lstm_tt{ee,bb} = NMSE_lstm_tt_temp;
        batch_flag = batch_flag+batch_size;

        NMSE_lstm_time_temp = [NMSE_lstm_time_temp,NMSE_lstm_tt_temp];
        NMSE_trans_time_temp = [NMSE_trans_time_temp,NMSE_trans_tt_temp];
    end
    
%     NMSE_temp= squeeze(NMSE_lstm_tt(ee,:,:))';
    NMSE_lstm_time(ee,:) = NMSE_lstm_time_temp;
%     NMSE_temp= squeeze(NMSE_trans_tt(ee,:,:))';
    NMSE_trans_time(ee,:) = NMSE_trans_time_temp;

    batch_flag = 1;
end
NMSE_lstm_bb = cellfun(@mean,NMSE_lstm_tt);
NMSE_trans_bb = cellfun(@mean,NMSE_trans_tt);

%% 结果保存 （改文件路径和文件名）
NMSE_time{1} = NMSE_trans_time;
NMSE_time{2} = NMSE_lstm_time;

%save_file_path = 'E:\hqy\Sigma_icpd_retry\NMSE\';
save_file_path = 'F:\黄清扬\Sigma_cpd\NMSE\';
save_file_name = ['NMSE_',method_char,'_sigma',num2str(sigma),'_epoch.mat'];
save([save_file_path,save_file_name],"NMSE_time");


