clear;
clc;
addpath(genpath('tensorlab_2016-03-28'));
close all;
%% 文件索引
sigma = 2;%dB
RR = 8;  %辐射源个数
v = 0.01; % 运动速度
snr = 20;
method_char = 'IDW';  %DWCPD/ICPD  方法选择
test_char = 'rho';    %sigma/trace/其他  ;对应实验参数设置是不同的sigma还是不同的轨迹trace
Sigma_index = sigma;   
rho = 0.13;


trace = 2;
trace_index = trace;

%%
time_step = 10;
input_window = 20;
dw_active = time_step+input_window+15;

%% 导入X-ground truth   （路径需改）
%file_path_cpd = 'E:\hqy\Sigma_icpd_retry\MATLAB\';
%file_path_cpd = ['E:\hqy\Mat_R',num2str(RR),'_SNR',num2str(snr),'_sinc2C\'];
%file_path_cpd = ['F:\黄清扬\SpectrumPrediction_2024\MATLABoutput\Mat_R',num2str(RR),'_v',num2str(v),'_SNR',num2str(snr),'_张量补全失败\Mat_rho',num2str(rho),'_R',num2str(RR),'_SNR',num2str(snr),'_sinc2C\'];
file_path_cpd = ['F:\黄清扬\SpectrumPrediction_2024\MATLABoutput\Mat_R',num2str(RR),'_v',num2str(v),'_SNR',num2str(snr),'\Mat_rho',num2str(rho),'_R',num2str(RR),'_SNR',num2str(snr),'_sinc2C\'];

%% Xtrue

load([file_path_cpd,'Xtrue_rho',num2str(rho),'_R',num2str(RR),'.mat']);
Xtrue = XtrueAll;

%% Xidw


% IDW数据补全结果
load([file_path_cpd,'IDW_rho',num2str(rho),'_R',num2str(RR),'.mat']);
% file_path = [file_path_cpd,'NEToutput\'];
% load([file_path,'IDW_convLSTM_sigma',num2str(sigma),'_offline.mat']);
[L,K,T] = size(Xhat_idw);

%% Xtps
% load([file_path,'TPS_sigma',num2str(sigma),'.mat']);

%% 计算张量补全误差并画图
%NMSE_TC_TPS = zeros(T,1);
NMSE_TC_IDW = zeros(T,1);

for tt = 1:T 
    Xhat_idw_tt = squeeze(Xhat_idw(:,:,tt));
    %Xhat_tps_tt = squeeze(Xhat_tps(:,:,tt));
    Xtrue_tt = squeeze(Xtrue(:,:,tt));
    NMSE_TC_IDW(tt) = frob(Xtrue_tt - Xhat_idw_tt )^2/frob(Xtrue_tt)^2;
    %NMSE_TC_TPS(tt) = frob(Xtrue_tt - Xhat_tps_tt )^2/frob(Xtrue_tt)^2;
end

figure(1);
plot(NMSE_TC_IDW(:,1),'linewidth',1.5,'DisplayName',['IDW-$\sigma$=',num2str(sigma),'dB']);
hold on;
%plot(NMSE_TC_TPS,'linewidth',1.5,'DisplayName',['TPS-$\sigma$=',num2str(sigma),'dB']);
xlabel('T');
ylabel('NMSE');
set(gca,'FontName','Times new roman');
grid on;
set(gca,'GridLineStyle',':');
legend('Interpreter','latex');
xlim([1,T]);
set(gca,'Yscale','log');
ylim([1e-2,1]);

%% 统计结果
NMSE_TC_mean_IDW = mean(NMSE_TC_IDW)
%NMSE_TC_mean_TPS = mean(NMSE_TC_TPS)

NMSE_TC_med_IDW = median(NMSE_TC_IDW);
%NMSE_TC_med_TPS = median(NMSE_TC_TPS)

%% 计算网络预测误差并画图Xidw_net
file_path = [file_path_cpd,'NEToutput\'];
Xnet_idw = load([file_path,'IDW_convLSTM_sigma',num2str(sigma),'_offline_epochs10.mat']);
Xnet_idw = Xnet_idw.X_pred;
[epochs,T_test,~,~,~] = size(Xnet_idw);
% TPS
% Xnet_tps = load([file_path,'TPS_convLSTM_sigma',num2str(sigma),'_offline.mat']);
% Xnet_tps = Xnet_tps.X_pred;
%
%test_size = (T-14)*0.1;
test_size = T_test;
%T_net = 300;
Xtrue_used = Xtrue(:,:,29:end);
Xtrue_test = Xtrue_used(:,:,end-test_size+1:end);

NMSE_NET_IDW = zeros(test_size,epochs);
%NMSE_NET_TPS = zeros(test_size,epochs);
for ee = 1:epochs
    for tt = 1:test_size
        Xnet_idw_tt = squeeze(Xnet_idw(ee,tt,:,:,:));
        %Xnet_tps_tt = squeeze(Xnet_tps(ee,tt,:,:,:));
        Xtrue_tt = squeeze(Xtrue_test(:,:,tt));
        NMSE_NET_IDW(tt,ee) = frob(Xtrue_tt - Xnet_idw_tt )^2/frob(Xtrue_tt)^2;
        %NMSE_NET_TPS(tt,ee) = frob(Xtrue_tt - Xnet_tps_tt )^2/frob(Xtrue_tt)^2;  
    end
end
figure(2);
for i = 1:epochs
    plot(NMSE_NET_IDW(:,i),'linewidth',1.5,'DisplayName',['IDW-$\sigma$=',num2str(sigma),'dB,epoch=',num2str(i*5)]);
    hold on;
    %plot(NMSE_NET_TPS(:,i),'linewidth',1.5,'DisplayName',['TPS-$\sigma$=',num2str(sigma),'dB,epoch=',num2str(10+i*5)]);
end
xlabel('T');
ylabel('NMSE');
set(gca,'FontName','Times new roman');
grid on;
set(gca,'GridLineStyle',':');
legend('Interpreter','latex');
xlim([1,test_size]);
set(gca,'Yscale','log');
%ylim([1e-2,1]);

%% 统计结果
NMSE_NET_mean_IDW = mean(NMSE_NET_IDW)
%NMSE_NET_mean_TPS = mean(NMSE_NET_TPS)

NMSE_NET_med_IDW = median(NMSE_NET_IDW);
%NMSE_NET_med_TPS = median(NMSE_NET_TPS)

%% 保存数据
NMSE_time{1} = NMSE_NET_IDW;
NMSE_time{2} = NMSE_TC_IDW;
save_file_path = [file_path_cpd,'result\'];
if ~exist(save_file_path,'dir')
    mkdir(save_file_path);
end
save_file_name = ['NMSE_',method_char,'_ConvLSTM_rho',num2str(rho),'_R',num2str(RR),'.mat'];
save([save_file_path,save_file_name],"NMSE_time");