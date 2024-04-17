clear;
clc;
addpath(genpath('tensorlab_2016-03-28'));
close all;
%% 文件索引
sigma = 2;%dB
RR = 2;
snr = 20;
method_char = 'IDW';  %DWCPD/ICPD  方法选择
test_char = 'sigma';    %sigma/trace/其他  ;对应实验参数设置是不同的sigma还是不同的轨迹trace
Sigma_index = sigma;   

trace = 2;
trace_index = trace;

% 预测数据文件索引
time_slot = [914:923];
loc_ii = [1000,2000,2601];

%%
time_step = 10;
input_window = 20;
dw_active = time_step+input_window+15;

%% 导入X-ground truth   （路径需改）
%file_path_cpd = 'E:\hqy\Sigma_icpd_retry\MATLAB\';
file_path_cpd = ['E:\hqy\Mat_R',num2str(RR),'_SNR',num2str(snr),'_sinc2C\'];

%% Xtrue

load([file_path_cpd,'Xtrue_sigma',num2str(sigma),'_R',num2str(RR),'.mat']);
Xtrue = XtrueAll;

%% Xidw
file_path = [file_path_cpd,'NEToutput\'];
load([file_path_cpd,'IDW_sigma',num2str(sigma),'_R',num2str(RR),'.mat']);
% file_path = [file_path_cpd,'NEToutput\'];
% load([file_path,'IDW_convLSTM_sigma',num2str(sigma),'_offline.mat']);
[L,K,T] = size(Xhat_idw);

%% Xnet
Xnet_hat = zeros(length(time_slot),2601,2,64);

Xhat_temp_tt = [];
for tt = 1:length(time_slot)
    Xhat_temp_ii = [];
    for ii = 1:length(loc_ii)
        file_nameX = ['IDW_SAEL_sigma',num2str(sigma),'_tt',num2str(time_slot(tt)),'_ii',num2str(loc_ii(ii)),'.mat'];
        [file_path,file_nameX]        
        load([file_path,file_nameX]);
        Xhat_temp_ii = cat(1,Xhat_temp_ii,X_pred);
    end
    Xhat_temp_tt(tt,:,:,:) = Xhat_temp_ii;
end

timeslot_label_index = time_slot+2;
X_label = Xtrue(:,:,timeslot_label_index);
Xidw_label = Xhat_idw(:,:,timeslot_label_index);
Xidw_label(Xidw_label<1e-15)=1e-15;
X_label(X_label<1e-15)=1e-15;
%% 计算误差
epochs = size(Xhat_temp_tt,3);
Time_testlength = size(Xhat_temp_tt,1);
for ee = 1:epochs
    %Xhat_temp_loc = permute(squeeze(Xhat_temp_tt(:,:,ee,:)),[2 3 1]);
    %NMSE_loc_SAEL(ee,:,:,:) = (Xhat_temp_loc - Xidw_label).^2./(Xidw_label).^2;
    for tt = 1:Time_testlength
        Xnet_sael_tt = squeeze(Xhat_temp_tt(tt,:,ee,:));
        Xtrue_tt = squeeze(X_label(:,:,tt));
        Xidw_tt = squeeze(Xidw_label(:,:,tt));
        NMSE_loc_SAEL_sys{tt,ee} = (Xtrue_tt - Xnet_sael_tt).^2./(Xtrue_tt).^2;
        NMSE_loc_SAEL_idw{tt,ee} = (Xidw_tt - Xnet_sael_tt).^2./(Xidw_tt).^2;
        NMSE_loc_idw_true{tt,ee} = (Xidw_tt - Xtrue_tt).^2./(Xtrue_tt).^2;
        NMSE_NET_SAEL(tt,ee) = frob(Xtrue_tt - Xnet_sael_tt )^2/frob(Xtrue_tt)^2;
        NMSE_idw_SAEL(tt,ee) = frob(Xidw_tt - Xnet_sael_tt )^2/frob(Xidw_tt)^2;
        NMSE_idw_true(tt,ee) = frob(Xidw_tt - Xtrue_tt )^2/frob(Xtrue_tt)^2;
    end
end


%% 画图
figure(2);
for i = 1:epochs
    plot(NMSE_NET_SAEL(:,i),'linewidth',1.5,'DisplayName',['IDW-SAEL-$\sigma$=',num2str(sigma),'dB,epoch=',num2str(i*10)]);
    hold on;
    %plot(NMSE_NET_TPS(:,i),'linewidth',1.5,'DisplayName',['TPS-$\sigma$=',num2str(sigma),'dB,epoch=',num2str(10+i*5)]);
end
xlabel('T');
ylabel('NMSE');
set(gca,'FontName','Times new roman');
grid on;
set(gca,'GridLineStyle',':');
legend('Interpreter','latex');
xlim([1,Time_testlength]);
set(gca,'Yscale','log');
%ylim([1e-2,1]);

%% 统计结果
NMSE_NET_mean_SAEL = mean(NMSE_NET_SAEL)
%NMSE_NET_mean_TPS = mean(NMSE_NET_TPS)

NMSE_NET_med_SAEL = median(NMSE_NET_SAEL)
%NMSE_NET_med_TPS = median(NMSE_NET_TPS)
