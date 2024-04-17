%% 比较不同辐射源在不同信噪比下的误差曲线
clear; clc;
close all;

%% 画误差与训练轮数的关系，一个辐射源一个图
snr = [20];
RR = [2];
sigma = 3;
v = 0.01;
%rho = 0.1:0.02:0.2;
rho = [0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2];
rho_set = rho;
file_pre = 'E:\黄清扬\SpectrumPrediction_2024\MATLABoutput';
for rr = 1:length(RR)
    figure(rr)
    for ss = 1:length(rho)
        file_path_nmse_ex = [file_pre,'\Mat_R',num2str(RR(rr)),'_v',num2str(v),'_SNR',num2str(snr),'_5'];%,'_未归一化'];
        file_path_nmse = [file_path_nmse_ex,'\Mat_rho',num2str(rho(ss)),'_R',num2str(RR(rr)),'_SNR',num2str(snr),'_sinc2C\result\'];%,'_sinc2C\result\'];
        file_name = ['NMSE_NMFCPD_rho',num2str(rho(ss)),'_R',num2str(RR(rr)),'_timestep=20.mat'];
        NMSE = load([file_path_nmse,file_name]);
        NMSE_rr{rr,ss} = NMSE.NMSE_time{1}(:,end);
        %% 统计结果
        NMSE_rr_mean{rr,ss} = mean(NMSE_rr{rr,ss});
        NMSE_rr_med{rr,ss} = median(NMSE_rr{rr,ss});
        NMSE_rr_TC{rr,ss} = mean(NMSE.NMSE_time{2},1);
        NMSE_rr_C{rr,ss} = mean(mean(NMSE.NMSE_time{3},1));
        %% 画图
        figure(rr)
        for ii = 1:size(NMSE_rr{rr,ss},2)
            plot(NMSE_rr{rr,ss}(:,ii),'linewidth',1.5,'DisplayName', ...
                ['NMFCPD-LSTM,$\rho$=',num2str(rho(ss)),',epochs=',num2str(10)]);
            hold on;
        end

%         figure(2);
%         for ii = 1:size(NMSE_rr{rr,ss},2)
%             subplot(6,1,ss)
%             plot(NMSE_rr{rr,ss}(:,ii),'linewidth',1.5,'DisplayName', ...
%                 ['NMFCPD-LSTM,$\rho$=',num2str(rho(ss)),',epochs=',num2str(10)]);
%             hold on;
%             xlabel('T');
%             ylabel('NMSE');
%             set(gca,'FontName','Times new roman');
%             grid on;
%             title(['$\rho$=',num2str(rho(ss)),',epochs=',num2str(10)],'Interpreter','latex')
%             set(gca,'GridLineStyle',':');
%             %legend('Interpreter','latex');
%             xlim([1,size(NMSE_rr{rr,ss},1)]);
%             set(gca,'Yscale','log');
%             ylim([5e-2,1])
%         end    
    end
    
    figure(rr)
    title(['R=',num2str(RR(rr))]);
    xlabel('T');
    ylabel('NMSE');
    set(gca,'FontName','Times new roman');
    grid on;
    set(gca,'GridLineStyle',':');
    legend('Interpreter','latex');
    xlim([1,size(NMSE_rr{rr,ss},1)]);
    set(gca,'Yscale','log');
end

%% 画在不同辐射源情况下，最后一轮的预测误差，画在一张图
sigma = 2;
% for rr = 1:length(RR)
%     for ss = 1:length(snr)
%         file_path_nmse = ['E:\hqy\Mat_R',num2str(RR(rr)),'_SNR',num2str(snr(ss)),'_sinc2C\result\'];
%         file_name = ['NMSE_NMFCPD_sigma',num2str(sigma),'_R',num2str(RR(rr)),'.mat'];
%         NMSE = load([file_path_nmse,file_name]);
%         NMSE_rr{rr} = NMSE.NMSE_time{1};
%         
%     
%     end
% 
% end

%% 画R=4，采样率0.1随时间变化曲线
figure();
color_Str = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE','#A2142F'};
rr = 2;
rho = 0.1;
plot(NMSE_rr{rr,1}(:,end),'linewidth',1.5,'DisplayName', ...
    ['NMFCPD-LSTM,','R=',num2str(RR(rr)),',rho=',num2str(rho),',$\sigma$=',num2str(sigma),'dB',',SNR=',num2str(snr(1)),'dB'], ...
    'Color',color_Str{rr},'LineStyle','-');
hold on;   
xlabel('T');
ylabel('NMSE');
set(gca,'FontName','Times new roman');
grid on;
set(gca,'GridLineStyle',':');
legend('Interpreter','latex');%,'NumColumns',length(RR));
xlim([1,size(NMSE_rr{rr},1)]);
set(gca,'Yscale','log');


%% 画采样率为0.1，辐射源数目2/4/6/8的误差时间平均
figure();
rho = 0.1;
NMSE_rr_mean_plot= cell2mat(NMSE_rr_mean(:,1));
plot(NMSE_rr_mean_plot,'-o','LineWidth',1.5);
legend(['M-NMSE,$\rho=$',num2str(rho)],'Interpreter','latex');
xlabel('R');
ylabel('M-NMSE');
%xticks([2 4 6 8])
xlim([1 length(RR)]);
xticks(1:1:length(RR));
xtick_R = num2str(RR');
xticklabels(xtick_R);
set(gca,'FontName','Times new roman');
grid on;
set(gca,'GridLineStyle',':');
legend('Interpreter','latex');%,'NumColumns',length(RR));
%xlim([1,size(NMSE_rr{rr},1)]);
set(gca,'Yscale','log');
ylim([1e-2,1]);

%% 画R=2,不同采样率的M-NMSE变化曲线
NMSE_2_mean_plot = cell2mat(NMSE_rr_mean(1,:));
figure();
rr=1;
plot(NMSE_2_mean_plot,'-o','LineWidth',1.5);
legend(['M-NMSE,$R=$',num2str(RR(rr))],'Interpreter','latex');
xlabel('R');
ylabel('M-NMSE');
%xticks([2 4 6 8])
xlim([1 length(rho_set)]);
xticks(1:1:length(rho_set));
xtick_R = num2str(rho_set');
xticklabels(xtick_R);
set(gca,'FontName','Times new roman');
grid on;
set(gca,'GridLineStyle',':');
legend('Interpreter','latex');%,'NumColumns',length(RR));
%xlim([1,size(NMSE_rr{rr},1)]);
set(gca,'Yscale','log');
ylim([1e-2,1]);