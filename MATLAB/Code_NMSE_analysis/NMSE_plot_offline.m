%% 比较不同辐射源在不同信噪比下的误差曲线
clear; clc;
close all;
color_Str = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE','#A2142F'};
%% 画误差与训练轮数的关系，一个辐射源一个图
snr = [10 20 30];
RR = [2 3 5 8];
sigma = 2;
for rr = 1:length(RR)
    %figure(rr)
    for ss = 1:length(snr)
        file_path_nmse = ['E:\hqy\Mat_R',num2str(RR(rr)),'_SNR',num2str(snr(ss)),'_sinc2C\result\'];
        file_name = ['NMSE_IDW_sigma',num2str(sigma),'_R',num2str(RR(rr)),'.mat'];
        NMSE = load([file_path_nmse,file_name]);
        NMSE_rr{rr,ss} = NMSE.NMSE_time{1};
        %% 统计结果
        NMSE_rr_mean{rr,ss} = mean(NMSE_rr{rr,ss});
        NMSE_rr_med{rr,ss} = median(NMSE_rr{rr,ss});
        
        %% 画图
         figure();
        for ii = 1:size(NMSE_rr{rr,ss},2)
            plot(NMSE_rr{rr,ss}(:,ii),'linewidth',1.5,...
                'DisplayName' ,...
                ['NMFCPD-LSTM,$\sigma$=',num2str(sigma),'dB',',SNR=',num2str(snr(ss)),',epochs=',num2str(5*ii)]);
            hold on;
                xlabel('T');
            ylabel('NMSE');
            set(gca,'FontName','Times new roman');
            grid on;
            set(gca,'GridLineStyle',':');
            legend('Interpreter','latex');
            xlim([1,size(NMSE_rr{rr,ss},1)]);
            set(gca,'Yscale','log');
        end
    
    end
    %title(['R='])
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
figure();
for rr = 1:length(RR)
        plot(NMSE_rr{rr,1}(:,end),'linewidth',1.5,'DisplayName', ...
            ['NMFCPD-LSTM,','R=',num2str(RR(rr)),',$\sigma$=',num2str(sigma),'dB',',SNR=',num2str(snr(1)),'dB'], ...
            'Color',color_Str{rr},'LineStyle','-');
        hold on;
        plot(NMSE_rr{rr,2}(:,end),'linewidth',1.5,'DisplayName', ...
            ['NMFCPD-LSTM,','R=',num2str(RR(rr)),',$\sigma$=',num2str(sigma),'dB',',SNR=',num2str(snr(2)),'dB'], ...
            'Color',color_Str{rr},'LineStyle','--');
        hold on;
        plot(NMSE_rr{rr,3}(:,end),'linewidth',1.5,'DisplayName', ...
            ['NMFCPD-LSTM,','R=',num2str(RR(rr)),',$\sigma$=',num2str(sigma),'dB',',SNR=',num2str(snr(3)),'dB'], ...
            'Color',color_Str{rr},'LineStyle',':');
    
end
xlabel('T');
ylabel('NMSE');
set(gca,'FontName','Times new roman');
grid on;
set(gca,'GridLineStyle',':');
legend('Interpreter','latex');%,'NumColumns',length(RR));
xlim([1,size(NMSE_rr{rr},1)]);
set(gca,'Yscale','log');
