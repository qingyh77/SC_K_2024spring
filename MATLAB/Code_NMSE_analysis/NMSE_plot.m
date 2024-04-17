
%% 
% 误差曲线绘制，旧版本，可以不用。用NMSE_net画图就行

clc;
%%
epoch_show = 10;


%% 画某一轮的结果
% 按batch
figure();
plot(NMSE_lstm_bb(epoch_show,:),'b-','linewidth',1.5,'DisplayName',['LSTM-predict']);
hold on
plot(NMSE_trans_bb(epoch_show,:),'r-','linewidth',1.5,'DisplayName',['transformer-predict']);
xlabel('batch');
ylabel('NMSE');
xlim([0 length(NMSE_trans_bb)])
grid on;
set(gca,'GridLineStyle',':');
set(gca,'Yscale','log');
legend('NumColumns',2);
title(['与Ground-truth tensor的误差（每个batch的误差均值）',',第',num2str(epoch_show*5),'轮']);

% 按时间
figure();

plot(NMSE_trans_time(epoch_show,:),'r-','linewidth',1,'DisplayName',['transformer-predict']);
hold on
plot(NMSE_lstm_time(epoch_show,:),'b-','linewidth',1,'DisplayName',['LSTM-predict']);

xlabel('T');
ylabel('NMSE');
grid on;
set(gca,'GridLineStyle',':');
set(gca,'Yscale','log');
legend('NumColumns',2);
title(['与Ground-truth tensor的误差',',第',num2str(epoch_show*5),'轮']);
xlim([0,length(NMSE_trans_time)])