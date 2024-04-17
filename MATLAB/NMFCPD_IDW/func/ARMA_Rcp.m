function [ct_next] = ARMA_Rcp(ct_w,Rcp)
%% t+1时刻 
% diff_count = 0;
% ct_test = ct;
% % 平稳性检测（若不平稳，差分处理直到平稳）
% t_window = dW;
% % 模型定阶
% %ct_w = C_save(max(dW,t_window-dW):tt,:);
% ct_w = C_save(dW:tt,:);
t_window_true = size(ct_w,1);
if t_window_true >10
    %predict_active = 1;
    for arma_rr = 1:Rcp
        %t_window_true = size(ct_w,1);
        ct_w_r = ct_w(:,arma_rr);
        t_window_true = length(ct_w_r);
%                     ad1 = adftest(ct_w_r);
%                     ad2 = kpsstest(ct_w_r);
        ad1 = 1;
        ad2 = 0;
        if (ad1 == 1 && ad2 ==0)
            type = 3;
        else
            type = 4;
        end
        switch type 
            case 1
                ct_arma_rr = 0;
            case 2
                ct_arma_rr = 0;
            case 3
                ct_arma_rr = function_arma(ct_w_r',1);
            case 4
                ct_arma_rr = function_arima(ct_w_r',1);
        end
        ct_arma(arma_rr,1) = ct_arma_rr(end);
    end

ct_next = ct_arma;
end