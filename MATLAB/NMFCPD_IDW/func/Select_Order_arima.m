%% Arima找最优阶数
function [best_p,best_q] = Select_Order_arima(data, pmax, qmax, d)
    min_aic = Inf;
    min_bic = Inf;
    best_p = 0;
    best_q = 0;
    for p=0:pmax
        for q=0:qmax
            model = arima(p,d,q);
            try
                [fit,~,logL]=estimate(model,data);
                [aic, bic] = aicbic(logL, p + q + 1, length(data));
            catch
                continue
            end
            
            if aic < min_aic
                min_aic = aic;
                min_bic = bic;
                best_p = p;
                best_q = q;
            end
            
        end
    end
end 
