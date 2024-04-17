%% ARIMA
%% 非平稳序列
function [preData,LLL,UUU] = function_arima(a4,ntest)
    % 差分
    count = 0;
    arr1 = a4; 
    while true
        arr21 = diff(arr1);
        count = count + 1;
        ad1 = adftest(arr21); % 1平稳 0非平稳
        ad2 = kpsstest(arr21); % 0 平稳  1 非平稳
        if ad1==1 && ad2 == 0
            break
        end
        arr1 = arr21;
    end
    
    arr22 = arr21';
    % 自相关
    figure(2)
    subplot(2,1,1)
    coll1 = autocorr(arr22);   
    stem(coll1)%绘制经线图
    title('自相关')

    subplot(2,1,2)
    coll2 = parcorr(arr22);   
    stem(coll2)%绘制经线图
    title('偏相关')
    
    % 此时是平稳的
    count2 = length(arr22);
    limvalue = round(count2 / 10);
    if(limvalue > 3)
        limvalue = 3;
    end
    % 计算最佳的p 和 q
    [bestp,bestq] = Select_Order_arima(arr22, limvalue, limvalue, count);
    
    modelo = arima(bestp,count,bestq);
    md1 = estimate(modelo, arr22);
    [Y, YMSE] = forecast(md1, ntest, 'Y0', arr22);
    
    lower = Y - 1.96*sqrt(YMSE); %95置信区间下限
    upper = Y + 1.96*sqrt(YMSE); 
    
    difvalue = [arr22;Y]; %差分序列
    difvalueL = [arr22;lower];
    difvalueU = [arr22;upper];
    count2 = length(a4);
    if(count >=1)
        hhh = cumsum([a4(1);difvalue]);%还原差分值  
        LLL1 = cumsum([a4(count2);difvalueL]);
        UUU1 = cumsum([a4(count2);difvalueU]);
    end  
    preData = hhh(length(a4)+1 : (length(a4) + ntest));    
    LLL = LLL1(length(a4)+1 : (length(a4) + ntest));
    UUU = UUU1(length(a4)+1 : (length(a4) + ntest));
end 
