%% 平稳平稳 arma预测
%% a4训练数据 ntest预测长度
function data = function_arma(a4,ntest)
    %自相关、偏自相关
%     figure(2)
%     subplot(2,1,1)
%     coll1 = autocorr(a4);   
%     stem(coll1)%绘制经线图
%     title('自相关')
% 
%     subplot(2,1,2)
%     coll2 = parcorr(a4);   
%     stem(coll2)%绘制经线图
%     title('偏相关')
    
    %
    count2 = length(a4);
    limvalue = round(count2 / 10);
    if(limvalue > 10)
        limvalue = 10;
    end
    % 计算最佳的p 和 q
    [bestp,bestq] = Select_Order_arma(a4, limvalue, limvalue);
    
    % 得到最佳模型
    xa4 = iddata(a4');
    model = armax(xa4,[bestp,bestq]);
    
    % 开始预测,获取预测数据
    arr1 = [a4';zeros(ntest,1)];
    arr2 = iddata(arr1);
    arr3=predict(model, arr2, ntest); %利用arr2去向后预测
    arr4 = get(arr3);
    dataPre = arr4.OutputData{1,1}(length(a4)+1:length(a4)+ntest);
    data=[a4';dataPre]; % 获取训练结果加预测结果
end 