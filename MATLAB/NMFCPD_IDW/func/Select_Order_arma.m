%% Arma找最优阶数
function [p1,p2] = Select_Order_arma(x2, limp, limq)
    array = zeros(limp, limq);
    x = iddata(x2');
    for i =1:limp
        for j = 1:limq 
            AmMode = armax(x,[i,j]);
            value = aic(AmMode);
            array(i,j) = value;
        end
    end
    
    [p1,p2]=find(array==min(min(array)));
end 