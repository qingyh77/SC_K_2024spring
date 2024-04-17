function P = IDW(x,y,Xvec,Yvec,obs,theta)
%%   反距离插值,on-grid采样
%   输入:观测横坐标x,纵坐标y
%        待估横坐标X,纵坐标Y
%        观测数据obs
%        路径损耗指数theta

%   输出: 插值结果向量
%%

M = length(obs);
Num = length(Xvec);
dist = sqrt(  (x-Xvec).^2 + (y-Yvec).^2  );
[loc_idx,Sample_idx] = find(dist==0);
idx = find(dist==0);
loc_notongrid = setdiff(1:Num,loc_idx)';
weight_notongrid_1 = dist(loc_notongrid,:).^(-theta);
weight_notongrid_2 = diag(sum(weight_notongrid_1,2).^-1);
weight_notongrid = weight_notongrid_2*weight_notongrid_1;
weight = zeros(Num,M);
weight(loc_notongrid,:) = weight_notongrid;
weight(idx) = 1;


P = (weight)*obs;



end