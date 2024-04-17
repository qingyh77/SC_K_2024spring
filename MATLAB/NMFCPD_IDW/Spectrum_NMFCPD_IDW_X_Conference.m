clc;clear;close all;
%% 张量函数包
addpath(genpath('tensorlab_2016-03-28'))
addpath(genpath('func'))
%% 路径修改：数据保存文件夹的绝对路径、阴影衰落矩阵的绝对路径
% ctrl+f 搜索：创建文件夹、阴影衰落矩阵

%% 网格图
loss_f = @(x,d,alpha) min(1,(x/d).^(-alpha));
d0 = 2;
rng(2,'twister')
gridLen = 50;
gridResolution = 1;%
x_grid = [0:gridResolution:gridLen];
y_grid = [0:gridResolution:gridLen];
[Xmesh_grid, Ymesh_grid] = meshgrid(x_grid, y_grid);
Xgrid = Xmesh_grid + 1i*Ymesh_grid;
% X1 = transpose(X1);

% 网格大小
[I,J] = size(Xgrid);

% 辐射源个数
RR = 2;

% 信噪比
snr = 20;

% 张量补全误差开关
completion_switch = 1;
completion_active = 0; %如果completion_switch=1，会在后续代码使completion_active置1，观察张量补全误差
%% 轨迹生成
loss_theta = 2;
v1 = 0.01;   % 辐射源速度
v2 = 0.01;   % 辐射源速度
T = 1000;
% % dW_active = [10:10:T];
rho_min = 0.1; %最小采样率

dW_max =1* ceil(1.5*2/rho_min); %dW最大窗口
T = dW_max+T-1;   %T = 1000+dW最大窗口
%ICPD_active = [dW_max+1:T];    
 
%% 生成轨迹路线
for tt = 1:T
    %% R=2
%         [ xt,yt ] = RectCircle(v1*tt,20.5757,2.41242,0 );
%         %[ xt,yt ] = RectCircle(v1*tt,0.01,0.01 );
%         xi{1}(tt) = 10 + xt;
%         yi{1}(tt) = 40 + yt;
%         [ xt,yt ] = RectCircle(v2*tt,2.74527,12.1121,0 );
%         xi{2}(tt) = 40 + xt;
%         yi{2}(tt) = 20 + yt;

        [ xt,yt ] = RectCircle(v1*tt,20.5757,2.41242,0 );
        %[ xt,yt ] = RectCircle(v1*tt,0.01,0.01 );
        xi{1}(tt) = 10 + xt;
        yi{1}(tt) = 40 + yt;
        [ xt,yt ] = RectCircle(v2*tt,2.74527,12.1121,0 );
        xi{2}(tt) = 40 + xt;
        yi{2}(tt) = 20 + yt;
    %% R=5
%     %[ xt,yt ] = RectCircle(v1*tt,70.5757,70.41242 );
%     [ xt,yt ] = RectCircle(v1*tt,20.5757,10.41242,0);
%     %[ xt,yt ] = RectCircle(v1*tt,0.01,0.01 );
%     xi{1}(tt) = 2 +  xt;
%     yi{1}(tt) = 25 +  yt;
%     
%     [ xt,yt ] = RectCircle(v2*tt,12.74527,2.1121,1 );
%     xi{2}(tt) = 8 +  xt;
%     yi{2}(tt) = 7 +  yt;
% 
%         [ xt,yt ] = RectCircle(v2*tt,15.74527,15.1121,0 );
%     xi{3}(tt) = 40 +  xt;
%     yi{3}(tt) = 4 +  yt;
% 
%         [ xt,yt ] = RectCircle(v2*tt,10.74527,5.1121,0 );
%     xi{4}(tt) = 35 +  xt;
%     yi{4}(tt) = 40 +  yt;
% 
% 
%         [ xt,yt ] = RectCircle(v2*tt,6.74527,6.1121,1 );
%     xi{5}(tt) = 10 +  xt;
%     yi{5}(tt) = 45 +  yt;

    %% R=3
%     %[ xt,yt ] = RectCircle(v1*tt,70.5757,70.41242 );
%     [ xt,yt ] = RectCircle(v1*tt,20.5757,2.41242,0);
%     %[ xt,yt ] = RectCircle(v1*tt,0.01,0.01 );
%     xi{1}(tt) = 5 +  xt;
%     yi{1}(tt) = 40 +  yt;
%     
%     [ xt,yt ] = RectCircle(v2*tt,12.74527,2.1121,1 );
%     xi{2}(tt) = 10 +  xt;
%     yi{2}(tt) = 4 +  yt;
% 
%     [ xt,yt ] = RectCircle(v2*tt,2.74527,20.1121,1 );
%     xi{3}(tt) = 40 +  xt;
%     yi{3}(tt) = 20 +  yt;


    %% R=6/8
%     [ xt,yt ] = RectCircle(v1*tt,20.5757,2.41242,1 );
%     %[ xt,yt ] = RectCircle(v1*tt,0.01,0.01 );
%     xi{1}(tt) = 2 + xt;
%     yi{1}(tt) = 4 + yt;
% 
%     [ xt,yt ] = RectCircle(v2*tt,2.74527,12.1121,1 );
%     xi{2}(tt) = 40 + xt;
%     yi{2}(tt) = 20 + yt;
% 
%         [ xt,yt ] = RectCircle(v2*tt,6.74527,8.1121,0 );
%     xi{3}(tt) = 40 + xt;
%     yi{3}(tt) = 40 + yt;
% 
%         [ xt,yt ] = RectCircle(v2*tt,30.74527,12.1121,0 );
%     xi{4}(tt) = 28 + xt;
%     yi{4}(tt) = 9 + yt;
% 
%         [ xt,yt ] = RectCircle(v2*tt,21.74527,7.1121,1 );
%     xi{5}(tt) = 10 + xt;
%     yi{5}(tt) = 27 + yt;
% 
%         [ xt,yt ] = RectCircle(v2*tt,4.74527,4.1121,1 );
%     xi{6}(tt) = 4 + xt;
%     yi{6}(tt) = 45 + yt;
% 
%         [ xt,yt ] = RectCircle(v2*tt,8.74527,14.1121,1 );
%     xi{7}(tt) = 18 + xt;
%     yi{7}(tt) = 44 + yt;
% 
%         [ xt,yt ] = RectCircle(v2*tt,2.74527,3.1121,0 );
%     xi{8}(tt) = 40 + xt;
%     yi{8}(tt) = 2 + yt;

end

for rr=1:RR
    location_set{rr} = xi{rr} + 1i*yi{rr};
end

%% 轨迹图示意
figure(4);
for i = 1:RR
    plot(xi{i},yi{i},'LineWidth',1.5,'DisplayName',['source',num2str(i)]);
    hold on;
    scatter(xi{i}(1),yi{i}(1),40,'DisplayName',['trace',num2str(i),'-starting point'],'LineWidth',1.25);
    scatter(xi{i}(end),yi{i}(end),40,'DisplayName',['trace',num2str(i),'-end point'],'LineWidth',1.25);
    
end
grid on;
set(gca,'GridLineStyle',':');
axis([0 I 0 J]);
set(gca,'FontName','times new roman')
legend;


%% Ground-Truth PSD
K=64;
Ctrue = zeros(K,RR);
F_s = [0.8 1.2 1.5];
% for i = 1:RR
%     Ctrue(:,i) = F_s(i)*ones(K,1);
% end
% Ctrue(:,1) = [F_s(1)*ones(25,1); zeros(K-25,1)];
% Ctrue(:,2) = [zeros(K-25,1);F_s(2)*ones(25,1) ];


indK = [1:K]';
Sx =@(f0,a) sinc((indK-f0)/a).^2.*( abs((indK-f0)/a)<=1);  % basis sinc wave function
Ctrue = zeros(K,RR);
Ctrue(:,1)= 1.2*Sx(8,5) + 0.8*Sx(40,5);
Ctrue(:,2)=  0.9*Sx(7,5) + 1.1*Sx(25,4);

% Ctrue(:,1)= 1.2*Sx(8,5) + 0.8*Sx(40,5);
% Ctrue(:,2)=  0.9*Sx(7,5) + 1.1*Sx(25,4);
% Ctrue(:,3)=  0.9*Sx(56,5) + 1.5*Sx(14,4);


% Ctrue(:,1)= 1.2*Sx(8,4);
% Ctrue(:,2)= 1.4*Sx(25,3);
% Ctrue(:,3)= 1.2*Sx(15,3);
% Ctrue(:,4)= 0.8*Sx(37,3);
% Ctrue(:,5)= 1.2*Sx(56,6);

% Ctrue(:,1)= 1.2*Sx(5,4);
% Ctrue(:,2)= 1.4*Sx(25,3);
% Ctrue(:,3)= 1.2*Sx(18,3);
% Ctrue(:,4)= 0.8*Sx(37,3);
% Ctrue(:,5)= 0.6*Sx(56,4);
% Ctrue(:,6)= 0.7*Sx(45,4);
% Ctrue(:,7)= 1.1*Sx(11,3);
% Ctrue(:,8)= 1*Sx(30,6);

% Ctrue(:,1)= 0.7*Sx(12,4) + 1.2*Sx(30,3);
% Ctrue(:,2)= 1.4*Sx(9,3) + 1*Sx(58,5);
% Ctrue(:,3)= 0.8*Sx(19,3) + 1.3*Sx(43,4);
% Ctrue(:,4)= 0.7*Sx(16,3)+  1.1*Sx(30,4) + 1.3*Sx(55,3);
% Ctrue(:,5)= 0.6*Sx(50,3) + 1.2*Sx(20,3);
% Ctrue(:,6)= 1*Sx(26,4) + 0.9*Sx(46,4);

% 画图示意
figure(2);
for pp = 1:RR
    plot(Ctrue(:,pp)/max(Ctrue(:,pp)),'LineWidth',1.25,'DisplayName',['PSD-Ground-truth',num2str(pp)]);
    hold on;
end
grid on;
set(gca,'GridLineStyle',':');
xlabel('k');
ylabel('Cr');
legend;

%% Cest-plot-test
% figure(3);
% for pp = 1:RR
%     plot(Cest(:,pp)/max(Cest(:,pp)),'LineWidth',1.25,'DisplayName',['PSD-est',num2str(pp)]);
%     hold on;
% end
% grid on;
% set(gca,'GridLineStyle',':');
% xlabel('k');
% ylabel('Cr');
% legend;



%%
% % % for tt = 1:T
% % %  plot(xi{1}(tt),yi{1}(tt),'o')
% % %  hold on;
% % %   plot(xi{2}(tt),yi{2}(tt),'x')
% % % xlim([0 100]);
% % % ylim([0 100]) ;
% % %      pause(0.01);
% % % end
%% 初始化

% CP秩
Rcp = 5;


sigma_set = 2:2;

% 采样率集合
%rho_set = 0.1:0.02:0.2;
%rho_set = 0.1:0.02:0.18;
rho_set = 0.1:0.01:0.2;%0.2:-0.02:0.12;


outiter = length(rho_set) ;  %大循环次数
% 保存变量

% 张量补全误差
NMSE_IDW = zeros(T,outiter);
NMSE_TPS = zeros(T,outiter);
%NMSE_TC = zeros(T,outiter);
Ai_R_T = cell(RR,T);
Bi_R_T = cell(RR,T);
Ci_R_T = cell(1,RR);


for oo=1:outiter
    fprintf('Outiter=%d \n',oo)
    
    
    shadow_sigma = 3;
    rho = rho_set(oo);

    dW = 1* ceil(1.5*2/rho);
    ICPD_active = [dW+1:T];
    %shadow_sigma = sigma_set(oo);
    alpha = {2.0,2.0};
    Svec = [];
    for rr = 1:RR
        %shadow{rr} = Shadowing(Xgrid,shadow_sigma);
        %load('shadow.mat')
        alpha{rr} = 2;
        %shadow_linear{rr} = 10.^(shadow{rr}/10);      
    end
    load("shadow_linear_R2_51_3dB");  %阴影衰落矩阵
    lambda =0.9;  %ICPD的遗忘因子
    Rrank = Rcp;  
% %     RdWrank =30;
    A = randn(I,Rrank);
    B = randn(J,Rrank);
    % % A = Atrue;
    % % B = Btrue;
    C = zeros(T,Rrank);

    
    
    mu = 1e-6;
    %rho =0.5;
    
    %% 采样策略：0.05的固定位置采样和rho-0.05的随机位置采样
    samp = round(rho*I*J);
    M_fixed = round(0.05*I*J);
    M_rand = samp - M_fixed;
    
    SampleIndex_fixed = randperm(I*J,M_fixed);
    SampleRandSet = setdiff([1:I*J],SampleIndex_fixed);   

    


    Ang1{1} = 0*pi;Ang2{1} = 9/12*pi;
    Ang1{2} = -1*pi;Ang2{2} = 0;
    
    fprintf('iter-%d:    ',T)
    
    
    %% 迭代变量初始化
    Wall = zeros(I*J,K,T);
    Wall_S = zeros(I,J,T);
    XtrueAll = zeros(I*J,K,T);
    Yall = zeros(I*J,K,T);
    
    for rr = 1:RR
        Ai{rr} = randn(I,Rrank);
        Bi{rr} = randn(J,Rrank);
        Ci{rr} = randn(dW,Rrank);
    end
    Snmf_T = zeros(samp,RR,T);
    Snmf_cpd_ob_vecT = zeros(I*J,RR,T);
    Snmf_cpd_ob_T = zeros(I,J,T,RR);


    for tt = 1:T
        fprintf('\b\b\b\b')
        fprintf('%4d',tt)
        if tt==T
            fprintf('\n')
        end
        for rr=1:RR
            location =location_set{rr}(tt);
            loss_mat = abs(Xgrid - location);
            %Ang{rr} = AgleMat(Xgrid,location,Ang1{rr},Ang2{rr});
            Sc{rr} = loss_f(loss_mat,d0,alpha{rr}).*shadow_linear{rr};
            Sc{rr} = Sc{rr}/norm(Sc{rr},'fro'); %omin-directional
%             Sc{rr} = Sc{rr}.*Ang{rr}; %directional
            Svec = [Svec,Sc{rr}(:)];
        end
%         Xtrue = Sc{1} + Sc{2};
        Xtrue = zeros(I,J,K);
        %% 形成第t时刻的I*J*K的张量，并加噪声
        for rr= 1:RR
            Xtrue = Xtrue + outprod(Sc{rr},Ctrue(:,rr));
        end
        X0 = Xtrue;
        Pn = X0.^2*10^(-snr/10);
        if 10^(snr/10)>=1e3
            Pn =0;
        end
        Xtrue = Xtrue + sqrt(Pn).*randn(I,J,K);  %第t时刻，Ground-truth态势
        
        % 转成IJ*K
        Xtrue_mode3 = tens2mat(Xtrue,[],3);
        


        rng('shuffle');
        SampleIndex_rand = randperm(length(SampleRandSet),M_rand);
        % 最终采样位置索引
        index = [SampleIndex_fixed,SampleRandSet(SampleIndex_rand)];
        %index = randperm(I*J,samp);
        
        Wt = zeros(I*J,K);
        Wt_S = zeros(I,J);

        Wt(index,:)=1;
        Wt_S(index) = 1;

        Wall(:,:,tt) = Wt;
        Wall_S(:,:,tt) = Wt_S;

        Yt = Xtrue_mode3.*Wt;
        XtrueAll(:,:,tt) = Xtrue_mode3;
        Yall(:,:,tt) = Yt;
        %yt = Yt(:);
        %W = Wall(:,:,1:tt); % time t: ALL past indices
        %W_S = Wall_S(:,:,1:tt); 
        %Y = Yall(:,:,1:tt);% time t: all Observed data
        
        Xhat_idw_tt = zeros(I*J,K);
        Xhat_tps_tt = zeros(I*J,K);
        
    %% NMF-CPD方法
        Snmf_cpd_ob_IJ = zeros(I*J,RR);
        Xomega = Xtrue_mode3(index,:);
        
        %% 生成SLF
        if tt == 1   
            SelectInd = SPA(Xomega,RR);
            Sest = Xomega(:,SelectInd);
            Cest =(Sest\Xomega)';
            Cest(Cest<=1e-15) = 1e-15;
            %Cnmf(:,:,tt) = Cest;
            %Snmf_cpd_ob(:,:,tt) = Sest; %(M*R*T)
        else
            Sest = Xomega*pinv(Cest');
       end
        
        %% 将SLF整合成张量I*J*t
        for dd = 1:samp
            Snmf_cpd_ob_IJ(index(dd),:) = Sest(dd,:);
        end
        Snmf_cpd_ob_vecT(:,:,tt) = Snmf_cpd_ob_IJ;
        for rr_cpd = 1:RR
            Snmf_cpd_ob_T(:,:,tt,rr_cpd) = reshape(Snmf_cpd_ob_IJ(:,rr_cpd),I,J);
        end
        

        %% DW-CPD初启动
        if tt == dW
            dW =1* ceil(1.5*2/rho);
            if completion_switch ==1
                completion_active = 1;
            end
            for rr_cpd = 1:RR
                Y = squeeze(Snmf_cpd_ob_T(:,:,1:tt,rr_cpd));
                W_S = Wall_S(:,:,1:tt); 
                [Ai{rr_cpd}, Bi{rr_cpd}, Ci{rr_cpd}] = DWCPD_C0(Y,W_S,Rrank,dW,Ai{rr_cpd},Bi{rr_cpd},Ci{rr_cpd});
                %Sthat_DWCPD = Ai{rr_cpd}*diag(Ci{rr_cpd}(end,:))*Bi{rr_cpd}';
                %% 结果保存
                Ai_R_T{rr_cpd,tt} = Ai{rr_cpd};
                Bi_R_T{rr_cpd,tt} = Bi{rr_cpd};
                Ci_R_T{rr_cpd}(tt,:) = Ci{rr_cpd}(end,:);
                %NMSE_project{oo}(tt,rr_cpd) = frob(Sest_rr - Sthat_DWCPD )^2/frob(Sest_rr )^2;
                
                %% I-CPD初始化
                St = squeeze(Snmf_cpd_ob_T(:,:,tt,rr_cpd));
                for ii = 1:I
                    Sit{rr_cpd}(ii,:) = St(ii,:)*diag(Wt_S(ii,:))* Bi{rr_cpd} *diag(Ci{rr_cpd}(end,:));
                    Rit{rr_cpd}{ii} = diag(Ci{rr_cpd}(end,:))* Bi{rr_cpd}' *diag(Wt_S(ii,:)) * Bi{rr_cpd} *diag(Ci{rr_cpd}(end,:));
                end
    %             Wt_trans = Wt';
                for jj = 1:J
                    Qjt{rr_cpd}(jj,:) = ( diag(Ci{rr_cpd}(end,:))*Ai{rr_cpd}' *diag(Wt_S(:,jj))*St(:,jj) )';
                    Pjt{rr_cpd}{jj} = diag(Ci{rr_cpd}(end,:))* Ai{rr_cpd}' *diag(Wt_S(:,jj)) * Ai{rr_cpd}* diag(Ci{rr_cpd}(end,:));
                end
                ct{rr_cpd} = Ci{rr_cpd}(end,:);
            end

            
            

        end
        
        % 后续——ICPD
        if ismember(tt,ICPD_active)
               
            for rr_cpd = 1:RR
                %% 对每个辐射源做I-CPD
                Sest_rr_vec = squeeze(Snmf_cpd_ob_vecT(:,rr_cpd,tt));
                Sest_rr = reshape(Sest_rr_vec,I,J);
                [Ai{rr_cpd},Bi{rr_cpd},ct{rr_cpd},Sit{rr_cpd},Rit{rr_cpd},Pjt{rr_cpd},Qjt{rr_cpd}] = ICPD_C0(Sest_rr,Wt_S,Rrank,Ai{rr_cpd} ...
                                                                                                ,Bi{rr_cpd}, ...
                                                                                                ct{rr_cpd}, ...
                                                                                                Sit{rr_cpd}, ...
                                                                                                Rit{rr_cpd}, ...
                                                                                                Pjt{rr_cpd}, ...
                                                                                                Qjt{rr_cpd}, ...
                                                                                                lambda);
                %Sthat_ICPD = Ai{rr_cpd}*diag(ct{rr_cpd})*Bi{rr_cpd}';
                %% 结果保存
                Ai_R_T{rr_cpd,tt} = Ai{rr_cpd};
                Bi_R_T{rr_cpd,tt} = Bi{rr_cpd};
                Ci_R_T{rr_cpd}(tt,:) = ct{rr_cpd};
                %% S的补全误差
                %NMSE_project{oo}(tt,rr) = frob(Sest_rr - Sthat_ICPD )^2/frob(Sest_rr )^2;
            end
        end
        
        % 张量补全误差确认
        if completion_active == 1
                Xthat_TC = zeros(I,J,K);
                for test_rr = 1:RR
                    S_TCrr = Ai_R_T{test_rr,tt}* diag(Ci_R_T{test_rr}(tt,:)) * Bi_R_T{test_rr,tt}';
                    Xthat_TC = Xthat_TC + outprod( S_TCrr, Cest(:,test_rr) );
                end
                NMSE_TC(tt,oo) = frob(Xtrue - Xthat_TC )^2/frob(Xtrue)^2;
        end



    %% 插值方法补全态势
        for kk = 1:K
            X_IND = real(Xgrid(:));
            Y_IND = imag(Xgrid(:));
%             [x_IND,y_IND] = ind2sub([I J],index);
            x_IND = X_IND(index)';
            y_IND = Y_IND(index)';

            Yt_vec = Yt(:,kk);
            Yt_vec_ob = Yt_vec(index);

            % IDW
            xhat_idw_kk = IDW(x_IND,y_IND,X_IND,Y_IND,Yt_vec_ob,loss_theta);
            %Xhat_idw_kk = reshape(xhat_idw,I,J);
            Xhat_idw_tt(:,kk) = xhat_idw_kk;
            
            % TPS
%             xhat_tps_kk = TPS(x_IND,y_IND,Yt_vec_ob,X_IND,Y_IND,0,0);
%             Xhat_tps_tt(:,kk) = xhat_tps_kk;
        end
        %% 结果保存和记录
        Xhat_idw(:,:,tt) = Xhat_idw_tt;
        %Xhat_tps(:,:,tt) = Xhat_tps_tt;
        NMSE_IDW(tt,oo) = frob(Xtrue_mode3 - Xhat_idw_tt )^2/frob(Xtrue_mode3)^2;
        %NMSE_TPS(tt,oo) = frob(Xtrue_mode3 - Xhat_tps_tt ).^2/frob(Xtrue_mode3).^2;



    end
%     C_batch(oo,:,:) = C_save;
%     A_batch{oo} = A_save;
%     B_batch{oo} = B_save;
    %X_batch{oo} = XtrueAll;
    %shadow_save{oo} = shadow_linear;
    
    %% 保存数据
%     file_savepath = ['F:\黄清扬\SpectrumPrediction_2024\MATLABoutput\','Mat_R',num2str(RR),'_v',num2str(v1),'_SNR',num2str(snr),...
%         '\Mat_R',num2str(RR),'_rho',num2str(rho),'_SNR',num2str(snr),'_sinc2C'];
    
    % 创建文件夹
    file_savepath = ['F:\黄清扬\SpectrumPrediction_2024\MATLABoutput\','Mat_R',num2str(RR),'_v',num2str(v1),'_SNR',num2str(snr),...
        '\Mat_rho',num2str(rho),'_R',num2str(RR),'_SNR',num2str(snr),'_sinc2C'];
    if ~exist(file_savepath,'dir')
        mkdir(file_savepath);
    end

    if ~exist([file_savepath,'\NEToutput'],'dir')
        mkdir([file_savepath,'\NEToutput']);
    end

    % 正式保存文件
    save([file_savepath,'\IDW_rho',num2str(rho),'_R',num2str(RR),'.mat'],"Xhat_idw");
    %save([file_savepath,'\TPS_rho',num2str(rho),'_R',num2str(RR),'.mat'],"Xhat_tps");
    save([file_savepath,'\shadow_linear_rho',num2str(rho),'_R',num2str(RR),'_51.mat'],"shadow_linear");
    save([file_savepath,'\NMFCPD_PSD_rho',num2str(rho),'_R',num2str(RR),'.mat'],"Cest");
    save([file_savepath,'\NMFCPD_S_A_rho',num2str(rho),'_R',num2str(RR),'.mat'],"Ai_R_T");
    save([file_savepath,'\NMFCPD_S_B_rho',num2str(rho),'_R',num2str(RR),'.mat'],"Bi_R_T");
    save([file_savepath,'\NMFCPD_S_C_rho',num2str(rho),'_R',num2str(RR),'.mat'],"Ci_R_T");
    save([file_savepath,'\Xtrue_rho',num2str(rho),'_R',num2str(RR),'.mat'],"XtrueAll");
    save([file_savepath,'\Snmf_rho',num2str(rho),'_R',num2str(RR),'.mat'],"Snmf_cpd_ob_T");
    for rr_save = 1:RR
        C_icpd = Ci_R_T{rr_save}(dW:end,:);
        save([file_savepath,'\NMFCPD_S_C_rho',num2str(rho),'_R',num2str(RR),'_',num2str(rr_save),'.mat'],"C_icpd");
    end

end



%% 保存数据
% savefilepath = 'E:\hqy\Base_line\';
% sigma_test = 2;
% save([savefilepath,'ICPD_C_sigma',num2str(sigma_test),'.mat'],"C_batch");
% save([savefilepath,'ICPD_X_sigma',num2str(sigma_test),'.mat'],"X_batch");
% save([savefilepath,'ICPD_A_sigma',num2str(sigma_test),'.mat'],"A_batch");
% save([savefilepath,'ICPD_B_sigma',num2str(sigma_test),'.mat'],"B_batch");

%%
% 
% figure();
% plot(NMSE_DWCPD(dW:end,6),'LineWidth',1.25,'DisplayName',['NMSE']);
% hold on;
% grid on;
% set(gca,'GridLineStyle',':');
% xlabel('t');
% ylabel('NMSE');
% legend;
% set(gca,'YScale','log');