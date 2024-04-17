function [A,B,c,Sit,Rit,Pjt,Qjt] = ICPD_ARMA(Yt,Wt,Rrank,A0,B0,C0,Si0,Ri0,Pj0,Qj0,c_bar,lambda,mju,beta)


if nargin<=12
    mju =1e-6;
    beta = 1e-4;
end

delta_1 = 1;
delta_2 = 1;
%% ICPD更新B和A

Y = Yt;
W = Wt;
Wvec = W(:);
Yvec = Y(:);
[I,J] = size(Y);
W_trans = Wt';

% 更新辅助矩阵
ct = C0;
A = A0;
B = B0;
iter = 0;
while (iter<=10)
    iter = iter+1;
    %% Update A
    for ii = 1:I
        sit = lambda * Si0(ii,:) + Yt(ii,:)*diag(Wt(ii,:))*B*diag(ct);
        rit = lambda * Ri0{ii} + diag(ct)* B' * diag(Wt(ii,:)) * B * diag(ct);
        Sit(ii,:) = sit;
        Rit{ii} = rit;
    
        ati = sit * pinv(rit + mju*eye(Rrank));
        A(ii,:) = ati;
    
    end
    
    %% Update B
    for jj = 1:J
        qjt = lambda*Qj0(jj,:)' + diag(ct)* A' *diag(W_trans(jj,:))*Yt(:,jj);
        pjt = lambda*Pj0{jj} + diag(ct)*A' *diag(W_trans(jj,:)) * A *diag(ct);
    
        Qjt(jj,:) = qjt';
        Pjt{jj} = pjt;
    
        btj = pinv(pjt+mju*eye(Rrank)) * qjt;
        B(jj,:) = btj';
    
    end

    %% Update C：ARMA+ICPD
    ckBA = kr(B,A);
    Sampleindex = find(Wvec);
    ckBAsparse = ckBA(Sampleindex,:);
    y_omega = Yvec(Sampleindex);
    
    
    % logg = Wvec==1;
    % cksparse = zeros(size(ckBA));
    % cksparse(:,logg) = ckBA(:,logg);

    
    ck1 = pinv(ckBAsparse'*ckBAsparse + (mju+beta)*eye(Rrank) )* (ckBAsparse'*y_omega + c_bar);
    ck2 = pinv(ckBAsparse'*ckBAsparse + (mju)*eye(Rrank) )* (ckBAsparse'*y_omega);
    Lc = 1/norm(ckBAsparse'*ckBAsparse);
    ck3 = ct - Lc*(  ckBAsparse'*ckBAsparse*ct - ckBAsparse'*y_omega + mju*ct +beta*ct -c_bar   );
    %c = ck1;
    
    delta_1 = delta_2;
    delta_2 = norm(ck3-ct,"fro");
    if iter>=2 && abs(delta_1 - delta_2)<=1e-2
        break;
    end
    ct = ck3;
end
c{1} = ck1;
c{2} = ck2;
c{3} = ck3;
