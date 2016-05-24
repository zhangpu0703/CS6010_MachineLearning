function hw_7
clear
format short g
train=load ('ZipDigits.train');
test=load('ZipDigits.test');
digits=train(:,1);
grayscale=train(:,2:end);
digits_test=test(:,1);
grayscale_test=test(:,2:end);
[n,d]=size(grayscale);
n
w=floor(sqrt(d));
[ind2,ind3,sym,den]= gainian(digits,grayscale,w);
[ind2_t,ind3_t,sym_t,den_t]=gainian(digits_test,grayscale_test,w);
size(sym)
size(den)
w1=LR(sym,den,ind2,ind3);
[w_opt,E_in]=pocket(sym,den,w1,ind2,ind3);
E_in
x11=linspace(0,120,100);
x22=linspace(-1,1,100);
g1=@(x11)(-w1(1)/w1(3)-w1(2)*x11/w1(3));
g2=@(x11)(-w_opt(1)/w_opt(3)-w_opt(2)*x11/w_opt(3));
figure(1)
plot(sym(ind2),den(ind2),'ob',sym(ind3),den(ind3),'xr',x11,g1(x11),'g',x11,g2(x11),'y')
ylim([-1 1])
legend('Digit 1','Digit 5','Linear regression','Pocket Algorithm')
xlabel('Symmetry')
ylabel('Density')
title('Training Result')
X=ones(length(sym_t),1);
X(:,2)=sym_t;
X(:,3)=den_t;
Y(ind2_t)=1;
Y(ind3_t)=-1;
Y=Y';
h=sign(X*w_opt);
k=find(h~=Y);
E_out=length(k)/length(sym_t)
figure(2)
plot(sym_t(ind2_t),den_t(ind2_t),'ob',sym_t(ind3_t),den_t(ind3_t),'xr',x11,g1(x11),'g',x11,g2(x11),'y')
ylim([-1 1])
legend('Digit 1','Digit 5','Linear regression','Pocket Algorithm')
xlabel('Symmetry')
ylabel('Density')
title('Testing Result')
[w3,E_in3]=third(sym,den,ind2,ind3)

X3=X;
X3(:,4)=sym_t.^2;
X3(:,5)=den_t.^2;
X3(:,6)=sym_t.*den_t;
X3(:,7)=sym_t.^3;
X3(:,8)=den_t.^3;
X3(:,9)=sym_t.*den_t.^2;
X3(:,10)=sym_t.^2.*den_t;
h=sign(X3*w3);
k=find(h~=Y);
E_out_3=length(k)/length(sym_t)

syms x0 y0
g3=w3(1)+w3(2)*x0+w3(3)*y0+w3(4)*x0.^2+w3(5)*y0.^2+...
    w3(6)*x0.*y0+w3(7)*x0.^3+w3(8)*y0.^3+w3(9)*x0.*y0.^2+w3(10)*x0.^2.*y0;
figure(3)
plot(sym(ind2),den(ind2),'ob',sym(ind3),den(ind3),'xr')
ylim([-1 1])
legend('Digit 1','Digit 5')
hold on
ezplot(g3,[0 120 -1 1])
xlabel('Symmetry')
ylabel('Density')
title('Training Result with 3rd Order Polynomial')
hold off

figure(4)
plot(sym_t(ind2_t),den_t(ind2_t),'ob',sym_t(ind3_t),den_t(ind3_t),'xr')
ylim([-1 1])
legend('Digit 1','Digit 5')
hold on
ezplot(g3,[0 120 -1 1])
xlabel('Symmetry')
ylabel('Density')
title('Testing Result with 3rd Order Polynomial')
hold off
end



function [ind2,ind3,sym,den]=gainian(digits,grayscale,w)
ind1=find(digits==1|digits==5);
x=digits(ind1);
y=grayscale(ind1,:);
for i=1:length(x);
    z=reshape(y(i,:),w,w);
	z=z';
    sym(i)=0;
    den(i)=0;
    for i1=1:16
        for i2=1:16
            sym(i)=sym(i)+abs((z(i1,i2)-z(i1,17-i2)));
            den(i)=den(i)+z(i1,i2);
        end
    end
    sym(i)=sym(i)/2;
    den(i)=den(i)/256;
end
ind2=find(x==1);
ind3=find(x==5);
end

%% Linear regression
function w1=LR(sym,den,ind2,ind3)
sym=sym'; 
den=den';
N=length(sym);
X=ones(N,1);
X(:,2)=sym;
X(:,3)=den;
Y(ind2)=1;
Y(ind3)=-1;
Y=Y';
X_t=inv(X'*X)*X';
w1=X_t*Y;
end
%% Pocket Algorithm
function [w_opt,E_in]=pocket(sym,den,w1,ind2,ind3)
tmax=10000; 
N=length(sym);
X=ones(N,1);
X(:,2)=sym;
X(:,3)=den;
Y(ind2)=1;
Y(ind3)=-1;
Y=Y';
w=w1;
h=sign(X*w);
k=find(h~=Y);
E_in=length(k)/N;
w_opt=w1;
t=0;
for i=1:tmax
    h=sign(X*w);
    k=find(h~=Y);
    E_in1=length(k)/N;
    if E_in1==0 break
    else
    if E_in1<E_in
       w_opt=w;
       E_in=E_in1;
    end
    t=t+1;
    w=w+Y(k(1))*X(k(1),:)';
    end
end
end

function [w_opt3,E_in3]=third(sym,den,ind2,ind3)
N=length(sym);
X=ones(N,1);
X(:,2)=sym;
X(:,3)=den;
Y(ind2)=1;
Y(ind3)=-1;
Y=Y';
X3=X;
X3(:,4)=sym.^2;
X3(:,5)=den.^2;
X3(:,6)=sym.*den;
X3(:,7)=sym.^3;
X3(:,8)=den.^3;
X3(:,9)=sym.*den.^2;
X3(:,10)=sym.^2.*den;
X3_t=inv(X3'*X3)*X3';
w31=X3_t*Y;

%% Pocket Algorithm
tmax=10000; %set maximum iteration
w3=w31;h=sign(X3*w3);
k=find(h~=Y);
E_in3=length(k)/N;
w_opt3=w31;
t3=0;
for i=1:tmax
    h=sign(X3*w3);
    k=find(h~=Y);
    E_in2=length(k)/N;
    if E_in2==0 break
    else
    if E_in2<E_in3
       w_opt3=w3;
       E_in3=E_in2;
    end
    t3=t3+1;
    w3=w3+Y(k(1))*X3(k(1),:)';
    end
end
end

