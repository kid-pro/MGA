
function [T_down] = MagDataDown(upT,sgm,iteration)
% upT
xint=40/3;yint=40/3;% grid spacing, 0.0045° corresponds to 500 m.
h=560;% extension height of 1000m, a multiple of grid point spacing
[row1,col1]=size(upT);
upt=reshape(upT,[],1);
%%
[m,n]=size(upT);
mn(1)=2^nextpow2(m);
mn(2)=2^nextpow2(n);% Expand the rows and columns of the original data by one edge each
m1=floor((mn(1)-m)/2);
n1=floor((mn(2)-n)/2);
data=zeros(mn(1),mn(2));
data(1+m1:m+m1,1+n1:n+n1)=upT;

for i=m1:-1:1
    for j=n1+1:n+n1
%         data(i,j)=0;
         data(i,j)=data(i+1,j);%方法1
 %        data(i,j)=data(i+1,j)*(1-cos((j-n1)*pi/n))/2;%方法2
   
    end
end
for i=m1+1+m:mn(1)
    for j=n1+1:n+n1
%         data(i,j)=0;
        data(i,j)=data(i-1,j);
%         data(i,j)=data(i-1,j)*(1-cos((j-n1)*pi/n))/2;

    end
end
for i=n1:-1:1
    for j=1:mn(1)
%         data(j,i)=0;
         data(j,i)=data(j,i+1);
%          data(j,i)=data(j,i+1)*(1-cos(j*pi/mn(1)))/2;
    end
end
for i=n+1+n1:mn(2)
    for j=1:mn(1)
%         data(j,i)=0;
      data(j,i)=data(j,i-1);
%     data(j,i)=data(j,i-1)*(1-cos(j*pi/mn(1)))/2;
    end
end

[ln,col]=size(data);
%Perform a 2D fft transformation
fx=fft2(data);
fx=fftshift(fx);
%Calculate the corresponding angular frequencies u and v
wnx=2*pi/(xint*ln);
wny=2*pi/(yint*col);
cx=ln/2+1;cy=col/2+1;
%In the fourth step, calculate the delay factor Q, and the delay formula to get the delayed magnetic anomaly spectrum

s=1;
nn=iteration;

for i=1:ln
    freqx=(i-cx)*wnx;
    for j=1:col 
        freqy=(j-cy)*wny;
        freq=sqrt(freqx^2+freqy^2);%sqrt(u^2+v^2)
        R=exp(-freq*h);
                ff(i,j)=fx(i,j)* Tik_iteration;
    end
end
fx_down=fftshift(ff);
%In the fifth step, the Fourier inverse transform is used to reconstruct the delayed magnetic anomaly
T_down=ifft2(fx_down);
T_down=T_down(1+m1:m+m1,1+n1:n+n1);% Remove edge expansion information
T_down=reshape(T_down,row1,col1);


end

