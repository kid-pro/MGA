%% This procedure is for constructing a delayed computation based on its own data, using a frequency domain downward delaying method to construct a dataset for deep learning.
clc
clear; 
close all


load("FZ_magdata1260.mat")% simulation data

upT=delta_all;  
load("FZ_magdata900.mat")% simulation data
down_data=delta_all;
%3x resolution improvement using sufer26's kriging algorithm for upT
% Super-resolution reconstruction of data, increased resolution spacing, tripled spacing for simulated data
upT;
%% Multiple Bayesian threshold denoising
CNNmatrix =upT;
I=CNNmatrix;
[thr,sorh,keepapp] = ddencmp('den','wv',I);
% thr = 20;
xd = wdencmp('gbl',I,'db35',5,thr,sorh,keepapp);
upT=xd;
%% Data set construction

F_data1.mag=upT;
% F_data1.mag=F_data1.mag*1000;  
[row_a,col_a]=size(F_data1.mag);
window_wide=128;  % determines the size of the sub-area  
row_c=row_a-(window_wide-1);
col_c=col_a-(window_wide-1);
n=0;
k=0;
z=0;
max_data=max(F_data1.mag(:));   
min_data=min(F_data1.mag(:));    
max_data1=max(down_data(:));   
min_data1=min(down_data(:));
% Scales the matrix elements to the [0, 1] range     
F_data1.mag = (F_data1.mag - min_data )   /  ( max_data - min_data); % normalisation
% Reduce the error by expanding the value of the matrix by a factor of 1000  
% F_data1.mag=F_data1.mag*1000;  
chains1=zeros(1,window_wide,window_wide);
chains2=zeros(1,window_wide,window_wide);           
chains3=zeros(1,window_wide,window_wide);
for i=1:5:row_c   %Per cent moving 10 points at a time
    for j=1:5:col_c
        slide_window=F_data1.mag(i:window_wide-1+i,j:window_wide-1+j);
% Storage of extracted subzones

slide_window_up1km=MagneticFrequencyExtension( slide_window);
slide_window_down1km=MagDataDown(slide_window_up1km,0.001,10);% parameter calculated using L-curve
z=z+1;
chains1(z,:,:)=slide_window_down1km;  
chains2(z,:,:)=slide_window_up1km;    
chains3(z,:,:)=slide_window;
    end
end

save('sc1down_data.mat', 'chains1');

save('sc1up_data.mat', 'chains2');

save('sc1Turedown_data.mat', 'chains3');









