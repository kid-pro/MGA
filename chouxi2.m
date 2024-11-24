function [data3] = chouxi2(data,n)
[~,col]=size(data);
data2=zeros(0);
j=1;
for i=1:col
if mod(i, n) == 0
  
    data2(:,j)=data(:,i-1);

    j=j+1;
end
end
j=1;
data3=zeros(0);
for i=1:col
if mod(i, n) == 0
  
    data3(j,:)=data2(i-1,:);

    j=j+1;
end
end
 data3=horzcat( data3(:,1),data3);
 data3=horzcat( data3,data3(:,end));
 data3=[ data3(1,:);data3];
 data3=[data3;data3(end,:)];
end
