%Input grid_data in xyz coordinates, and grid_row, grid_col are known beforehand to generate grid size, output latitude, longitude, and magnetic anomaly grid data.
function [grid_latitude,grid_longitude,magnetic_anomally] = data_import(grid_data,grid_row,grid_col)%
%DATA_IMPORT 
%   
[length,~]=size(grid_data);
%% Read latitude, longitude and magnetic anomalies for all data
magnetic_anomally=zeros(grid_row,grid_col);row=1;col=0;
grid_longitude=zeros(grid_row,grid_col);
grid_latitude=zeros(grid_row,grid_col);
for i=1:length
    if mod(col,grid_col)==0
        row=row+1;
        col=1;
        magnetic_anomally(row,col)=grid_data(i,3);
        grid_latitude(row,col)=grid_data(i,2);
        grid_longitude(row,col)=grid_data(i,1);
    else
        col=col+1;
        magnetic_anomally(row,col)=grid_data(i,3);
        grid_latitude(row,col)=grid_data(i,2);
        grid_longitude(row,col)=grid_data(i,1);
    end
end
magnetic_anomally(1,:)=[];
grid_longitude(1,:)=[];
grid_latitude(1,:)=[];
% magnetic_anomally(:,1)=[];
% grid_longitude(:,1)=[];
% grid_latitude(:,1)=[];
end

