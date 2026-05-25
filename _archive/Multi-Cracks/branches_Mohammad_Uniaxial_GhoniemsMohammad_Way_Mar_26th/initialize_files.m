function [elastic_fields]=initialize_files(input1)
    
fid1=fopen(['./COMSOL_IO/ux',num2str(input1),'.txt'],'w+');
fid2=fopen(['./COMSOL_IO/uy',num2str(input1),'.txt'],'w+');
fid3=fopen(['./COMSOL_IO/sxx',num2str(input1),'.txt'],'w+');
fid4=fopen(['./COMSOL_IO/sxy',num2str(input1),'.txt'],'w+');
fid5=fopen(['./COMSOL_IO/syy',num2str(input1),'.txt'],'w+');


% fid6=fopen('./COMSOL_IO/boundary_loads.txt','w+');


elastic_fields=[fid1,fid2,fid3,fid4,fid5];

% fid7=fopen('./Comsol_IO_Files/FDsxx.txt','w+');
% fid8=fopen('./Comsol_IO_Files/FDsxy.txt','w+');
% fid9=fopen('./Comsol_IO_Files/FDsyy.txt','w+');

% FD_elastic_fields=[fid7,fid8,fid9];



