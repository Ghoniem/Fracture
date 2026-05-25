function createFieldFiles(X,Y,du,dv,Sxx,Sxy,Syy,fileID)

for i=1:size(X,1)
    for j=1:size(X,2)
        fprintf(fileID(1),'%e %e %e \n',X(i,j),Y(i,j),du(i,j));
        fprintf(fileID(2),'%e %e %e \n',X(i,j),Y(i,j),dv(i,j));
        fprintf(fileID(3),'%e %e %e \n',X(i,j),Y(i,j),Sxx(i,j));
        fprintf(fileID(4),'%e %e %e \n',X(i,j),Y(i,j),Sxy(i,j));
        fprintf(fileID(5),'%e %e %e \n',X(i,j),Y(i,j),Syy(i,j));
    end
end