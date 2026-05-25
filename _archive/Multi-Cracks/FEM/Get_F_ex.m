function F_ex=Get_F_ex(Force,NDOF)

F_ex=zeros(NDOF,1);                          
for i=1:1:length(Force)
    F_ex(Force{i}.DOF)=Force{i}.Magnitude;
end

end