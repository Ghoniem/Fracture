function Model=Set_F_s(Model,F_s)

for i=1:1:length(Model.Constraint)                        
Model.Constraint{i,1}.F=F_s(i);
end

end

