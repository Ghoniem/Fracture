function Model=Set_U_s(Model,U_s)

for i=1:1:length(Model.Constraint)                        
    Model.Constraint{i}.U=U_s(i);    
end

end
