function Model=Analysis_Linear(Model,Force,U_s)


if Model.Analysis_Status.Analyzed==0
Model=Model_Set_Initial_Conditions(Model);                %Initial Conditions
Modeli=Model;                                             %Intitial Model       
Model=Model_Update_Elements(Model,Modeli);      
else
Modeli=Model;                                             %Intitial Model       
end

[Support]=Get_Support(Model);                             %Get Constraint Vector
[U_s_i]=Get_U_s(Model);                                   %Get Initial Constraint Displacement
if isempty(U_s);  U_s=U_s_i; end                          %Use initial Constraint Dispalcement (as applied constraint disp) if it is empty
Model=Set_U_s(Model,U_s);                                 %Update Model Constraint Displacement
dU_s=U_s-U_s_i;                                           %Incremental Constraint Displacement


NDOF=Model.NDOF;
Free=[1:1:NDOF]'; Free(Support)=[];                       %Get Free DOF

Model.Force=Force;                                        %Set Model external Force
F_ex=Get_F_ex(Force,NDOF);                                %Get external force Vector


U=Get_U(Model);                                           %Get initial displacement from model 

%Start Iterations
%------------------------------------

[F_i]=Model_Get_F_i(Model);                                %Get Initial internal force from model 
dF=F_ex-F_i;                                               %Initial Residual Force

U_i=U;                                                     %Displacement from previous iteration
        
K=Global_Stiffness(Model);                                 %Get global stiffness
[K_ff, K_fs]=Stiffness_Submatrices(K,Support,Free);
dF_eff=dF(Free)-K_fs*dU_s;                                 %Effective Force Increment  
[dU_f]=Solve_Linear_System(K_ff,dF_eff);                   %Solve Linear System
dU(Free,1)=dU_f; dU(Support,1)=dU_s;                       %Displacement Incremenet

U=U_i+dU;                                                  %New displacement
Model=Set_U(Model,U);                                      %Update Model displacement
Model=Model_Update_Elements(Model,Modeli);                 %Update Stiffness after displacement update  $
[F_i]=Model_Get_F_i(Model);                                %Get internal force from model 

%End Iterations
%------------------------------------------------

F_s=Get_F_s(F_ex,F_i,Support);                            %Get final constraint force
Model=Set_F_s(Model,F_s);                                 %Update Model F_s

if Model.Analysis_Status.Analyzed==0
Model.Analysis_Status.Analyzed=1;
end

end