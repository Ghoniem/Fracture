clc; clear; close all;
 
%Input
%------------
L=2000;  %Length
H=200;   %Height
nx=80; 
ny=16; 

dx=L/nx;
dy=H/ny;

[xx, yy]=meshgrid([0:dx:L],[0:dy:H]);
xxx=xx(:);
yyy=yy(:);
Nodes=[xxx yyy];

Triangles=delaunay(Nodes(:,1),Nodes(:,2));
[Triangles, I1]=sort(Triangles,2);
[junk, I2]=sort(Triangles(:,1));
Triangles=Triangles(I2,:);
I1=I1(I2,:);

for i=1:1:size(Triangles,1)
Triangles(i,:)=Triangles(i,I1(i,:));    
end

t=50;                                               %Thickness of 2D FEM elements

Material{1}.Name='Concrete';
Material{1}.E=30000;                                %Modulus of elasticity of mortar
Material{1}.v=0.2;                                  %Poisson ratio of mortar
Material{1}.Type='Elastic Plane Stress';

%First Constraint
[DOFx1, DOFy1]=find_DOF_from_Nodes(Nodes,0,0);
Constraint{1,1}.DOF=DOFx1;     Constraint{1,1}.U=0;   %X
Constraint{2,1}.DOF=DOFy1;     Constraint{2,1}.U=0;   %Y

%Second Constraint
[DOFx2, DOFy2]=find_DOF_from_Nodes(Nodes,L,0);
Constraint{3,1}.DOF=DOFy2;     Constraint{3,1}.U=0;   %Y

%External Load
%----------------------
U_s=zeros(3,1);

[DOFxm, DOFym]=find_DOF_from_Nodes(Nodes,L/2,H);

Force{1}.Magnitude=-10000;
Force{1}.DOF=DOFym;

Loading_Steps{1}.Force=Force;
Loading_Steps{1}.U_s=U_s;

%Prepare Model variable for the solver
%--------------------------------------

for i=1:1:size(Nodes,1)
    
Model.Node{i,1}.x=Nodes(i,1);
Model.Node{i,1}.y=Nodes(i,2);
Model.Node{i,1}.DOFx=2*i-1;
Model.Node{i,1}.DOFy=2*i;

end

for i=1:1:size(Triangles,1)

Model.Element{i,1}.Nodes=Triangles(i,:);
Model.Element{i,1}.Type='cst';
Model.Element{i,1}.Material=1; 

end

Model.Material=Material;
Model.Thickness=t;
Model.NDOF=2*size(Model.Node,1);
Model.Constraint=Constraint;

Model.Analysis_Status.Analyzed=0;

%Solver
%--------------

Job=Run_Job_Linear(Model,Loading_Steps);

%Plot Results
%--------------
Show_Deformed=1;                                     %Show deformed shape
Mag_Factor=50;                                      %Magnification factor

Result_type1=0;                                      %No Contour  
Result_type2=6;                                      %Result type sigma x 
Result_type3=8;                                      %Result type sigma xy 

Plot_Results(Job{end}.Model,Show_Deformed,Mag_Factor,Result_type1);

Plot_Results(Job{end}.Model,Show_Deformed,Mag_Factor,Result_type2);
legend('Sigma xx');

Plot_Results(Job{end}.Model,Show_Deformed,Mag_Factor,Result_type3);
legend('Sigma xy');