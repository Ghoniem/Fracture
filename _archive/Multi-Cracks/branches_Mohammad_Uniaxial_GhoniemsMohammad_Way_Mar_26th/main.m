close all; clc; clear all;
format short e

global sim_step 

%% Initial loading,Crack size, Geometry and Material Properties %%
control_data()

%% Initial crack location and orientation %%
get_cracks()

%% Initiate parameters %%
initial()
sim_step =1 ;
propagate=1;

%%  Establish initial distribution of dislocations %%
Dislocations_Distribution(0)
%% Propagate cracks by adding segment %%
GetSegments(propagate)

%% Stress Field %%
GetDislStressField()

%% FEM Correction (COMSOL)%%              
            import com.comsol.model.*
            import com.comsol.model.util.*
            model = ModelUtil.create('Model');
            model.modelNode.create('mod1');

run_Comsol(model,0)
model.save([pwd '/Uniaxial_0crack.mph']);

%% Acheive corrected distribution of dislocations %%
Dislocations_Distribution(1)
sim_step = sim_step+1;
GetDislStressField()
% GetCrackSegments(propagate)
