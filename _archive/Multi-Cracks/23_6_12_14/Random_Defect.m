% Mohammad Alabdullah 
% KU

% clc; close all; clear all

% % Import COMSOL
% import com.comsol.model.*
% import com.comsol.model.util.*
% model = ModelUtil.create('Model');
% model.modelNode.create('mod');
function [Line_Pos] = Random_Defect(hight,width,num_cracks,minc,maxc)

ind = 1; Number_Of_Cracks = num_cracks;

% Positions
hx = 0; hy = 0; hr = 0;

% Geometry Dimensions
H = hight; W = hight; 

% Crack Geometry 
Crack_Min_W = minc; Crack_Max_W = maxc;
Theta_Min = 0; Theta_Max = 90;

% % Initiate COMSOL Geometry 
% model.component.create('comp1', true);  
% model.component('comp1').geom.create('geom1', 2);
% model.component("comp1").geom("geom1").lengthUnit("mm");
% model.component("comp1").geom("geom1").selection().create("csel1", "CumulativeSelection");

% Intitiate Cracks Population 
while ind <= Number_Of_Cracks 
hx = (2.0*rand-1.0)*W/2;
hy = (2.0*rand-1.0)*H/2;
hr = rand*(Crack_Max_W-Crack_Min_W)+Crack_Min_W;
Theta = rand*(Theta_Max - Theta_Min)+Theta_Min;

% Stay With In The Boundary Conditions 
if Theta <= 90
    if abs(hy) + (hr/2)*sind(Theta) > abs(H/2)*0.9 || abs(hx) + (hr/2)*cosd(Theta) > abs(W/2)*0.9
        
        continue 
    end
    
else
        
    if abs(hy)+(hr/2)*sind(180-Theta) > abs(H/2)*0.9 || abs(hx) + (hr/2)*cosd(180-Theta) > abs(W/2)*0.9
        
    continue 
    
    end
end

% Overlap Prevention Conditions

    if ind >1
    
       x = hx-Line_Pos(1,:);
       y = hy-Line_Pos(2,:);
       dis = sqrt(x.^2+y.^2);
       
       if dis > 1.5*(hr/2+Line_Pos(3,:)/2);
       else
           continue 
       end
       
    end
    
    Line_Pos(:,ind) = [hx;hy;hr;Theta];
   

% Plot Distribution    
    if Theta<=90
    plot([hx-(hr/2)*cosd(Theta),hx+(hr/2)*cosd(Theta)],[hy-(hr/2)*sind(Theta), hy + (hr/2)*sind(Theta)])
    hold on
    else
    plot([hx+(hr/2)*cosd(180-Theta),hx-(hr/2)*cosd(180-Theta)],[hy-(hr/2)*sind(180-Theta), hy + (hr/2)*sind(180-Theta)])
    hold on
    end

% % Populate Geometry With Cracks In COMSOL     
% model.component('comp1').geom('geom1').create(['r' num2str(ind)], 'Rectangle');
% model.component('comp1').geom('geom1').feature(['r' num2str(ind)]).set('size', {num2str(hr), '1'});
% model.component('comp1').geom('geom1').feature(['r' num2str(ind)]).set('base', 'center');
% model.component('comp1').geom('geom1').feature(['r' num2str(ind)]).set('pos', [hx hy]);
% model.component('comp1').geom('geom1').feature(['r' num2str(ind)]).set('rot', Theta);
% model.component("comp1").geom("geom1").feature("r"+ind).set("contributeto", "csel1");

 ind = ind+1;
end

xlim([-W/2 W/2])
ylim([-H/2 H/2])

% % Finalize The Geometry In COMSOL 
% model.component('comp1').geom('geom1').create(['r' num2str(ind+1)], 'Rectangle');
% model.component('comp1').geom('geom1').feature(['r' num2str(ind+1)]).set('size', {num2str(H), num2str(W)});
% model.component('comp1').geom('geom1').feature(['r' num2str(ind+1)]).set('base', 'center');
% model.component('comp1').geom('geom1').create('dif1', 'Difference');
% model.component("comp1").geom("geom1").feature("dif1").selection("input").set(['r' num2str(ind+1)] );
% model.component("comp1").geom("geom1").feature("dif1").selection("input2").named("csel1"); 
% model.component('comp1').geom('geom1').run();   

% % Save 
% model.save([pwd '/Random_Crack.mph']);
end


        
    

    


