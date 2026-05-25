function run_Comsol(model,flag)

global  n_grid del_grid Length_unit sim_step E nu rho C n_c n_branch sig_e G xbound ybound 

ext_stress = sig_e*G;

% if strcmp(simulationtype,'Correction')==1
%     modnum=1;
% else
%     modnum=2;
% end

% ymin=0;ymax=n_grid*del_grid;
% xmin=0;xmax=n_grid*del_grid;
xmin = -(n_grid/2*del_grid)*Length_unit;
xmax =  (n_grid/2*del_grid)*Length_unit;
ymin = -(n_grid/2*del_grid)*Length_unit;
ymax =  (n_grid/2*del_grid)*Length_unit;

p=[0,ymin];

  


%------------------------------------------------
% elastic fields interpolation

    
model.func.create(['ux',num2str(sim_step)], 'Interpolation');
model.func(['ux' num2str(sim_step)]).model(['mod' num2str(sim_step)]);
model.func(['ux' num2str(sim_step)]).set('source', 'file');
model.func(['ux' num2str(sim_step)]).set('filename', [pwd '/COMSOL_IO/ux' num2str(sim_step) '.txt']);
model.func(['ux' num2str(sim_step)]).setIndex('fununit', 'm', 0);
model.func(['ux' num2str(sim_step)]).setIndex('argunit', 'm', 0);
model.func(['ux' num2str(sim_step)]).setIndex('argunit', 'm', 1);

model.func.create(['uy' num2str(sim_step)], 'Interpolation');
model.func(['uy' num2str(sim_step)]).model(['mod' num2str(sim_step)]);
model.func(['uy' num2str(sim_step)]).set('source', 'file');
model.func(['uy' num2str(sim_step)]).set('filename', [pwd '/COMSOL_IO/uy' num2str(sim_step) '.txt']);
model.func(['uy' num2str(sim_step)]).setIndex('fununit', 'm', 0);
model.func(['uy' num2str(sim_step)]).setIndex('argunit', 'm', 0);
model.func(['uy' num2str(sim_step)]).setIndex('argunit', 'm', 1);

model.func.create(['sxx' num2str(sim_step)], 'Interpolation');
model.func(['sxx' num2str(sim_step)]).model(['mod' num2str(sim_step)]);
model.func(['sxx' num2str(sim_step)]).set('source', 'file');
model.func(['sxx' num2str(sim_step)]).set('filename', [pwd '/COMSOL_IO/sxx' num2str(sim_step) '.txt']);
model.func(['sxx' num2str(sim_step)]).setIndex('fununit', 'Pa', 0);
model.func(['sxx' num2str(sim_step)]).setIndex('argunit', 'm', 0);
model.func(['sxx' num2str(sim_step)]).setIndex('argunit', 'm', 1);

model.func.create(['sxy' num2str(sim_step)], 'Interpolation');
model.func(['sxy' num2str(sim_step)]).model(['mod' num2str(sim_step)]);
model.func(['sxy' num2str(sim_step)]).set('source', 'file');
model.func(['sxy' num2str(sim_step)]).set('filename', [pwd '/COMSOL_IO/sxy' num2str(sim_step) '.txt']);
model.func(['sxy' num2str(sim_step)]).setIndex('fununit', 'Pa', 0);
model.func(['sxy' num2str(sim_step)]).setIndex('argunit', 'm', 0);
model.func(['sxy' num2str(sim_step)]).setIndex('argunit', 'm', 1);

model.func.create(['syy' num2str(sim_step)], 'Interpolation');
model.func(['syy' num2str(sim_step)]).model(['mod' num2str(sim_step)]);
model.func(['syy' num2str(sim_step)]).set('source', 'file');
model.func(['syy' num2str(sim_step)]).set('filename', [pwd '/COMSOL_IO/syy' num2str(sim_step) '.txt']);
model.func(['syy' num2str(sim_step)]).setIndex('fununit', 'Pa', 0);
model.func(['syy' num2str(sim_step)]).setIndex('argunit', 'm', 0);
model.func(['syy' num2str(sim_step)]).setIndex('argunit', 'm', 1);
%----------------------------------------------------------
%----------------------------------------------------------
% GEOMETRY
% create rectangle
if flag==0
model.geom.create('geom1', 2);
% model.geom('geom1').scaleUnitValue(true);
model.geom('geom1').lengthUnit('m');
model.geom('geom1').feature.create('r1', 'Rectangle');
model.geom('geom1').feature.create('pt1', 'Point');
model.geom('geom1').feature('r1').set('pos', [0 0]);
model.geom('geom1').feature('r1').set('size', {num2str(xbound) num2str(ybound)});
model.geom('geom1').feature('pt1').set('p',{num2str(p(1));num2str(p(2))});
model.geom('geom1').feature('r1').set('base', 'center');
model.component('mod1').geom('geom1').feature('r1').set('pos', [0 0]);
model.component('mod1').geom('geom1').runPre('fin');
model.geom('geom1').feature('pt1').set('createselection', 'on');
model.geom('geom1').run;
end
%----------------------------------------------------------
%----------------------------------------------------------

% PHYSICS & BOUNDARY CONDITIONS
if flag==0 
model.physics.create('solid', 'SolidMechanics', 'geom1');


% properties
model.physics('solid').feature('lemm1').set('E_mat', 'userdef');
model.physics('solid').feature('lemm1').set('E',[ num2str(E) '[Pa]']);
model.physics('solid').feature('lemm1').set('nu_mat', 'userdef');
model.physics('solid').feature('lemm1').set('nu',[ num2str(nu)]);
model.physics('solid').feature('lemm1').set('rho_mat', 'userdef');
model.physics('solid').feature('lemm1').set('rho',[ num2str(rho)]);

end

% boundary loads
    if flag==0
model.physics('solid').feature.create('bndl1', 'BoundaryLoad', 1);
model.physics('solid').feature('bndl1').selection.set(1);
model.physics('solid').feature.create('bndl2', 'BoundaryLoad', 1);
model.physics('solid').feature('bndl2').selection.set(2);
model.physics('solid').feature.create('bndl3', 'BoundaryLoad', 1);
model.physics('solid').feature('bndl3').selection.set(3);
model.physics('solid').feature.create('bndl4', 'BoundaryLoad', 1);
model.physics('solid').feature('bndl4').selection.set(4);
model.physics('solid').feature.create('bndl5', 'BoundaryLoad', 1);
model.physics('solid').feature('bndl5').selection.set(5);
    end
model.physics('solid').feature('bndl1').set('FperArea', {['sxx' num2str(sim_step) '(' num2str(xmin) ',y[1/m])'];['sxy' num2str(sim_step) '(' num2str(xmin) ',y[1/m])'];'0'});   %left
model.physics('solid').feature('bndl2').set('FperArea', {['sxy' num2str(sim_step) '(x[1/m],' num2str(ymin) ')-' num2str(ext_stress(2,1))];['syy' num2str(sim_step) '(x[1/m],' num2str(ymin) ')-'  num2str(ext_stress(2,2))];'0'}); %bottom
model.physics('solid').feature('bndl3').set('FperArea', {['-sxy' num2str(sim_step) '(x[1/m],' num2str(ymax) ')+' num2str(ext_stress(1,2))];['-syy' num2str(sim_step) '(x[1/m],' num2str(ymax) ')+'  num2str(ext_stress(2,2))];'0'}); %top
model.physics('solid').feature('bndl4').set('FperArea', {['sxy' num2str(sim_step) '(x[1/m],' num2str(ymin) ')-' num2str(ext_stress(2,1))];['syy' num2str(sim_step) '(x[1/m],' num2str(ymin) ')-'  num2str(ext_stress(2,2))];'0'}); %bottom
model.physics('solid').feature('bndl5').set('FperArea', {['-sxx' num2str(sim_step) '(' num2str(xmax) ',y[1/m])'];['-sxy' num2str(sim_step) '(' num2str(xmax) ',y[1/m])'];'0'}); %right


% body load
% model.physics('solid').feature.create('bl1', 'BodyLoad', 2);
% model.physics('solid').feature('bl1').selection.set(1);
% model.physics('solid').feature('bl1').set('FperVol', {'d(sxx(x[1/m],y[1/m]),x)+d(sxy(x[1/m],y[1/m]),y)';'d(sxy(x[1/m],y[1/m]),x)+d(syy(x[1/m],y[1/m]),y)'; '0'});

% initial stress
% model.physics('solid').feature('lemm1').feature.create('iss1', 'InitialStressandStrain', 2);
% model.physics('solid').feature('lemm1').feature('iss1').set('Sil', {'sxx(x[1/m],y[1/m])' 'sxy(x[1/m],y[1/m])' '0' 'sxy(x[1/m],y[1/m])' 'syy(x[1/m],y[1/m])' '0' '0' '0' '0'});

if flag==0
% prescribed displacement at bottom corners and bottom center points
model.physics('solid').feature.create('disp1', 'Displacement0', 0);
model.physics('solid').feature('disp1').selection.set([1 4]);
model.physics('solid').feature.create('disp2', 'Displacement0', 0);
model.physics('solid').feature('disp2').selection.set([3]);
model.physics('solid').feature('disp1').set('Direction', {'0'; '1'; '0'});
model.physics('solid').feature('disp2').set('Direction', {'1'; '0'; '0'});
end

% boundary displacement
% model.physics('solid').feature.create('disp1', 'Displacement1', 1);
% model.physics('solid').feature('disp1').selection.set(2);
% model.physics('solid').feature('disp1').set('U0', {['-ux(x[1/m],' num2str(ymin) ')' ]; ['-uy(x[1/m],' num2str(ymin) ')']; '0'});
% model.physics('solid').feature('disp1').set('Direction', {'1'; '1'; '0'});

%----------------------------------------------------------
%----------------------------------------------------------
% MESH
if flag==0
model.mesh.create('mesh1', 'geom1');
model.mesh('mesh1').feature.create('ftri1', 'FreeTri');
model.mesh('mesh1').run;
model.mesh('mesh1').feature('size').set('hauto', '1');
model.mesh('mesh1').run;
end

%----------------------------------------------------------
%----------------------------------------------------------
% STUDY
model.study.create(['std' num2str(sim_step)]);
model.study(['std' num2str(sim_step)]).feature.create('stat', 'Stationary');
%----------------------------------------------------------
%----------------------------------------------------------
% SOLUTION
model.sol.create(['sol' num2str(sim_step)]);
model.sol(['sol' num2str(sim_step)]).study(['std' num2str(sim_step)]);
model.sol(['sol' num2str(sim_step)]).attach(['std' num2str(sim_step)]);
model.sol(['sol' num2str(sim_step)]).feature.create('st1', 'StudyStep');
model.sol(['sol' num2str(sim_step)]).feature.create('v1', 'Variables');
model.sol(['sol' num2str(sim_step)]).feature.create('s1', 'Stationary');
model.sol(['sol' num2str(sim_step)]).feature('s1').feature.create('fc1', 'FullyCoupled');
model.sol(['sol' num2str(sim_step)]).feature('s1').feature.remove('fcDef');
model.sol(['sol' num2str(sim_step)]).feature('st1').name('Compile Equations: Stationary');
model.sol(['sol' num2str(sim_step)]).feature('st1').set('studystep', 'stat');
model.sol(['sol' num2str(sim_step)]).feature('v1').set('control', 'stat');
model.sol(['sol' num2str(sim_step)]).feature('s1').set('control', 'stat');
%----------------------------------------------------------
%----------------------------------------------------------
% create cut-lines on the boundaries & on crack line

input_c=4*(sim_step-1)+1;

model.result.dataset.create(['cln' num2str(input_c)], 'CutLine2D'); %left
model.result.dataset(['cln' num2str(input_c)]).set('data', ['dset' num2str(sim_step)]);
model.result.dataset(['cln' num2str(input_c)]).set('genpoints', {num2str(xmin) num2str(xmin); num2str(ymin) num2str(ymax)});
model.result.dataset.create(['cln' num2str(input_c+1)], 'CutLine2D'); % bottom
model.result.dataset(['cln' num2str(input_c+1)]).set('data', ['dset' num2str(sim_step)]);
model.result.dataset(['cln' num2str(input_c+1)]).set('genpoints', {num2str(xmin) num2str(xmax); num2str(ymin) num2str(ymin)});
model.result.dataset.create(['cln' num2str(input_c+2)], 'CutLine2D'); %top
model.result.dataset(['cln' num2str(input_c+2)]).set('data', ['dset' num2str(sim_step)]);
model.result.dataset(['cln' num2str(input_c+2)]).set('genpoints', {num2str(xmin) num2str(xmax); num2str(ymax) num2str(ymax)});
model.result.dataset.create(['cln' num2str(input_c+3)], 'CutLine2D'); %right
model.result.dataset(['cln' num2str(input_c+3)]).set('data', ['dset' num2str(sim_step)]);
model.result.dataset(['cln' num2str(input_c+3)]).set('genpoints', {num2str(xmax) num2str(xmax); num2str(ymin) num2str(ymax)});

% model.result.dataset.create('cln5', 'CutLine2D'); %crack line
% model.result.dataset('cln5').set('genpoints', {num2str(FP(1)) num2str(FP(2)); num2str(FP(3)) num2str(FP(4))});

% create cut points on dislocation position
% Pos=load('../Outputs/Positions.txt');
% Xo=Pos(1,:);
% Yo=Pos(2,:);
% for i=1:length(Pos)
%     model.result.dataset.create(['cpt' num2str(i)], 'CutPoint2D');
%     model.result.dataset(['cpt' num2str(i)]).set('pointx', Xo(i));
%     model.result.dataset(['cpt' num2str(i)]).set('pointy', Yo(i));
% end
%----------------------------------------------------------
%----------------------------------------------------------
% RESULT
input_p=6*(sim_step-1)+1;
    
model.result.create(['pg' num2str(input_p)], 'PlotGroup2D');
model.result(['pg' num2str(input_p)]).set('data', ['dset' num2str(sim_step)]);
model.result(['pg' num2str(input_p)]).feature.create('surf1', 'Surface');
model.result(['pg' num2str(input_p)]).feature('surf1').feature.create('def', 'Deform');
model.result(['pg' num2str(input_p)]).name(['Elastic Fields' num2str(sim_step)]);
model.result(['pg' num2str(input_p)]).feature('surf1').set('expr', ['solid.sy+syy' num2str(sim_step) '(x,y)']);
model.result(['pg' num2str(input_p)]).feature('surf1').set('unit', 'MPa');
model.result(['pg' num2str(input_p)]).feature('surf1').set('descr', 'stress tensor, y component');
% model.result(['pg' num2str(input_p)]).feature('surf1').feature('def').set('expr', {'u+ux(x[1/m],y[1/m])' 'v+uy(x[1/m],y[1/m])'});
model.result(['pg' num2str(input_p)]).feature('surf1').feature('def').set('descr', '');
model.result(['pg' num2str(input_p)]).feature('surf1').feature('def').set('scale', '1');
model.result(['pg' num2str(input_p)]).feature('surf1').feature('def').set('scaleactive', false);

model.result.create(['pg' num2str(input_p+1)], 'PlotGroup2D');
model.result(['pg' num2str(input_p+1)]).set('data', ['dset' num2str(sim_step)]);
model.result(['pg' num2str(input_p+1)]).feature.create('SXX', 'Contour');
model.result(['pg' num2str(input_p+1)]).feature('SXX').name('SXX');
model.result(['pg' num2str(input_p+1)]).feature('SXX').set('expr', ['sxx' num2str(sim_step) '(x,y)']);
model.result(['pg' num2str(input_p+1)]).feature('SXX').set('unit', 'MPa');
model.result(['pg' num2str(input_p+1)]).feature('SXX').set('descr', ['sxx' num2str(sim_step) '(x,y)']);

model.result.create(['pg' num2str(input_p+2)], 'PlotGroup2D');
model.result(['pg' num2str(input_p+2)]).set('data', ['dset' num2str(sim_step)]);
model.result(['pg' num2str(input_p+2)]).feature.create('SXY', 'Contour');
model.result(['pg' num2str(input_p+2)]).feature('SXY').name('SXY');
model.result(['pg' num2str(input_p+2)]).feature('SXY').set('expr', ['sxy' num2str(sim_step) '(x,y)']);
model.result(['pg' num2str(input_p+2)]).feature('SXY').set('unit', 'MPa');
model.result(['pg' num2str(input_p+2)]).feature('SXY').set('descr', ['sxy' num2str(sim_step) '(x,y)']);

model.result.create(['pg' num2str(input_p+3)], 'PlotGroup2D');
model.result(['pg' num2str(input_p+3)]).set('data', ['dset' num2str(sim_step)]);
model.result(['pg' num2str(input_p+3)]).feature.create('SYY', 'Contour');
model.result(['pg' num2str(input_p+3)]).feature('SYY').name('SYY');
model.result(['pg' num2str(input_p+3)]).feature('SYY').set('expr', ['syy' num2str(sim_step) '(x,y)']);
model.result(['pg' num2str(input_p+3)]).feature('SYY').set('unit', 'MPa');
model.result(['pg' num2str(input_p+3)]).feature('SYY').set('descr', ['syy' num2str(sim_step) '(x,y)']);

model.result.create(['pg' num2str(input_p+4)], 'PlotGroup2D');
model.result(['pg' num2str(input_p+4)]).set('data', ['dset' num2str(sim_step)]);
model.result(['pg' num2str(input_p+4)]).feature.create('s_total', 'Contour');
model.result(['pg' num2str(input_p+4)]).feature('s_total').name('Elastic Fields');
model.result(['pg' num2str(input_p+4)]).feature('s_total').set('expr', ['syy' num2str(sim_step) '(x,y)']);
model.result(['pg' num2str(input_p+4)]).feature('s_total').set('unit', 'MPa');
model.result(['pg' num2str(input_p+4)]).feature('s_total').set('descr', 'stress tensor, y component');
model.result(['pg' num2str(input_p+4)]).feature('s_total').set('number', '50');

model.result.create(['pg' num2str(input_p+5)], 'PlotGroup2D');
model.result(['pg' num2str(input_p+5)]).set('data', ['dset' num2str(sim_step)]);
model.result(['pg' num2str(input_p+5)]).feature.create('disp_total', 'Contour');
model.result(['pg' num2str(input_p+5)]).feature('disp_total').name('Displacement Fields');
model.result(['pg' num2str(input_p+5)]).feature('disp_total').set('expr', ['v+uy' num2str(sim_step) '(x,y)']);
model.result(['pg' num2str(input_p+5)]).feature('disp_total').set('unit', 'm');
model.result(['pg' num2str(input_p+5)]).feature('disp_total').set('descr', 'displacement, y component');
model.result(['pg' num2str(input_p+5)]).feature('disp_total').set('number', '50');


% model.result.create('pg7', 'PlotGroup1D');
% model.result('pg7').set('probetag', 'none');
% model.result('pg7').set('data', 'cln5');
% model.result('pg7').feature.create('lngr1', 'LineGraph');
% model.result('pg7').feature('lngr1').set('expr', 'solid.sy+syy(x,y)');
% model.result('pg7').feature('lngr1').set('unit', '');
% model.result('pg7').feature('lngr1').set('descr', 'solid.sy+syy(x,y)');
% model.result('pg7').set('xlabel', 'x-coordinate (m)');
% model.result('pg7').set('ylabel', 'solid.sy+syy(x,y)');
% model.result('pg7').set('xlabelactive', false);
% model.result('pg7').set('ylabelactive', false);
% model.result('pg7').feature('lngr1').set('expr', 'solid.sy+syy(x,y)');
% model.result('pg7').feature('lngr1').set('unit', '');
% model.result('pg7').feature('lngr1').set('descr', 'solid.sy+syy(x,y)');
% model.result('pg7').feature('lngr1').set('xdata', 'expr');
% model.result('pg7').feature('lngr1').set('xdataexpr', 'x');
% model.result('pg7').feature('lngr1').set('xdatadescr', 'x-coordinate');
%----------------------------------------------------------
%----------------------------------------------------------
% RUN
model.sol(['sol' num2str(sim_step)]).runAll;
%----------------------------------------------------------
%----------------------------------------------------------
% Plot and save results

% figure(j),mphplot(model,'pg7');hold on
% c1 = ['r' 'k' 'b' 'g' 'c' 'm' 'y' 'r' 'k' 'b' 'g' 'c' 'm' 'y' 'r' 'k' 'b' 'g' 'c' 'm' 'y' ...
%     'r' 'k' 'b' 'g' 'c' 'm' 'y' 'r' 'k' 'b' 'g' 'c' 'm' 'y' 'r' 'k' 'b' 'g' 'c' 'm' 'y'];
% c2 = c1(1:length(Xo)/2);
% COLOR = [c2,wrev(c2)];
% for i=1:length(Xo)
%     plot(Xo(i),Yo(i),['+' num2str(COLOR(i)) ]);hold on
% end
% grid on
% set(gca,'xtick',[FP(1):0.001:FP(3)]);
% grid on
% saveas(gcf,['../results/crack_surface_syy_' num2str(j) '.png']);

% % figure,mphplot(model,'pg2');
% % figure,mphplot(model,'pg3');
% figure,mphplot(model,'pg4');saveas(gcf,'../results/SYY_Contour.png');
% figure,mphplot(model,'pg5');saveas(gcf,'../results/Elastic_Fields_Contour.png');
% figure,mphplot(model,'pg6');saveas(gcf,'../results/Displ_Contour.png');

% Evaluate results
% fid=fopen(['./COMSOL_IO/corrected_dislocation_stress' num2str(j) '.txt'],'w+');

for i=1:n_c
    for k=1:n_branch
        for m=1:C(i).B(k).n_modes
            for j=1:2*C(i).B(k).mode(m).n_d
    sxxFEM=mphinterp(model,'solid.sx','coord',[C(i).B(k).mode(m).D(j).R_p(1)*Length_unit;C(i).B(k).mode(m).D(j).R_p(2)*Length_unit]);
    sxyFEM=mphinterp(model,'solid.sxy','coord',[C(i).B(k).mode(m).D(j).R_p(1)*Length_unit;C(i).B(k).mode(m).D(j).R_p(2)*Length_unit]);
    syyFEM=mphinterp(model,'solid.sy','coord',[C(i).B(k).mode(m).D(j).R_p(1)*Length_unit;C(i).B(k).mode(m).D(j).R_p(2)*Length_unit]);
    C(i).B(k).mode(m).D(j).Corrected_Stress=[sxxFEM,sxyFEM,0;
        sxyFEM,syyFEM,0;
        0   ,  0   ,0]/G;
            end 
        end
    end    
end


return


DS =cell(length(Xo),1);
xo1=FP(1);
xo2=FP(3);
Lo=length(Xo);

% L=Lo+2;
% xnew=ones(1,L);
% add in the middle
% xnew(1:L/2-1)=Xo(1:Lo/2);
% xnew(L/2+2:end)=Xo(Lo/2+1:end);
% xnew(L/2)=xo1/6;
% xnew(L/2+1)=xo2/6;
% add at the before end
% xnew(1)=Xo(1);
% xnew(end)=Xo(end);
% xnew(3:end-2)=Xo(2:end-1);
% xnew(2)=0.98*xo1;
% xnew(end-1)=0.98*xo2;
% pause
xnew=Xo;
L=Lo;
Yo=zeros(L,1);
Burgers(1:L/2)=-1;
Burgers(L/2+1:L)=1;

for i=1:L
    sxx=mphinterp(model,'solid.sx+sxx(x,y)','coord',[xnew(i);Yo(i)]);
    sxy=mphinterp(model,'solid.sxy+sxy(x,y)','coord',[xnew(i);Yo(i)]);
    syy=mphinterp(model,'solid.sy+syy(x,y)','coord',[xnew(i);Yo(i)]);
    sxxFEM=mphinterp(model,'solid.sx','coord',[xnew(i);Yo(i)]);
    sxyFEM=mphinterp(model,'solid.sxy','coord',[xnew(i);Yo(i)]);
    syyFEM=mphinterp(model,'solid.sy','coord',[xnew(i);Yo(i)]);
    DisStressFEM{i} = [sxxFEM,sxyFEM,syyFEM];
    DisStressAll{i} = [sxx,sxy,syy];
    sigma=[sxx,sxy,0;sxy,syy,0;0,0,0];
    PK1=cross(sigma*bo*[0;Burgers(i);0],t);
    
    fprintf(fid,'%e %e %e %e %e %e %e %e  %e \n',xnew(i),Yo(i),sxxFEM,sxyFEM,syyFEM,sxx,sxy,syy,PK1(1));
end


% DS=cell(4,1);
% DS{1}=SS1;
% DS{2}=SS2;
% DS{3}=SS1FEM;
% DS{4}=SS2FEM;


