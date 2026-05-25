function plot_field(S_xx,S_yy,S_xy,U,V)
global X Y C n_c x_cod_up y_cod_up x_cod_down y_cod_down  ...
    Length_unit G  limit n_branch

fsize=14;
zlevs=500;
color_scale2=3;
color_scale1=-3;

x=X*Length_unit*1e3;
y=Y*Length_unit*1e3;
for i=1:n_c
  for k=1:n_branch   
    n_dis=2*C(i).B(k).n_d;
        R_Left=C(i).B(k).D(1).R_p-5*C(i).B(k).b_mag*C(i).B(k).e_p;
        R_Right=C(i).B(k).D(n_dis).R_p+5*C(i).B(k).b_mag*C(i).B(k).e_p;
    if C(i).B(k).D(1).recomb==0
            xx_cod_up=[x_cod_up(i,2:n_dis,k),R_Right(1)];
            xx_cod_down=[x_cod_down(i,2:n_dis,k),R_Right(1)];
            yy_cod_up=[y_cod_up(i,2:n_dis,k),R_Right(2)];
            yy_cod_down=[y_cod_down(i,2:n_dis,k),R_Right(2)];
    elseif C(i).B(k).D(n_dis).recomb==0
            xx_cod_up=[R_Left(1),x_cod_up(i,1:n_dis-1,k)];
            xx_cod_down=[R_Left(1),x_cod_down(i,1:n_dis-1,k)];
            yy_cod_up=[R_Left(2),y_cod_up(i,1:n_dis-1,k)];
            yy_cod_down=[R_Left(2),y_cod_down(i,1:n_dis-1,k)];
    else
            xx_cod_up=[R_Left(1),x_cod_up(i,1:n_dis,k),R_Right(1)];
            xx_cod_down=[R_Left(1),x_cod_down(i,1:n_dis,k),R_Right(1)];
        yy_cod_up=[R_Left(2),y_cod_up(i,1:n_dis,k),R_Right(2)];
        yy_cod_down=[R_Left(2),y_cod_down(i,1:n_dis,k),R_Right(2)];

    end 
    C(i).B(k).x_u=xx_cod_up*Length_unit*1e3;
    C(i).B(k).x_d=xx_cod_down*Length_unit*1e3;
    C(i).B(k).y_u=yy_cod_up*Length_unit*1e3;
    C(i).B(k).y_d=yy_cod_down*Length_unit*1e3;
  end
end



% Z=S_yy;
% zmin = floor(min(Z(:))); 
% % zmax = ceil(max(Z(:)));
% zmax=2*sig_amp;
% zinc = (zmax - zmin) / 100;
% zlevs = zmin:zinc:zmax;


th = 0:pi/50:2*pi;
xunit =limit * cos(th) ;
yunit =limit * sin(th) ;

figure(1)
plot(xunit, yunit,'k-','LineWidth',4);
hold on
    contourf(x,y,S_xx*G*1e-6,zlevs);
    colormap(parula(256));
    colorbar;
    pbaspect([1 1 1]);
    xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
    ylabel('y [mm]','FontSize',14,'FontName', 'Times New Roman')
    set(gca,'fontsize',14,'fontname','Times New Roman')
%     caxis([color_scale1, color_scale2]);
            for i=1:n_c
                plot( C(i).B(k).x_u, C(i).B(k).y_u,'r-',C(i).B(k).x_d, C(i).B(k).y_d,'r-','LineWidth',1)
                hold on
            end
    hold off
        legend('\sigma_{xx} [MPa]','location','northoutside')



figure(2)
plot(xunit, yunit,'k-','LineWidth',4);
hold on
    contourf(x,y,S_yy*G*1e-6,zlevs);
    colormap(jet(256));
    colorbar;
    pbaspect([1 1 1]);
    xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
    ylabel('y [mm]','FontSize',14,'FontName', 'Times New Roman')
    set(gca,'fontsize',14,'fontname','Times New Roman')
%     caxis([color_scale1, color_scale2]);
        for i=1:n_c
            plot( C(i).B(k).x_u, C(i).B(k).y_u,'r-',C(i).B(k).x_d, C(i).B(k).y_d,'r-','LineWidth',1)
            hold on
        end
    hold off
    legend('\sigma_{yy} [MPa]','location','northoutside')



figure(3)
plot(xunit, yunit,'k-','LineWidth',4);
hold on
    contourf(x,y,S_xy*G*1e-6,zlevs);
    colorbar;
    colormap(jet(256));
    pbaspect([1 1 1]);
    xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
    ylabel('y [mm]','FontSize',14,'FontName', 'Times New Roman')
    set(gca,'fontsize',14,'fontname','Times New Roman')
%     caxis([color_scale1, color_scale2]);
        for i=1:n_c
            plot( C(i).B(k).x_u, C(i).B(k).y_u,'r-',C(i).B(k).x_d, C(i).B(k).y_d,'r-','LineWidth',1)
            hold on
        end
    hold off
     legend('\sigma_{xy} [MPa]','location','northoutside')


% disp_scale=10;
% U=disp_scale*U;
% V=disp_scale*V;
%  figure (11)
%  quiver(X,Y,U,V,'r');
%  axis equal
% 
%  figure (12)
%  plot(X+U,Y+V,'.','LineWidth',0.1)
%  axis equal
%  grid on


end