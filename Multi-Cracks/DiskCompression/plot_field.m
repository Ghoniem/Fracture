function plot_field(S_xx,S_yy,S_xy,U,V)
global X Y C n_c x_cod_up y_cod_up x_cod_down y_cod_down  ...
    Length_unit G  limit

fsize=14;
zlevs=500;
color_scale2=3;
color_scale1=-3;

x=X*Length_unit*1e3;
y=Y*Length_unit*1e3;
for i=1:n_c
    n_dis=2*C(i).n_d;
        R_Left=C(i).D(1).R_p-5*C(i).b_mag*C(i).e_p;
        R_Right=C(i).D(n_dis).R_p+5*C(i).b_mag*C(i).e_p;
    if C(i).D(1).recomb==0
            xx_cod_up=[x_cod_up(i,2:n_dis),R_Right(1)];
            xx_cod_down=[x_cod_down(i,2:n_dis),R_Right(1)];
            yy_cod_up=[y_cod_up(i,2:n_dis),R_Right(2)];
            yy_cod_down=[y_cod_down(i,2:n_dis),R_Right(2)];
    elseif C(i).D(n_dis).recomb==0
            xx_cod_up=[R_Left(1),x_cod_up(i,1:n_dis-1)];
            xx_cod_down=[R_Left(1),x_cod_down(i,1:n_dis-1)];
            yy_cod_up=[R_Left(2),y_cod_up(i,1:n_dis-1)];
            yy_cod_down=[R_Left(2),y_cod_down(i,1:n_dis-1)];
    else
            xx_cod_up=[R_Left(1),x_cod_up(i,1:n_dis),R_Right(1)];
            xx_cod_down=[R_Left(1),x_cod_down(i,1:n_dis),R_Right(1)];
        yy_cod_up=[R_Left(2),y_cod_up(i,1:n_dis),R_Right(2)];
        yy_cod_down=[R_Left(2),y_cod_down(i,1:n_dis),R_Right(2)];

    end 
    C(i).x_u=xx_cod_up*Length_unit*1e3;
    C(i).x_d=xx_cod_down*Length_unit*1e3;
    C(i).y_u=yy_cod_up*Length_unit*1e3;
    C(i).y_d=yy_cod_down*Length_unit*1e3;
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
                plot( C(i).x_u, C(i).y_u,'r-',C(i).x_d, C(i).y_d,'r-','LineWidth',1)
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
            plot( C(i).x_u, C(i).y_u,'r-',C(i).x_d, C(i).y_d,'r-','LineWidth',1)
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
            plot( C(i).x_u, C(i).y_u,'r-',C(i).x_d, C(i).y_d,'r-','LineWidth',1)
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