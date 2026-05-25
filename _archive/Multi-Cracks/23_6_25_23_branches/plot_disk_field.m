function plot_disk_field(S_xx,S_yy,S_xy)
global X Y Length_unit   limit G

fsize=14;
zlevs=80;
color_scale2=3;
color_scale1=-3;

x=X*Length_unit*1e3;
y=Y*Length_unit*1e3;

% Z=S_yy;
% zmin = floor(min(Z(:))); 
% % zmax = ceil(max(Z(:)));
% zmax=2*sig_amp;
% zinc = (zmax - zmin) / 100;
% zlevs = zmin:zinc:zmax;


th = 0:pi/50:2*pi;
xunit =limit * cos(th) ;
yunit =limit * sin(th) ;

figure(101)
plot(xunit, yunit,'k-','LineWidth',4);
hold on
    contourf(x,y,S_xx*G*1e-6,zlevs);
    colormap(parula(256));
    colorbar;
    pbaspect([1 1 1]);
    xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
    ylabel('y [mm]','FontSize',14,'FontName', 'Times New Roman')
    set(gca,'fontsize',14,'fontname','Times New Roman')
    legend('\sigma_{xx} [MPa]','location','northoutside')
hold off

    figure(102)
plot(xunit, yunit,'k-','LineWidth',4);
hold on
    contourf(x,y,S_yy*G*1e-6,zlevs);
    colormap(parula(256));
    colorbar;
    pbaspect([1 1 1]);
    xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
    ylabel('y [mm]','FontSize',14,'FontName', 'Times New Roman')
    set(gca,'fontsize',14,'fontname','Times New Roman')
legend('\sigma_{yy} [MPa]','location','northoutside')
hold off

    figure(103)
plot(xunit, yunit,'k-','LineWidth',4);
hold on
    contourf(x,y,S_xy*G*1e-6,zlevs);
    colormap(parula(256));
    colorbar;
    pbaspect([1 1 1]);
    xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
    ylabel('y [mm]','FontSize',14,'FontName', 'Times New Roman')
    set(gca,'fontsize',14,'fontname','Times New Roman')
    legend('\sigma_{xy} [MPa]','location','northoutside')
hold off

end
