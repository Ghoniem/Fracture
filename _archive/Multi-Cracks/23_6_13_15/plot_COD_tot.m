function plot_COD_tot()

global x_cod_up y_cod_up x_cod_down y_cod_down C DX DY JJ_max II_max
fsize=14;n_dis=2*C(2).n_d;


R_Left=C(1).D(1).R_p-5*C(1).b_mag*C(1).e_p;
R_Right=C(2).D(n_dis).R_p+5*C(2).b_mag*C(2).e_p;

x_up=[R_Left(1),x_cod_up(1,:), x_cod_up(2,:),R_Right(1)];
y_up=[R_Left(2),y_cod_up(1,:), y_cod_up(2,:),R_Right(2)];
x_dn=[R_Left(1),x_cod_down(1,:), x_cod_down(2,:),R_Right(1)];
y_dn=[R_Left(2),y_cod_down(1,:), y_cod_down(2,:),R_Right(2)];

figure (30)
plot(x_up,y_up,'r-',....
   x_dn,y_dn,'r-','LineWidth',1)
xlabel('x [b]','FontName','Times New Roman','FontSize',fsize)
ylabel('y [b]','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on
xlim([0 DX*II_max]); ylim([0 DY*JJ_max])
hold on




end
