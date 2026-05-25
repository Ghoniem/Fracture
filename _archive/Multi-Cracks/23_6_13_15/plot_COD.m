function plot_COD(i)
global x_cod_up y_cod_up x_cod_down y_cod_down C del_grid n_grid X_intercept Y_intercept


fsize=14;n_dis=2*C(i).n_d;fig_num=20;

           
limit=n_grid*del_grid;
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

figure (fig_num)
plot(xx_cod_up,yy_cod_up,'r-',....
    xx_cod_down,yy_cod_down,'r-','LineWidth',1)
xlabel('x [b]','FontName','Times New Roman','FontSize',fsize)
ylabel('y [b]','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on
xlim([0 limit]); ylim([0 limit])
pbaspect([1 1 1]);
hold on
plot(X_intercept(i,:),Y_intercept(i,:),'b.','LineWidth',1)

 end