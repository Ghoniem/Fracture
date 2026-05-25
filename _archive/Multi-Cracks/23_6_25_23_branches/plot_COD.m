function plot_COD()
global C limit X_intercept Y_intercept n_c n_branch

fsize=14;

figure(200);
        for i=1:n_c
            for k=1:n_branch   
            plot( C(i).B(k).x_u, C(i).B(k).y_u,'r-',C(i).B(k).x_d, C(i).B(k).y_d,'r-','LineWidth',1)
            hold on
            end
        end
xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
ylabel('y [mm]','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on
xlim([-limit limit]); ylim([-limit limit])
pbaspect([1 1 1]);
hold on

th = 0:pi/50:2*pi;
xunit =limit * cos(th) ;
yunit =limit * sin(th) ;
plot(xunit, yunit,'c-','LineWidth',4);
hold on



% plot(X_intercept(i,:),Y_intercept(i,:),'b.','LineWidth',1)

 end