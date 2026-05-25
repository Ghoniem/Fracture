function plot_COD()
global C limit X_intercept Y_intercept n_c n_branch

fsize=14;


        for i=1:n_c
            for k=1:n_branch   
            switch(C(i).B(k).mode)
                case 'pure'
                    figure(200);
            plot( C(i).B(k).x_u, C(i).B(k).y_u,'r-',C(i).B(k).x_d, C(i).B(k).y_d,'r-','LineWidth',1)
            hold on
                case 'mixed'
                    figure(200);
                    plot( C(i).B(k).x_u_1, C(i).B(k).y_u_1,'r-',C(i).B(k).x_d_1, C(i).B(k).y_d_1,'r-','LineWidth',1)
                    hold on
                    figure(201)
                    plot( C(i).B(k).x_u_2, C(i).B(k).y_u_2,'b-',C(i).B(k).x_d_2, C(i).B(k).y_d_2,'r-','LineWidth',1)
                    hold on
            end
            end
        end
        figure(200);
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

    figure(201);
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