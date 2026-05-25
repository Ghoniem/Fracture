function plot_COD()
global C limit X_intercept Y_intercept n_c n_branch 

fsize=14;

        for i=1:n_c
            for k=1:n_branch   
            switch(C(i).B(k).mode)
                case 'pure'
                    figure(200);
            plot( C(i).B(k).x_u, C(i).B(k).y_u,'r-',C(i).B(k).x_d, C(i).B(k).y_d,'r-','LineWidth',1)
            % limits for uniaxial case
            xlim([-limit limit]); ylim([-limit limit])
            hold on
                case 'mixed'
                    figure(200);
                    plot( C(i).B(k).x_u1, C(i).B(k).y_u1,'r-',C(i).B(k).x_d1, C(i).B(k).y_d1,'r-','LineWidth',1)
             % limits for uniaxial case
                    xlim([-limit limit]); ylim([-limit limit])
                    hold on
                    figure(201)
                    plot( C(i).B(k).x_u2, C(i).B(k).y_u2,'b-',C(i).B(k).x_d2, C(i).B(k).y_d2,'r-','LineWidth',1)
             % limits for uniaxial case
                    xlim([-limit limit]); ylim([-limit limit])
                    hold on
            end
            end
        end

        % Plot disk boundaries
%         figure(200);
% xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
% ylabel('y [mm]','FontSize',fsize,'FontName', 'Times New Roman')
% set(gca,'fontsize',fsize,'fontname','Times New Roman')
% grid on
% xlim([-limit limit]); ylim([-limit limit])
% pbaspect([1 1 1]);
% hold on
% 
% th = 0:pi/50:2*pi;
% xunit =limit * cos(th) ;
% yunit =limit * sin(th) ;
% plot(xunit, yunit,'c-','LineWidth',4);
% hold on

% Plot disk boundaries
% %     figure(201);
% % xlabel('x [mm]','FontName','Times New Roman','FontSize',fsize)
% % ylabel('y [mm]','FontSize',fsize,'FontName', 'Times New Roman')
% % set(gca,'fontsize',fsize,'fontname','Times New Roman')
% % grid on
% % xlim([-limit limit]); ylim([-limit limit])
% % pbaspect([1 1 1]);
% % hold on
% % 
% % th = 0:pi/50:2*pi;
% % xunit =limit * cos(th) ;
% % yunit =limit * sin(th) ;
% % plot(xunit, yunit,'c-','LineWidth',4);
% % hold on


% plot(X_intercept(i,:),Y_intercept(i,:),'b.','LineWidth',1)

 end