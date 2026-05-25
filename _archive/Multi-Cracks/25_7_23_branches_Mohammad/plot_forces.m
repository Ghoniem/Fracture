
function plot_forces(i,k,flag)
global  t position G C Length_unit

switch(C(i).B(k).mode)
    case 'pure'

fsize=14;
f_scale=G*C(i).B(k).b_mag*Length_unit;
K=1e-6*(t(i,:,k)*f_scale);
% x=position(i,:,k)*Length_unit*1e3-a_crack*1e-3;
x=position(i,:,k)*Length_unit*1e3;

figure(i*k+3)

plot(x,K,'o','LineWidth',2)
xlabel('Position [mm]','FontName','Times New Roman','FontSize',fsize)
ylabel('Climb Force [MPa.(m)]','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on
% hold on

    case 'mixed'

fsize=14;

        switch(flag)

            case 1

f_scale1=G*C(i).B(k).b_mag1*Length_unit;
K1=1e-6*(t(i,:,k,1)*f_scale1);
x1=position(i,:,k,1)*Length_unit*1e3;

figure(i*k+3)

plot(x1,K1,'o','LineWidth',2)
xlabel('Position [mm]','FontName','Times New Roman','FontSize',fsize)
ylabel('Climb Force [MPa.(m)]','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on

            case 2

f_scale2=G*C(i).B(k).b_mag2*Length_unit;
K2=1e-6*(t(i,:,k,2)*f_scale2);
x2=position(i,:,k,2)*Length_unit*1e3;

figure(i*k+4)

plot(x2,K2,'o','LineWidth',2)
xlabel('Position [mm]','FontName','Times New Roman','FontSize',fsize)
ylabel('Climb Force [MPa.(m)]','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on

        end
end
end

