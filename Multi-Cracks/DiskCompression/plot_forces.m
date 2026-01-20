
function plot_forces(i)
global  t position G C Length_unit E a_crack
fsize=14;
f_scale=G*C(i).b_mag*Length_unit;
K=1e-6*(t(i,:)*f_scale);
% x=position(i,:)*Length_unit*1e3-a_crack*1e-3;
x=position(i,:)*Length_unit*1e3;

figure(i+3)

plot(x,K,'o','LineWidth',2)
xlabel('Position [mm]','FontName','Times New Roman','FontSize',fsize)
ylabel('Climb Force [MPa.(m)]','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on
% hold on

 end

