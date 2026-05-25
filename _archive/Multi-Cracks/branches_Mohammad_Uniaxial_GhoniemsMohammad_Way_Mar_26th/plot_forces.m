
function plot_forces(i,k,m)
global   G C Length_unit

fsize=14;
f_scale=G*C(i).B(k).mode(m).b_mag*Length_unit;
K=1e-6*(C(i).B(k).mode(m).t(:)*f_scale);
% x=position(i,:,k)*Length_unit*1e3-a_crack*1e-3;
x=C(i).B(k).mode(m).position(:)*Length_unit*1e3;

figure(i*k+3+(m-1))

plot(x,K,'o','LineWidth',2)
xlabel('Position [mm]','FontName','Times New Roman','FontSize',fsize)
ylabel('Climb Force [MPa.(m)]','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on
% hold on

 
end

