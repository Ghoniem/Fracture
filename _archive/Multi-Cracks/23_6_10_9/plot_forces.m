
function plot_forces(i)
global  t position sig_amp C
fsize=14;
f_scale=sig_amp*C(i).b_mag;
figure(i+3)

plot(position(i,:),t(i,:)/f_scale,'o','LineWidth',2)
xlabel('Position [b]','FontName','Times New Roman','FontSize',fsize)
ylabel('Climb Force','FontSize',fsize,'FontName', 'Times New Roman')
set(gca,'fontsize',fsize,'fontname','Times New Roman')
grid on
% hold on

 end

