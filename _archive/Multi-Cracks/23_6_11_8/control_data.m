function control_data()

global n_c n_grid del_grid eps sig_amp n_d_min out_iter_max...
    inner_iter_max n_steps sig_e recomb_L 

n_c=30;
n_d_min=2;
eps=1e-3;
out_iter_max=1;
inner_iter_max=200*n_d_min;
% inner_iter_max=1;
n_steps=5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recomb_L=4*n_steps;
sig_amp=1e-3;
sig_e=sig_amp*[1 1 0;1 1 0;0 0 0];
n_grid=200;
del_grid=100;
get_cracks()
initial()


        end
