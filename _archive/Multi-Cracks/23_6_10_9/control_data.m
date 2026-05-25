function control_data()

global n_c n_grid del_grid eps sig_amp n_d_min out_iter_max...
    inner_iter_max n_steps sig_e recomb_L tough_factor

n_c=9;
n_d_min=2;
n_grid=200;
del_grid=100;
eps=1e-3;
out_iter_max=1;
inner_iter_max=100*n_d_min;
% inner_iter_max=10;
recomb_L=5;
n_steps=3;
sig_amp=1e-3;
sig_e=sig_amp*[1 1 0;1 1 0;0 0 0];
tough_factor=100;

get_cracks()

        end
