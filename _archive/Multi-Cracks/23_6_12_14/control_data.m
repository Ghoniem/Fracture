function control_data()

global n_c n_grid del_grid eps sig_amp n_d_min max_iterations...
   n_steps sig_e recomb_L n_b

n_c=1;
n_d_min=3;
eps=1e-3;
max_iterations=200*n_d_min;
n_steps=5;
n_b=200;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recomb_L=4*n_steps;
sig_amp=1e-3;
sig_e=sig_amp*[0 0 0;0 1 0;0 0 0];
n_grid=200;
del_grid=100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

get_cracks()
initial()


        end
