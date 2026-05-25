function control_data()

global n_c n_grid del_grid eps sig_amp n_d_min iter_max...
   n_steps sig_e recomb_L n_b a_crack D Length_unit G limit E n_branch

%%%%%%   GENERAL CONTROL KNOBS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_c=2;  
n_branch=2;                %  Number of cracks
n_d_min=2;              % Number of dipoles in each crack
eps=1e-3;               % Convergence criterion for PK forces
n_steps=2;              % number of steps for a dislocation to reach its neighbor, the smaller the faster they move
n_b=1;                  % number of random circular barriers
iter_max=300;           % max iterations for convergence= iter_max*n_d_min
recomb_L=5*n_steps;     % displocation-dislocation recombination length

%%%%%%%%%%%%%%%%%%%%%%%%%   MATERIAL DATA AND SAMPLE GEOMETRY   %%%%%%%%%%%

E=231.52e9 ;            % Elastic modulus of Alumina [Pa]
nu=1/3;                 % Poisson's ratio of Alumina
D=25.4e-3;               % Disk diameter [m]
crack_size=8e-3;         %%%%% initial crack size (2a=8 mm)

%%%% %%%%%%%%%%%% CALCULATE NON-DIMENSIONAL VALUES %%%%%%%%%%%
G=E/(2*(1+nu));
sig_app=10e6;          %applied stress [Pa]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig_amp=sig_app/G;                      % Relative applied stress amplitude
sig_e=sig_amp*[1 0 0;0 0 0;0 0 0];      % Stress tensor
n_grid=100;                             % grid points for plotting fields
del_grid=50;                           % distance between grid points
Length_unit=D/(n_grid*del_grid);        %%%  length unit [m]
a_crack=crack_size/(2*Length_unit);     %% Relative crack length
limit=n_grid*del_grid*Length_unit*1e3/2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

get_cracks()
initial()


        end
