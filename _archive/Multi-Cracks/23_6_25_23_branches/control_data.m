function control_data()

global n_c n_grid del_grid eps sig_amp n_d_min iter_max...
   n_steps sig_e recomb_L n_b a_crack D Length_unit G limit E n_branch ...
   tough_limit c_scale P h alpha q resistance

%%%%%%   GENERAL CONTROL KNOBS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_c=1;  
n_branch=1;                %  Number of cracks
n_d_min=5;              % minimum number of dipoles on crack branches
eps=1e-3;               % Convergence criterion for PK forces
n_steps=2;              % number of steps for a dislocation to reach its neighbor, the smaller the faster they move
n_b=1;                  % number of random circular barriers
iter_max=300;           % max iterations for convergence= iter_max*n_d_min
recomb_L=5*n_steps;     % displocation-dislocation recombination length

%%%%%%%%%%%%%%%%%%%%%%%%%   MATERIAL DATA AND SAMPLE GEOMETRY   %%%%%%%%%%%

E=231.52e9 ;            % Elastic modulus of Alumina [Pa]
nu=1/3;                 % Poisson's ratio of Alumina
D=25.4e-3;               % Disk diameter [m]
crack_size=4e-3;         %%%%% initial crack size (2a=8 mm)
P=1e3;                  % disk load [N]
h=5e-3;                 %disk thickness [m]
alpha=pi/10;             % Disk pressure angle
q=200e6;                 % Disk pressure [Pa]
%%%% %%%%%%%%%%%% CALCULATE NON-DIMENSIONAL VALUES %%%%%%%%%%%
G=E/(2*(1+nu));
sig_app=1e5;          %applied stress [Pa]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig_amp=sig_app/G;                      % Relative applied stress amplitude
sig_e=sig_amp*[1 0 0;0 0 0;0 0 0];      % Stress tensor
n_grid=100;                             % grid points for plotting fields
del_grid=50;                           % distance between grid points
Length_unit=D/(n_grid*del_grid);        %%%  length unit [m]
a_crack=crack_size/(2*Length_unit);     %% Relative crack length
limit=n_grid*del_grid*Length_unit*1e3/2;
tough_limit=n_grid*del_grid/1;
c_scale=0.05/sig_amp;   % controls how COD looks like on a plot
resistance=0*sig_amp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

get_cracks()
initial()


        end
