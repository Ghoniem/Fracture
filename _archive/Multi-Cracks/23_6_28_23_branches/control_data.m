function control_data()

global n_c n_grid del_grid eps sig_amp n_d_min iter_max...
   n_steps sig_e recomb_L a_crack D Length_unit G nu limit E n_branch ...
   tough_limit c_scale alpha q resistance C C_T C_L lambda

%%%%%%   GENERAL CONTROL KNOBS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_c=1;  
n_branch=1;                %  Number of cracks
n_d_min=4;              % minimum number of dipoles on crack branches
eps=1e-3;               % Convergence criterion for PK forces
n_steps=5;              % number of steps for a dislocation to reach its neighbor, the smaller the faster they move
iter_max=100;           % max iterations for convergence= iter_max*n_d_min
recomb_L=5*n_steps;     % displocation-dislocation recombination length

%%%%%%%%%%%%%%%%%%%%%%%%%   MATERIAL DATA AND SAMPLE GEOMETRY   %%%%%%%%%%%

E=231.52e9 ;            % Elastic modulus of Alumina [Pa]
nu=1/3;                 % Poisson's ratio of Alumina
G=E/(2*(1+nu));
lambda=E*nu/((1-2*nu)*(1+nu));  %Lame's ConstantRegisterCompileCheck
rho=3.95e3;             % density, [kg/m^3]
C_T=sqrt(G/rho);       % Transverse speed of sound [m/s]
C_L=sqrt((lambda+2*G)/rho);       % Longitudinal speed of sound [m/s]

%%%% %%%%%%%%%%%% LOADING CONDITIONS, now for a disk %%%%%%%%%%%

D=25.4e-3;               % Disk diameter [m]
crack_size=4e-3;         %%%%% initial crack size (2a=8 mm)
% P=1e3;                  % disk load [N]
% h=5e-3;                 %disk thickness [m]
alpha=pi/15;             % Disk pressure angle
q=500e6;                 % Disk pressure [Pa]

%%%%%%%%%%%%%%%%%%%  GRID AND UNITS   %%%%%%%%%%%%%%%%%%%%%%%%%%
n_grid=100;                             % grid points for plotting fields
del_grid=50;                           % distance between grid points
Length_unit=D/(n_grid*del_grid);        %%%  length unit [m]
a_crack=crack_size/(2*Length_unit);     %% Relative crack length
limit=n_grid*del_grid*Length_unit*1e3/2;
tough_limit=n_grid*del_grid/1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

get_cracks()

sig_e=disk_field_pressure(C(1).B(1).R_c);      % External stress tensor
sig_amp=max(abs(sig_e),[],"all");             % Relative applied stress amplitude
c_scale=0.05/sig_amp;                       % controls how COD looks like on a plot
resistance=0*sig_amp;

initial()


        end
