function [X,Y]=generate_grid()
global del_grid n_grid
%%%%%  Rectangular Grid  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         x_grid=del_grid*(0:n_grid-1);
%         y_grid=del_grid*(0:n_grid-1);
%         [X,Y]=meshgrid(x_grid,y_grid);
%%   Polar Grid %%%%%%%%%%%%%%%%%%
rr = del_grid*(0:n_grid-1)/2;
thth = (0:1/(n_grid-1):1)*2*pi;
[r, th] = meshgrid(rr,thth);
X = r.*cos(th);
Y = r.*sin(th);
end
