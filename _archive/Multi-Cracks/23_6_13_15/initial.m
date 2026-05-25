
function initial()
global  n_c n_grid del_grid C  ...
        S_xx S_yy S_xy Sigma_xx Sigma_yy Sigma_xy X Y  force_ratio ...
         e_x e_y e_z t position u_x u_y U V n_d_min x_cod_up y_cod_up ...
         x_cod_down y_cod_down sig_amp sig_e R_min R_max theta_min ...
         theta_max n_b C_C_tol
e_x=[1 0 0];
e_y=[0 1 0];
e_z=[0 0 1];
L=zeros(n_c);
for i=1:n_c
L(i)=C(i).a;
end
a_min=min(L);
C_C_tol=a_min(1,1)/10;

for i=1:n_c
        C(i).n_d=n_d_min*fix(L(i)/a_min(1,1));
        C(i).e_p=[cos(C(i).phi), sin(C(i).phi), 0];
        C(i).e_n=[-sin(C(i).phi), cos(C(i).phi),0];
        C(i).R_L=C(i).R_c-C(i).a*C(i).e_p;
        C(i).R_R=C(i).R_c+C(i).a*C(i).e_p;
        C(i).d=2*C(i).a/(2*C(i).n_d-1);
        t=ones(n_c,2*C(i).n_d)*sig_amp;
        position=zeros(n_c,2*C(i).n_d);
        x_cod_up=zeros(n_c,2*C(i).n_d);
        y_cod_up=zeros(n_c,2*C(i).n_d);
        x_cod_down=zeros(n_c,2*C(i).n_d);
        y_cod_down=zeros(n_c,2*C(i).n_d);
        C(i).steps=max(50,C(i).d/10);
        C(i).iter_max=10*C(i).steps;
        C(i).b_mag=2*a_min(1,1)*sig_amp/n_d_min;
        C(i).trac=sig_e*C(i).e_n';
        C(i).b_direction=C(i).trac'/norm(C(i).trac);
        C(i).theta=atan2(C(i).trac(2),C(i).trac(1))+pi/2-C(i).phi;
    for j=1:2*C(i).n_d     
        C(i).D(j).R_p=C(i).R_L+C(i).d*(j-1)*C(i).e_p;
        C(i).D(j).F_PK=ones(3);
        C(i).D(j).b=-C(i).b_mag*C(i).b_direction;
        C(i).D(j).recomb=1;


            if j<=C(i).n_d
            C(i).D(j).orientation=+1;
            else
            C(i).D(j).orientation=-1;
            end
    end
end

        S_xx=zeros(n_grid,n_grid);
        S_yy=zeros(n_grid,n_grid);
        S_xy=zeros(n_grid,n_grid);
        Sigma_xx=zeros(n_grid,n_grid);
        Sigma_yy=zeros(n_grid,n_grid);
        Sigma_xy=zeros(n_grid,n_grid);
        u_x=zeros(n_grid,n_grid);
        u_y=zeros(n_grid,n_grid);
        U=zeros(n_grid,n_grid);
        V=zeros(n_grid,n_grid);
        x_grid=del_grid*(0:n_grid-1);
        y_grid=del_grid*(0:n_grid-1);
        [X,Y]=meshgrid(x_grid,y_grid);
        force_ratio=ones(1,n_c);


R_min=zeros(n_b,1);
R_max=zeros(n_b,1);
theta_min=zeros(n_b,1);
theta_max=zeros(n_b,1);


end