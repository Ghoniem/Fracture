function [c_x_up,c_y_up,c_x_down,c_y_down]=COD(i,j)
global C sig_amp

c_scale=0.1/sig_amp;
if j <= C(i).n_d 
    r_cod_up=(j)*c_scale*C(i).b_mag*C(i).b_direction+C(i).D(j).R_p+c_scale*C(i).b_mag*C(i).e_n;
    r_cod_down=-(j)*c_scale*C(i).b_mag*C(i).b_direction+C(i).D(j).R_p-c_scale*C(i).b_mag*C(i).e_n;
 elseif j > C(i).n_d 
    r_cod_up=(2*C(i).n_d-j+1)*c_scale*C(i).b_mag*C(i).b_direction+C(i).D(j).R_p+c_scale*C(i).b_mag*C(i).e_n;
    r_cod_down=-(2*C(i).n_d-j+1)*c_scale*C(i).b_mag*C(i).b_direction+C(i).D(j).R_p-c_scale*C(i).b_mag*C(i).e_n;
end
c_x_up=r_cod_up(1);
c_y_up=r_cod_up(2);
c_x_down=r_cod_down(1);
c_y_down=r_cod_down(2);
end