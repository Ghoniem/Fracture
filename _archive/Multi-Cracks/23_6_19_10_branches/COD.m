function [c_x_up,c_y_up,c_x_down,c_y_down]=COD(i,j,k)
global C sig_amp

c_scale=0.02/sig_amp;
if j <= C(i).B(k).n_d 
    r_cod_up=(j)*c_scale*C(i).B(k).b_mag*C(i).B(k).b_direction+C(i).B(k).D(j).R_p+c_scale*C(i).B(k).b_mag*C(i).B(k).e_n;
    r_cod_down=-(j)*c_scale*C(i).B(k).b_mag*C(i).B(k).b_direction+C(i).B(k).D(j).R_p-c_scale*C(i).B(k).b_mag*C(i).B(k).e_n;
 elseif j > C(i).B(k).n_d 
    r_cod_up=(2*C(i).B(k).n_d-j+1)*c_scale*C(i).B(k).b_mag*C(i).B(k).b_direction+C(i).B(k).D(j).R_p+c_scale*C(i).B(k).b_mag*C(i).B(k).e_n;
    r_cod_down=-(2*C(i).B(k).n_d-j+1)*c_scale*C(i).B(k).b_mag*C(i).B(k).b_direction+C(i).B(k).D(j).R_p-c_scale*C(i).B(k).b_mag*C(i).B(k).e_n;
end
c_x_up=r_cod_up(1);
c_y_up=r_cod_up(2);
c_x_down=r_cod_down(1);
c_y_down=r_cod_down(2);
end