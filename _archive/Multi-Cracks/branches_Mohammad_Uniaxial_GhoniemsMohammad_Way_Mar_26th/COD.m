function [c_x_up,c_y_up,c_x_down,c_y_down]=COD(i,j,k,m)
global C c_scale


if j <= C(i).B(k).mode(m).n_d 
    r_cod_up=(j)*c_scale*C(i).B(k).mode(m).b_mag*C(i).B(k).mode(m).b_direction+C(i).B(k).mode(m).D(j).R_p+c_scale*C(i).B(k).mode(m).b_mag*C(i).B(k).e_n;
    r_cod_down=-(j)*c_scale*C(i).B(k).mode(m).b_mag*C(i).B(k).mode(m).b_direction+C(i).B(k).mode(m).D(j).R_p-c_scale*C(i).B(k).mode(m).b_mag*C(i).B(k).e_n;
 elseif j > C(i).B(k).mode(m).n_d 
    r_cod_up=(2*C(i).B(k).mode(m).n_d-j+1)*c_scale*C(i).B(k).mode(m).b_mag*C(i).B(k).mode(m).b_direction+C(i).B(k).mode(m).D(j).R_p...
        +c_scale*C(i).B(k).mode(m).b_mag*C(i).B(k).e_n;
    r_cod_down=-(2*C(i).B(k).mode(m).n_d-j+1)*c_scale*C(i).B(k).mode(m).b_mag*C(i).B(k).mode(m).b_direction+C(i).B(k).mode(m).D(j).R_p...
        -c_scale*C(i).B(k).mode(m).b_mag*C(i).B(k).e_n;
end
c_x_up=r_cod_up(1);
c_y_up=r_cod_up(2);
c_x_down=r_cod_down(1);
c_y_down=r_cod_down(2);

end

