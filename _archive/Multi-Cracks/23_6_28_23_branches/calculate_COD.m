function calculate_COD()
global C Length_unit n_c n_branch x_cod_up y_cod_up x_cod_down y_cod_down
for i=1:n_c
  for k=1:n_branch   
    n_dis=2*C(i).B(k).n_d;
        R_Left=C(i).B(k).D(1).R_p-2*C(i).B(k).b_mag*C(i).B(k).e_p;
        R_Right=C(i).B(k).D(n_dis).R_p+2*C(i).B(k).b_mag*C(i).B(k).e_p;
        xx_cod_up=[R_Left(1)+C(i).B(k).b_mag,x_cod_up(i,1:n_dis,k),R_Right(1)];
        xx_cod_down=[R_Left(1)-C(i).B(k).b_mag,x_cod_down(i,1:n_dis,k),R_Right(1)];
        yy_cod_up=[R_Left(2)+C(i).B(k).b_mag,y_cod_up(i,1:n_dis,k),R_Right(2)];
        yy_cod_down=[R_Left(2)-C(i).B(k).b_mag,y_cod_down(i,1:n_dis,k),R_Right(2)];

        C(i).B(k).x_u=xx_cod_up*Length_unit*1e3;
        C(i).B(k).x_d=xx_cod_down*Length_unit*1e3;
        C(i).B(k).y_u=yy_cod_up*Length_unit*1e3;
        C(i).B(k).y_d=yy_cod_down*Length_unit*1e3;
  end
end
end