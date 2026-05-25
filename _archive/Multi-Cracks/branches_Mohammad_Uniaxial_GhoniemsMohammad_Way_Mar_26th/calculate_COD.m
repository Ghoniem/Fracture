function calculate_COD()
global C Length_unit n_c n_branch x_cod_up y_cod_up x_cod_down y_cod_down 

for i=1:n_c
  for k=1:n_branch  

      switch(C(i).B(k).mode)
          
          case 'pure'
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

          case 'mixed'

        n_dis=2*C(i).B(k).n_d;
        R_Left_1=C(i).B(k).D1(1).R_p-2*C(i).B(k).b_mag1*C(i).B(k).e_p;
        R_Right_1=C(i).B(k).D1(n_dis).R_p+2*C(i).B(k).b_mag1*C(i).B(k).e_p;
        xx_cod_up_1=[R_Left_1(1)+C(i).B(k).b_mag1,x_cod_up(i,1:n_dis,k,1),R_Right_1(1)];
        xx_cod_down_1=[R_Left_1(1)-C(i).B(k).b_mag1,x_cod_down(i,1:n_dis,k,1),R_Right_1(1)];
        yy_cod_up_1=[R_Left_1(2)+C(i).B(k).b_mag1,y_cod_up(i,1:n_dis,k,1),R_Right_1(2)];
        yy_cod_down_1=[R_Left_1(2)-C(i).B(k).b_mag1,y_cod_down(i,1:n_dis,k,1),R_Right_1(2)];

        R_Left_2=C(i).B(k).D2(1).R_p-2*C(i).B(k).b_mag2*C(i).B(k).e_p;
        R_Right_2=C(i).B(k).D2(n_dis).R_p+2*C(i).B(k).b_mag2*C(i).B(k).e_p;
        xx_cod_up_2=[R_Left_2(1)+C(i).B(k).b_mag2,x_cod_up(i,1:n_dis,k,2),R_Right_2(1)];
        xx_cod_down_2=[R_Left_2(1)-C(i).B(k).b_mag2,x_cod_down(i,1:n_dis,k,2),R_Right_2(1)];
        yy_cod_up_2=[R_Left_2(2)+C(i).B(k).b_mag2,y_cod_up(i,1:n_dis,k,2),R_Right_2(2)];
        yy_cod_down_2=[R_Left_2(2)-C(i).B(k).b_mag2,y_cod_down(i,1:n_dis,k,2),R_Right_2(2)];


        C(i).B(k).x_u1=xx_cod_up_1*Length_unit*1e3;
        C(i).B(k).x_d1=xx_cod_down_1*Length_unit*1e3;
        C(i).B(k).y_u1=yy_cod_up_1*Length_unit*1e3;
        C(i).B(k).y_d1=yy_cod_down_1*Length_unit*1e3;

        C(i).B(k).x_u2=xx_cod_up_2*Length_unit*1e3;
        C(i).B(k).x_d2=xx_cod_down_2*Length_unit*1e3;
        C(i).B(k).y_u2=yy_cod_up_2*Length_unit*1e3;
        C(i).B(k).y_d2=yy_cod_down_2*Length_unit*1e3;

      end
  end
end
end