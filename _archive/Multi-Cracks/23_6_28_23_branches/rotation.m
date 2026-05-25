function QQ=rotation (i,j,k)
global C 


        if j<=C(i).B(k).n_d
            C(i).B(k).D(j).rot=-pi/2+C(i).B(k).phi+C(i).B(k).theta;

          else
            C(i).B(k).D(j).rot=pi/2+C(i).B(k).phi+C(i).B(k).theta;

        end


            sn=sin(C(i).B(k).D(j).rot);
            cs=cos(C(i).B(k).D(j).rot);
            C(i).B(k).D(j).Q=[cs sn 0; -sn cs 0;0 0 1];
            QQ=C(i).B(k).D(j).Q;
end
