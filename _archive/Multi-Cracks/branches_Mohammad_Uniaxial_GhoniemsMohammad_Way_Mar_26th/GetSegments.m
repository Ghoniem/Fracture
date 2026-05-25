function GetSegments(propagate)

global n_c n_branch K_IC C 

while propagate==1
    propagate=0;
for i=1:n_c
    for k=1:n_branch
        for m=1:2
        if C(i).B(k).K(m) > K_IC                
              C(i).B(n_branch+1).a = C(i).B(k).a/2*(1-K_IC/C(i).B(k).K(m));
              if C(i).B(k).case == "mixed"
                 C(i).B(n_branch+1).phi = acosd((3*C(i).B(k).mode(2).K.^2+sqrt((C(i).B(k).mode(1).K.^4 ...
                 +8.*C(i).B(k).mode(2).K.^2.*C(i).B(k).mode(1).K.^2)))./(C(i).B(k).mode(1).K.^2+9.*C(i).B(k).mode(2).K.^2));
                 % rset(i,k,1,2);
                 % rset(i,k,2,2);
              else
                  C(i).B(n_branch+1).phi = 0;
                 % rset(i,k,1,2);
              end
              if m==1
                 x = C(i).B(k).R_c(1)-C(i).B(k).a*cos(C(i).B(k).phi)-C(i).B(n_branch+1).a*cos(C(i).B(n_branch+1).phi);
                 y =  C(i).B(k).R_c(2)-C(i).B(k).a*sin(C(i).B(k).phi)-C(i).B(n_branch+1).a*sin(C(i).B(n_branch+1).phi);
              else
                  x = C(i).B(k).R_c(1)+C(i).B(k).a*cos(C(i).B(k).phi)+C(i).B(n_branch+1).a*cos(C(i).B(n_branch+1).phi);
                  y =  C(i).B(k).R_c(2)+C(i).B(k).a*sin(C(i).B(k).phi)+C(i).B(n_branch+1).a*sin(C(i).B(n_branch+1).phi); 
              end
              C(i).B(n_branch+1).R_c=[x y 0];
              initial_Segment(i,n_branch+1)
        end
        end
    end
Dislocations_Distribution(0)
end
end
end



