function GetCrackSegments(propagate)

global n_c n_branch K_IC C sim_step Length_unit G

while propagate==1
    propagate=0;
for i=1:n_c
    for k=1:n_branch
        for m=1:2
        if C(i).B(k).K(m) > K_IC 
               propagate=1;
              C(i).B(k+1).a = C(i).B(k).a/2*(1-K_IC/C(i).B(k).K(1));
              if C(i).B(k).case == "mixed"
                 C(i).B(k+1).phi = acosd((3*C(i).B(k).mode(2).K.^2+sqrt((C(i).B(k).mode(1).K.^4 ...
                 +8.*C(i).B(k).mode(2).K.^2.*C(i).B(k).mode(1).K.^2)))./(C(i).B(k).mode(1).K.^2+9.*C(i).B(k).mode(2).K.^2));
              else
                  C(i).B(k+1).phi = 0;
              end
              if m==1
                 x = C(i).B(k).R_c(1)-C(i).B(k).a*cos(C(i).B(k).phi)-C(i).B(k+1).a*cos(C(i).B(k+1).phi);
                 y =  C(i).B(k).R_c(1)-C(i).B(k).a*sin(C(i).B(k).phi)-C(i).B(k+1).a*sin(C(i).B(k+1).phi);
              else
                  x = C(i).B(k).R_c(1)+C(i).B(k).a*cos(C(i).B(k).phi)+C(i).B(k+1).a*cos(C(i).B(k+1).phi);
                  y =  C(i).B(k).R_c(1)+C(i).B(k).a*sin(C(i).B(k).phi)+C(i).B(k+1).a*sin(C(i).B(k+1).phi); 
              end
              C(i).B(k+1).R_c=[x y 0];
         end
           

           if propagate==1
           initial_Segment(i,k+1)
           

           for m=1:C(i).B(k+1).n_modes
              for j=1:2*C(i).B(k+1).mode(m).n_d
    sxxFEM=mphinterp(model,'solid.sx','coord',[C(i).B(k).mode(m).D(j).R_p(1)*Length_unit;C(i).B(k).mode(m).D(j).R_p(2)*Length_unit]);
    sxyFEM=mphinterp(model,'solid.sxy','coord',[C(i).B(k).mode(m).D(j).R_p(1)*Length_unit;C(i).B(k).mode(m).D(j).R_p(2)*Length_unit]);
    syyFEM=mphinterp(model,'solid.sy','coord',[C(i).B(k).mode(m).D(j).R_p(1)*Length_unit;C(i).B(k).mode(m).D(j).R_p(2)*Length_unit]);
    C(i).B(k+1).mode(m).D(j).Corrected_Stress=[sxxFEM,sxyFEM,0;
        sxyFEM,syyFEM,0;
        0   ,  0   ,0]/G;
              end 
           end
sim_step = sim_step +1;
        Dislocations_Distribution(0)
        GetDislStressField()
           end
        end
    end
end

end
end


