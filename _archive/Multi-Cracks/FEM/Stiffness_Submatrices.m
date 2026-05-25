function [K_ff, K_fs]=Stiffness_Submatrices(K,Support,Free)

K_ff=K(Free,Free);
K_fs=K(Free,Support);
K_sf=K(Support,Free);
K_ss=K(Support,Support);

end