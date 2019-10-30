p = 0.95;
N = 100;
N_s = 5;
N_min = 2;
alpha = 4;
in = alpha*1/N;
out = 1-in;


%Generic RANSAC
w = 1/N; %proportion of inliers from all possible bipartite connections
k1 = log(1 - p)./log(1 - in^N_min)

P_atleast2 = 1 - (out)^N_s - N_s*(in)*(out)^(N_s-1);
k2 = log(1-p)/log(1-P_atleast2)

reduction = k2/k1