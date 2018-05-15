function g = steady_cont_bistable(u,lambda)

%% space discretisaiton
% Domain where I would be solving the FP equation: x in (-L/2,L/2)
L = 10;                  
dx = 0.05;
N = L/dx;
x = dx*(0:N);
x = x-L/2;
x = x';        

    
%% parameters
theta = 0.5;
beta = lambda;

%% confining potential

V = @(y) y.^4/4 - y.^2/2 ;

%% get m and R(m)
% the vector ss stores the steady state solution p_inf(x;m) AND R(m) for
% THAT p
ss = u(1:N+1);     % recover p (the current guess for the steady state)
m = u(end);       % recover m for the current guess

SS = exp(-beta*(V(x)+theta*x.^2/2 - theta*x*m)); 
SS = SS/trapz(x,SS);

F_out = zeros(N+2,1);
F_out(1:N+1) = ss-SS;
result = m-trapz(x,x.*SS)
F_out(end) =  result;   %computes R(m)-m and solves for R(m)-m=0

g=F_out;
end