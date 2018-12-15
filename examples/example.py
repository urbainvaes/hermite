import hermipy as hm
import sympy as sym

# Set library options
hm.settings['sparse'] = True
hm.settings['tensorize'] = True

x, y = sym.symbols('x y', real=True)
f = sym.Function('f')(x, y)

# Set parameters for the Fokker-Planck equation
theta, beta, epsilon, m = .2, 1, .1, .2
Vp, Vy = x**4/4 - x**2/2 + theta*(x-m)**2/2, y**2/2

# Construct the Fokker-Planck operator
flux_x = - (Vp.diff(x)*f - sym.sqrt(1/beta)*y*f/epsilon)
flux_y = - (1/epsilon**2) * (f*Vy.diff(y) + f.diff(y))
fp_operator = - flux_x.diff(x) - flux_y.diff(y)

d = 50             # Number of basis functions in each direction
sx, sy = .05, .05  # Scaling of basis functions along x and y

# Set multiplier function
# (= ratio between basis functions and scaled Hermite polynomials)
factor = sym.exp(-x*x/sx/4 - y*y/sy/4) * sym.exp(- beta*Vp/2 - Vy/2)

# Define a quadrature
quad = hm.Quad.gauss_hermite(
        n_points=(2*d + 1),      # Number of quadrature points
        dim=2,                   # Dimension of the problem
        mean=[0, 0],             # Optional translation
        cov=[[sx, 0], [0, sy]],  # Scaling of basis functions
        factor=factor)           # Multiplier function

# Discretize the operator using (d + 1)**2 basis functions
discrete_operator = quad.discretize_op(fp_operator, d, index_set='cube')

# Calculate the eigenvector corresponding to the stationary solution
[e_val], [e_vec] = discrete_operator.eigs(k=1, which='LR')

# Plot the solution
quad.plot(e_vec)
