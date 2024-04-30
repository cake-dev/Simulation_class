import numpy as np
import matplotlib.pyplot as plt
import ode_methods as om

def get_dt_max(dx,k):
    return dx**2/(2*k)

class DiffusionLinear:
    def __init__(self, nx, dx, k=0.01, dir_left=False, dir_right=False, neu_left=False, neu_right=False):
        self.nx = nx
        self.dx = dx
        self.k = k
        self.dir_left = dir_left
        self.dir_right = dir_right
        self.neu_left = neu_left
        self.neu_right = neu_right
        self.A = self.create_matrix()

    def create_matrix(self):
        # Initialize matrix
        A = np.zeros((self.nx, self.nx))

        # Fill the matrix
        for i in range(self.nx):
            if i == 0 or i == self.nx - 1: # First and last row are boundary conditions
                A[i, i] = 0
            else:
                A[i, i - 1] = 1
                A[i, i] = -2
                A[i, i + 1] = 1
        
        return A

    def rhs(self, t, u):
        u_bounded = u.copy()
        dudt = self.k / self.dx**2 * np.dot(self.A, u_bounded) # Compute the right-hand side (du/dt)
        if self.neu_right:
            dudt[-1] = dudt[-2]
        if self.neu_left:
            dudt[0] = dudt[1]
        return dudt

def diffuExplicit(dleft=False, dright=False, nleft=False, nright=False, _k=0.01, _dt=0.005, _t_span=(0, 50)):
    # Setup
    L = 1.0  # Length of the domain
    nx = 101  # Number of grid points
    dx = L / (nx - 1)  # Grid spacing
    x = np.linspace(0, L, nx)  # Position array
    u0 = np.zeros(nx)  # Initial condition array
    d_left = dleft
    d_right = dright
    n_left = nleft
    n_right = nright
    if d_left or n_right:
        u0[0] = 1
    if d_right or n_left:
        u0[-1] = 1

    # Parameters
    k = _k  # Diffusion coefficient
    dt = _dt  # Time step
    t_span = _t_span  # Time span

    diffusion_model = DiffusionLinear(nx, dx, k, d_left, d_right, n_left, n_right)
    euler_method = om.Euler()
    integrator = om.Integrator(diffusion_model, euler_method)

    # Solve
    t, u = integrator.integrate(t_span, dt, u0)

    # Visualization
    cmap = plt.cm.viridis
    for tt, uu in zip(t[::100], u[::100]): # Plot every 100th solution
        plt.plot(x, uu, color=cmap(tt/t[-1]))
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Diffusion with Boundary Conditions (Linear Approach)")
    # Show which conditions are active on the plot, text box upper right
    if d_left:
        plt.text(0.5, 0.9, "Dirichlet left", transform=plt.gca().transAxes)
    if d_right:
        plt.text(0.5, 0.865, "Dirichlet right", transform=plt.gca().transAxes)
    if n_left:
        plt.text(0.5, 0.830, "Neumann left", transform=plt.gca().transAxes)
    if n_right:
        plt.text(0.5, 0.795, "Neumann right", transform=plt.gca().transAxes)
    plt.show()