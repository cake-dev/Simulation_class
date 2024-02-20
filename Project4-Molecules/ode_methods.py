import numpy as np

class ParticleMotion1D:
    """ This is an example class for an ODE specification"""
    
    def __init__(self,g=-9.81):
        
        self.n_dof = 2
        self.g = g
        
    def rhs(self,t,u):
        # the right hand side of the ode (or $\mathcal{F}(t,u)$)
        dudt = np.zeros(self.n_dof)
        dudt[0] = u[1]
        dudt[1] = self.g
        return dudt
    
class ParticleMotion2DWithDrag:
    
    def __init__(self, g=-9.81, c=1.0, m=1.0):
        self.n_dof = 4  # number of degrees of freedom
        self.g = g  # acceleration due to gravity
        self.c = c  # drag coefficient
        self.m = m  # mass of the particle

    def rhs(self, t, u):
        # the right hand side of the ode (or $\mathcal{F}(t,u)$)
        dudt = np.zeros(self.n_dof)
        dudt[0] = u[1]  # dx/dt = vx (velocity is the derivative of position)
        dudt[1] = -self.c / self.m * np.sqrt(u[1]**2 + u[3]**2) * u[1]  # dvx/dt = -c/m * sqrt(vx^2 + vz^2) * vx (drag force)
        dudt[2] = u[3]  # dz/dt = vz 
        dudt[3] = self.g - self.c / self.m * np.sqrt(u[1]**2 + u[3]**2) * u[3]  # dvz/dt = g - c/m * sqrt(vx^2 + vz^2) * vz 
        return dudt
    

class Euler:
    def __init__(self):
        pass   
    
    def step(self,ode,t,dt,u_0):
        u_1 = u_0 + dt*ode.rhs(t,u_0)
        return u_1
    
class Heun:
    def __init__(self):
        pass   
    
    def step(self,ode,t,dt,u_0):
        # Do some stuff here
        k1 = ode.rhs(t,u_0)
        k2 = ode.rhs(t+dt,u_0+dt*k1)
        u_1 = u_0 + 0.5*dt*(k1+k2)
        return u_1
    
class RK4:
    def __init__(self):
        pass
    
    def step(self,ode,t,dt,u_0):
        k1 = dt*ode.rhs(t,u_0)
        k2 = dt*ode.rhs(t+0.5*dt,u_0+(0.5*k1))
        k3 = dt*ode.rhs(t+0.5*dt,u_0+(0.5*k2))
        k4 = dt*ode.rhs(t+dt,u_0+k3)
        u_1 = u_0 + 1/6*(k1+2*k2+2*k3+k4)
        return u_1

class Integrator:
    def __init__(self,ode,method): # ode is provided as one of the ParticleMotion classes, method is one of the integrator classes
        self.ode = ode # store the ode
        self.method = method # store the method
        
    def integrate(self,interval,dt,u_0): # interval is a list [t_0,t_end], dt is the time step, u_0 is the initial state
        t_0 = interval[0] # initial time
        t_end = interval[1] # final time
        
        times = [t_0] # list to store the times
        states = [u_0] # list to store the states
        
        t = t_0 # current time
        while t<t_end: # loop over the time interval
            dt_ = min(dt,t_end-t) # time step
            u_1 = self.method.step(self.ode,t,dt_,u_0) # integrate the ODE
            t = t + dt_ # update the time
            u_0 = u_1 # update the state
            
            times.append(t) # store the time
            states.append(u_1) # store the state
            
        return np.array(times),np.array(states) # return the times and states as numpy arrays
    

class PlanetaryMotion:
    def __init__(self, masses, G=6.67e-11):
        self.masses = masses  # Array of masses for each particle
        self.G = G  # Gravitational constant
        self.N = len(masses)  # Number of particles
        
    def rhs(self, t, u):
        """
        Compute the right-hand side of the ODEs for positions and velocities.
        
        Parameters:
        - t: Time variable (not used, as the force is time-independent)
        - u: A flattened array containing positions and velocities of all particles
        
        Returns:
        - dudt: A flattened array of the derivatives of positions and velocities
        """
        # extract positions and velocities from the state vector
        positions = u[:2*self.N].reshape((self.N, 2)) # reshape for easier indexing in loop
        velocities = u[2*self.N:].reshape((self.N, 2))
        
        # initialize derivatives of positions (which are just the velocities)
        dxdt = velocities
        
        # initialize derivatives of velocities (accelerations) as zeros
        dvdt = np.zeros_like(velocities, dtype=float)
        
        # compute gravitational forces and resulting accelerations
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    r_ij = positions[j] - positions[i]  # Vector from particle i to particle j
                    dist_ij = np.linalg.norm(r_ij)  # Distance between particles i and j
                    # update acceleration of particle i due to particle j
                    dvdt[i] += self.G * self.masses[j] * r_ij / dist_ij**3
        
        # flatten the derivatives to match the structure of the input state vector [x1, y1, x2, y2, ... vx1, vy1, vx2, vy2, ...]
        dudt = np.concatenate([dxdt.flatten(), dvdt.flatten()])
        return dudt
    
class EulerCromer:
    def __init__(self, ode):
        self.ode = ode

    def step(self, t, dt, u_0):
        rhs_result = self.ode.rhs(t, u_0) # gives us the derivatives i.e. for 2 bodies [vx1,vy1,vx2,vy2,dvx1/dt,dvy1/dt,dvx2/dt,dvy2/dt]
        v = u_0[2*self.ode.N:] + rhs_result[2*self.ode.N:] * dt # update the velocities (2*self.ode.N: is the index of the velocities in the state vector u_0)
        u = u_0[:2*self.ode.N] + v * dt # update the positions with the new velocities
        return np.concatenate([u, v]) # return the new state vector [x1,y1,x2,y2,vx1,vy1,vx2,vy2]