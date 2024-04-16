import numpy as np

class Integrator:
    def __init__(self,ode,method):
        self.ode = ode
        self.method = method
        
    def integrate(self,interval,dt,u_0):
        t_0 = interval[0]
        t_end = interval[1]
        
        times = [t_0]
        states = [u_0]
        
        t = t_0
        while t<t_end:
            dt_ = min(dt,t_end-t)
            u_1 = self.method.step(self.ode,t,dt_,u_0)
            t = t + dt_
            u_0 = u_1
            
            times.append(t)
            states.append(u_1)
            
        return np.array(times),np.array(states)

class Verlet:
    def __init__(self):
        pass   
    
    def step(self,ode,t,dt,u_0):
        u_half = u_0 + dt/2.*ode.rhs(t,u_0)
        u_1 = u_0 + dt*ode.rhs(t,u_half)
        dudt_end = ode.rhs(t,u_1)
        u_2 = u_half + dt/2.*dudt_end
        u_final = np.zeros_like(u_0)
        u_final[:ode.n_dof*2] = u_1[:ode.n_dof*2]
        u_final[ode.n_dof*2:] = u_2[ode.n_dof*2:]

        return u_final
    
class Euler:
    def __init__(self):
        pass   
    
    def step(self,ode,t,dt,u_0):
        return u_0 + dt*ode.rhs(t,u_0)
    
class Cromer:
    def __init__(self):
        pass   
    
    def step(self,ode,t,dt,u_0):
        u_star = u_0 + dt*ode.rhs(t,u_0)
        #u_star[:len(u_star)//2] = u_star[:len(u_star)//2] % L
        u_1 = u_0 + dt*ode.rhs(t,u_star)
        #u_1[:len(u_1)//2] = u_1[:len(u_1)//2] % L
        u_final = np.zeros_like(u_0)
        u_final[:ode.n_dof*2] = u_1[:ode.n_dof*2]
        u_final[ode.n_dof*2:] = u_star[ode.n_dof*2:]
        return u_final

class RungeKutta2:
    def __init__(self):
        pass
    def step(self,ode,t,dt,u_0):
        k1 = ode.rhs(t, u_0)
        k2 = ode.rhs(t + dt, u_0 + dt*k1)
        u_1 = u_0 + dt/2.*(k1 + k2)
        return u_1

class Midpoint:
    def __init__(self):
        pass
    def step(self,ode,t,dt,u_0):
        k1 = ode.rhs(t, u_0)
        k2 = ode.rhs(t + dt/2., u_0 + dt/2.*k1)
        u_1 = u_0 + dt*k2
        return u_1

class RungeKutta4:
    def __init__(self):
        pass
    def step(self,ode,t,dt,u_0):
        k1 = ode.rhs(t,u_0)
        k2 = ode.rhs(t + dt/2.,u_0 + dt/2.*k1)
        k3 = ode.rhs(t + dt/2.,u_0 + dt/2.*k2)
        k4 = ode.rhs(t+dt,u_0 + dt*k3)
        u_1 = u_0 + dt/6.*(k1 + 2*k2 + 2*k3 + k4)
        return u_1



