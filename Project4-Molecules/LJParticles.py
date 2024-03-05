import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import HTML
import ode_methods as om
import matplotlib
import pygame
from numba import jit, float64, int64, boolean, types

@jit(float64[:](float64[:], float64[:], float64), nopython=True)
def periodic_distance(xi,xj,L):
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]
    
    if dx>(0.5*L):
        dx -= L
    elif dx<(-0.5*L):
        dx += L
    
    if dy>(0.5*L):
        dy -= L
    elif dy<(-0.5*L):
        dy += L
    
    return np.array([dx,dy])

@jit(nopython=True)
def rhs(t,u, N, epsilon, sigma, box_size=None):
    positions = u[:2*N].reshape((N, 2))
    velocities = u[2*N:].reshape((N, 2))
    dxdt = velocities
    dvdt = np.zeros(velocities.shape, dtype=np.float64)

    for i in range(N):
        for j in range(i+1, N): # Avoid double calculation and self interaction
            r_ij = positions[j] - positions[i]
            # Apply minimum image convention for periodic boundary conditions
            if box_size is not None:
                r_ij = periodic_distance(positions[i], positions[j], box_size)
            r = np.linalg.norm(r_ij)
            dvdt[i] += -24*epsilon/r*(2*(sigma/r)**12 - (sigma/r)**6) * (r_ij/r)
            dvdt[j] -= -24*epsilon/r*(2*(sigma/r)**12 - (sigma/r)**6) * (r_ij/r) # Newton's third law

    dudt = np.empty(dxdt.size + dvdt.size, dtype=np.float64)
    dudt[:dxdt.size] = dxdt.flatten()
    dudt[dxdt.size:] = dvdt.flatten()
    return dudt

class LennardJones:
    
    def __init__(self, sigma, epsilon, N, masses, box_size=None):
        self.sigma = sigma
        self.epsilon = epsilon
        self.N = N
        self.masses = masses
        self.box_size = box_size

    def periodic_distance(self, xi,xj,L):
        return periodic_distance(xi, xj, L)
    
    def rhs(self,t,u):
        return rhs(t,u, self.N, self.epsilon, self.sigma, self.box_size)
        # positions = u[:2*self.N].reshape((self.N, 2))
        # velocities = u[2*self.N:].reshape((self.N, 2))
        # dxdt = velocities
        # dvdt = np.zeros_like(velocities, dtype=float)

        # for i in range(self.N):
        #     for j in range(i+1, self.N): # Avoid double calculation and self interaction
        #         r_ij = positions[j] - positions[i]
        #         # Apply minimum image convention for periodic boundary conditions
        #         if self.box_size is not None:
        #             r_ij = self.periodic_distance(positions[i], positions[j], self.box_size)
        #         r = np.linalg.norm(r_ij)
        #         dvdt[i] += -24*self.epsilon/r*(2*(self.sigma/r)**12 - (self.sigma/r)**6) * (r_ij/r)
        #         dvdt[j] -= -24*self.epsilon/r*(2*(self.sigma/r)**12 - (self.sigma/r)**6) * (r_ij/r) # Newton's third law

        # dudt = np.concatenate([dxdt.flatten(), dvdt.flatten()])
        # return dudt
    
class Cromer:
    def __init__(self,callbacks=[]):
        self.callbacks = callbacks
    
    def step(self,ode,t,dt,u_0):
        u_star = u_0 + dt*ode.rhs(t,u_0)
        for c in self.callbacks:
            u_star = c.apply(u_star)
        u_1 = u_0 + dt*ode.rhs(t,u_star)
        for c in self.callbacks:
            u_1 = c.apply(u_1)
        u_final = np.zeros_like(u_0)
        u_final[:ode.N*2] = u_1[:ode.N*2]
        u_final[ode.N*2:] = u_star[ode.N*2:]
        return u_final
    
class PBCCallback:
    def __init__(self,position_indices,L):
        """ Accepts a list of which degrees of freedom in u are positions 
        (which varies depending on how you organized them) as well as a 
        maximum domain size. """
        self.position_indices = position_indices
        self.L = L
                 
    def apply(self,u):
        # Set the positions to position modulo L
        u[self.position_indices] = u[self.position_indices] % self.L
        return u
    

# a class for both the simulation and the animation
    
@jit(nopython=True)
def fixPositions(positions, box_size, sigma):
    for i in range(positions.shape[0]):
        for j in range(i+1, positions.shape[0]):
            while np.linalg.norm(positions[i]-positions[j]) < 2**(1/6)*sigma:
                positions[j] = np.random.rand(2)*box_size
    return positions
class LJParticleSim:

    def __init__(self, N, box_size, sigma, epsilon, dt, t_end, x0, v0, isRandom=False):
        self.N = N
        self.box_size = box_size
        self.sigma = sigma
        self.epsilon = epsilon
        self.dt = dt
        self.t_end = t_end
        self.num_frames = 0
        self.total_energy = 0
        # ensure that particles are within the box and also not overlapping
        if isRandom:
            x0 = x0.reshape((N, 2))
            x0 = fixPositions(x0, box_size, sigma)
            # for i in range(N):
            #     for j in range(i+1, N):
            #         while np.linalg.norm(x0[i]-x0[j]) < 2**(1/6)*sigma:
            #             x0[j] = np.random.rand(2)*box_size
            x0 = x0.flatten()
        self.x0 = x0
        self.v0 = v0
        self.lj = LennardJones(sigma, epsilon, N, np.ones(N), box_size)
        self.integrator = om.Integrator(self.lj, Cromer([PBCCallback(np.arange(2*N), box_size)]))
        self.times, self.states = self.integrator.integrate([0, t_end], dt, np.concatenate([x0, v0]))

    def animate(self):
        # Set the limit to 30 MB (or whatever value you prefer)
        matplotlib.rcParams['animation.embed_limit'] = 50
        fig, ax = plt.subplots()
        fig.set_size_inches(4,4)
        L = self.box_size
        u = np.array(self.states)
        N = u.shape[1] // 4
        positions = u[0, :2*N].reshape((N, 2))
        im = plt.plot(positions[:, 0], positions[:, 1], 'ro')
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        def animate(frame_number):
            positions = u[frame_number, :2*N].reshape((N, 2))
            im[0].set_xdata(positions[:, 0])
            im[0].set_ydata(positions[:, 1])
            return im
        animation = anim.FuncAnimation(fig, animate, frames=len(u), interval=1);
        self.num_frames = len(u)
        # save animation to a file
        animation.save('lj_particles.mp4', writer='ffmpeg', fps=240);
        return 0;
        # return HTML(animation.to_jshtml());

    def animate_pygame(self):
        pygame.init()

        # Set up some constants
        WIDTH, HEIGHT = 600, 600
        FPS = 240
        DT = 1.0 / FPS  # Time step

        # Create the Pygame window
        screen = pygame.display.set_mode((WIDTH, HEIGHT))

        # Convert states to integer coordinates
        states = np.array(self.states)
        N = states.shape[1] // 4  # Number of particles
        positions = states[:, :2*N].reshape((-1, N, 2))
        velocities = states[:, 2*N:].reshape((-1, N, 2))

        # Scale positions and velocities to screen size
        positions = positions / self.box_size * np.array([WIDTH, HEIGHT])
        velocities = velocities / self.box_size * np.array([WIDTH, HEIGHT])

        # Main loop
        running = True
        frame_number = 0
        while running:
            # Limit the frame rate
            pygame.time.Clock().tick(FPS)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Clear the screen
            screen.fill((0, 0, 0))

            # Draw the particles for the current frame
            for i in range(N):
                x, y = positions[frame_number, i]
                pygame.draw.circle(screen, (255, 0, 0), (int(x), HEIGHT - int(y)), 5)

            # Update the display
            pygame.display.flip()

            # Go to the next frame
            frame_number += 1
            if frame_number >= len(states):
                frame_number = 0

        pygame.quit()

    def anim_pygame_interactive(self):
        pygame.init()

        # Set up some constants
        WIDTH, HEIGHT = 600, 600
        FPS = 60
        DT = 1.0 / FPS  # Time step

        # Create the Pygame window
        screen = pygame.display.set_mode((WIDTH, HEIGHT))

        # Convert states to integer coordinates
        states = np.array(self.states)
        N = states.shape[1] // 4  # Number of particles
        positions = states[:, :2*N].reshape((-1, N, 2))
        velocities = states[:, 2*N:].reshape((-1, N, 2))

        # Scale positions and velocities to screen size
        positions = positions / self.box_size * np.array([WIDTH, HEIGHT])
        velocities = velocities / self.box_size * np.array([WIDTH, HEIGHT])

        # Variables for controlling simulation speed
        frame_step = 1

        # Variables for selecting a particle
        selected_particle = None

        # Main loop
        running = True
        frame_number = 0
        while running:
            # Limit the frame rate
            pygame.time.Clock().tick(FPS)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        frame_step += 1
                    elif event.key == pygame.K_DOWN:
                        frame_step = max(1, frame_step - 1)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Select a particle
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    for i in range(N):
                        if np.linalg.norm(mouse_pos - positions[frame_number, i]) < 10:
                            selected_particle = i
                            break

            # Clear the screen
            screen.fill((0, 0, 0))

            # Draw the particles for the current frame
            for i in range(N):
                x, y = positions[frame_number, i]
                pygame.draw.circle(screen, (255, 0, 0), (int(x), HEIGHT - int(y)), 5)

            # Display the position and velocity of the selected particle
            if selected_particle is not None:
                pos = positions[frame_number, selected_particle] / np.array([WIDTH, HEIGHT]) * self.box_size
                vel = velocities[frame_number, selected_particle] / np.array([WIDTH, HEIGHT]) * self.box_size
                print(f"Particle {selected_particle}: position = {pos}, velocity = {vel}")

            # Update the display
            pygame.display.flip()

            # Go to the next frame
            frame_number += frame_step
            if frame_number >= len(states):
                frame_number = 0

        pygame.quit()

    def plot(self):
        u = np.array(self.states)
        N = u.shape[1] // 4
        fig, ax = plt.subplots()
        fig.set_size_inches(4,4)
        L = self.box_size
        for i in range(N):
            positions = u[:, i*2]
            ax.plot(self.times, positions, label=f'particle {i+1}')
        ax.set_xlabel('time')
        ax.set_ylabel('x position')
        ax.legend()
        return ax
    
    def calculate_total_energy(self):
        # compute the temperature of the system through time, which is simply the average kinetic energy of all the particles
        u = np.array(self.states)
        N = u.shape[1] // 4 # N used to extract the velocities from the states
        v = u[:, 2*N:] # velocities used in kinetic energy calculation
        kinetic_energy = 0.5*np.sum(v**2, axis=1) # kinetic energy = 1/2 * m * v^2
        # potential_energy = np.zeros(len(self.times))
        # for i in range(N):
        #     for j in range(i+1, N):
        #         r_ij = u[:, j*2:j*2+2] - u[:, i*2:i*2+2]
        #         r = np.linalg.norm(r_ij, axis=1)
        #         potential_energy += 4*self.epsilon*((self.sigma/r)**12 - (self.sigma/r)**6)
        return kinetic_energy# + potential_energy

if __name__ == '__main__':
    # N particles in a LxL box
    N = 10
    box_size = 10
    x0 = np.random.rand(N*2)*box_size
    v0 = np.random.rand(N*2)
    sigma = 1
    epsilon = 1
    dt = 0.01
    t_end = 20
    lj_sim = LJParticleSim(N, box_size, sigma, epsilon, dt, t_end, x0, v0)
    lj_sim.animate()